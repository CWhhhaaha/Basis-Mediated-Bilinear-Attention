"""Distributed FineWeb-Edu pretraining for the paper's GQA+BMB-UV settings."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from bmb.llama_bmb import BMBConfig, BMBForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-path")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def cosine_lr(step: int, cfg: dict) -> float:
    warmup = cfg["warmup_steps"]
    if step < warmup:
        return cfg["learning_rate"] * (step + 1) / warmup
    decay_steps = cfg["max_steps"]
    if step >= decay_steps:
        return cfg["min_learning_rate"]
    ratio = (step - warmup) / (decay_steps - warmup)
    coefficient = 0.5 * (1 + math.cos(math.pi * ratio))
    return cfg["min_learning_rate"] + coefficient * (
        cfg["learning_rate"] - cfg["min_learning_rate"]
    )


class FineWebShards:
    """Random-access uint16 shards produced by scripts/prepare_fineweb.py."""

    def __init__(self, directory: str, sequence_length: int, seed: int):
        self.sequence_length = sequence_length
        self.header_tokens = 512
        self.random = random.Random(seed)
        root = Path(directory)
        train_files = sorted(root.glob("fineweb_train_*.bin"))
        val_file = root / "fineweb_val_000000.bin"
        if not train_files or not val_file.exists():
            raise FileNotFoundError(
                f"Expected FineWeb train shards and {val_file}; run the data preparation script first"
            )
        self.train = [np.memmap(path, dtype=np.uint16, mode="r") for path in train_files]
        self.val = np.memmap(val_file, dtype=np.uint16, mode="r")

    def batch(self, split: str, batch_size: int, device: torch.device):
        if split == "train":
            data = self.random.choice(self.train)
        else:
            data = self.val
        available = len(data) - self.header_tokens - self.sequence_length - 1
        offsets = torch.randint(available, (batch_size,)).tolist()
        x = torch.stack([
            torch.from_numpy(
                data[self.header_tokens + i:self.header_tokens + i + self.sequence_length].astype(np.int64)
            ) for i in offsets
        ])
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float, config: dict):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step, "best_val": best_val},
        path / "optimizer.pt",
    )
    (path / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.no_compile:
        config["training"]["compile"] = False

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        master = rank == 0
    else:
        rank = local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master = True

    train_cfg = config["training"]
    torch.manual_seed(train_cfg["seed"] + rank)
    np.random.seed(train_cfg["seed"] + rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    data = FineWebShards(
        config["data"]["path"], config["data"]["sequence_length"], train_cfg["seed"] + rank
    )
    model_config = BMBConfig(**config["model"])
    if args.resume:
        model = BMBForCausalLM.from_pretrained(args.resume, config=model_config).to(device)
    else:
        model = BMBForCausalLM(model_config).to(device)
    raw_model = model
    parameter_count = sum(p.numel() for p in model.parameters())
    if master:
        print(json.dumps({"parameters": parameter_count, "world_size": world_size, **config}, indent=2))

    if train_cfg["compile"] and device.type == "cuda":
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["beta1"], train_cfg["beta2"]),
        eps=1e-8,
        weight_decay=train_cfg["weight_decay"],
    )
    start_step = 0
    best_val = float("inf")
    if args.resume:
        state = torch.load(Path(args.resume) / "optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_val = state["best_val"]

    output_dir = Path(config["output_dir"])
    tokens_per_step = (
        train_cfg["micro_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * config["data"]["sequence_length"]
        * world_size
    )
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if train_cfg["bf16"] and device.type == "cuda"
        else nullcontext()
    )

    @torch.no_grad()
    def validation_loss():
        model.eval()
        losses = []
        for _ in range(config["data"]["validation_batches"]):
            inputs = data.batch("val", train_cfg["micro_batch_size"], device)
            with amp_context:
                output = model(inputs, labels=inputs)
            losses.append(output.loss.detach())
        model.train()
        return torch.stack(losses).mean()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, train_cfg["max_steps"]):
        lr = cosine_lr(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % train_cfg["eval_interval"] == 0 and step > start_step:
            val_loss = validation_loss()
            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            if master:
                print(f"step={step} val_loss={val_loss.item():.6f} val_ppl={val_loss.exp().item():.4f}")
                save_checkpoint(output_dir / f"checkpoint-{step}", raw_model, optimizer, step, best_val, config)
                if val_loss.item() < best_val:
                    best_val = val_loss.item()
                    save_checkpoint(output_dir / "best", raw_model, optimizer, step, best_val, config)

        started = time.perf_counter()
        for micro_step in range(train_cfg["gradient_accumulation_steps"]):
            if distributed:
                model.require_backward_grad_sync = (
                    micro_step == train_cfg["gradient_accumulation_steps"] - 1
                )
            inputs = data.batch("train", train_cfg["micro_batch_size"], device)
            with amp_context:
                output = model(inputs, labels=inputs)
                loss = output.loss / train_cfg["gradient_accumulation_steps"]
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if master and (step + 1) % train_cfg["log_interval"] == 0:
            elapsed = time.perf_counter() - started
            print(
                f"step={step + 1} train_loss={output.loss.item():.6f} lr={lr:.7g} "
                f"tokens={(step + 1) * tokens_per_step / 1e9:.4f}B "
                f"tokens/s={tokens_per_step / elapsed:.0f}"
            )

    if master:
        save_checkpoint(output_dir / "final", raw_model, optimizer, train_cfg["max_steps"], best_val, config)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
