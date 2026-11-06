from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TokenizedDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


def build_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained("bert-base-uncased")


def load_tokenized_cache(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict cache at {path}, got {type(payload)!r}")
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Expected examples list in tokenized cache: {path}")
    return payload


def build_model(model_config: dict[str, Any], tokenizer: BertTokenizerFast) -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(model_config["hidden_size"]),
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        intermediate_size=int(model_config["intermediate_size"]),
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        position_embedding_type="absolute",
    )
    return BertForMaskedLM(config)


def iter_attention_layers(model: BertForMaskedLM) -> Iterable[Any]:
    return model.bert.encoder.layer


def apply_symmetric_query_key_initialization(model: BertForMaskedLM) -> None:
    with torch.no_grad():
        for layer in iter_attention_layers(model):
            attention = layer.attention.self
            attention.key.weight.copy_(attention.query.weight)
            if attention.query.bias is not None and attention.key.bias is not None:
                attention.key.bias.copy_(attention.query.bias)


def compute_layer_asymmetry_ratios(model: BertForMaskedLM) -> list[float]:
    ratios: list[float] = []
    with torch.no_grad():
        for layer in iter_attention_layers(model):
            attention = layer.attention.self
            w_q = attention.query.weight.detach()
            w_k = attention.key.weight.detach()
            gram = w_q @ w_k.T
            asym = 0.5 * (gram - gram.T)
            denom = torch.sum(gram * gram).clamp_min(1e-12)
            ratio = torch.sum(asym * asym) / denom
            ratios.append(float(ratio.item()))
    return ratios


def compute_reg_loss(
    model: BertForMaskedLM,
    intervals: list[dict[str, float]],
    penalty: str = "squared_hinge",
) -> tuple[torch.Tensor, list[float]]:
    penalties: list[torch.Tensor] = []
    ratios: list[float] = []
    for layer, interval in zip(iter_attention_layers(model), intervals):
        attention = layer.attention.self
        w_q = attention.query.weight
        w_k = attention.key.weight
        gram = w_q @ w_k.T
        asym = 0.5 * (gram - gram.T)
        denom = torch.sum(gram * gram).clamp_min(1e-12)
        ratio = torch.sum(asym * asym) / denom
        lower = torch.relu(torch.tensor(interval["rho_min"], device=ratio.device) - ratio)
        upper = torch.relu(ratio - torch.tensor(interval["rho_max"], device=ratio.device))
        if penalty == "linear_hinge":
            penalties.append(lower + upper)
        elif penalty == "squared_hinge":
            penalties.append(lower.square() + upper.square())
        else:
            raise ValueError(f"Unsupported regularization penalty: {penalty}")
        ratios.append(float(ratio.detach().item()))

    if not penalties:
        zero = next(model.parameters()).new_zeros(())
        return zero, ratios

    return torch.stack(penalties).mean(), ratios


def regularization_active(epoch: int, schedule: list[dict[str, Any]]) -> bool:
    for item in schedule:
        start = int(item["epoch_start"])
        end = int(item["epoch_end"])
        every = item.get("every_n_epochs")
        if start <= epoch <= end and every:
            return (epoch - start) % int(every) == 0
    return False


def count_reg_flops(hidden_size: int, num_layers: int) -> int:
    matrix_mul = 2 * (hidden_size**3)
    aux = 8 * (hidden_size**2)
    return num_layers * (matrix_mul + aux)


def count_analysis_flops(hidden_size: int, num_layers: int) -> int:
    return count_reg_flops(hidden_size, num_layers)


def resolve_max_length(training_config: dict[str, Any]) -> int:
    return int(training_config.get("max_seq_length", 128))


def resolve_num_workers(training_config: dict[str, Any]) -> int:
    return int(training_config.get("num_workers", 2))


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def estimate_model_flops(model: BertForMaskedLM, batch: dict[str, torch.Tensor]) -> tuple[int | None, str]:
    try:
        flops = model.floating_point_ops(batch)
    except Exception:
        return None, "unavailable"
    return int(flops), "transformers.floating_point_ops"


@dataclass
class RunPaths:
    output_dir: Path
    metrics_path: Path
    epoch_metrics_path: Path
    ratio_path: Path
    summary_path: Path
    model_dir: Path


def prepare_run_paths(output_dir: Path) -> RunPaths:
    analysis_dir = output_dir / "analysis"
    model_dir = output_dir / "model"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        output_dir=output_dir,
        metrics_path=output_dir / "metrics.json",
        epoch_metrics_path=analysis_dir / "epoch_metrics.json",
        ratio_path=analysis_dir / "layer_asymmetry_by_epoch.json",
        summary_path=output_dir / "run_summary.json",
        model_dir=model_dir,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def format_epoch_summary(
    *,
    experiment_name: str,
    epoch: int,
    total_epochs: int,
    total_loss: float,
    task_loss: float,
    reg_loss: float | None,
    learning_rate: float,
    training_time_sec: float,
    eval_time_sec: float | None,
    eval_loss: float | None,
    analysis_time_sec: float,
    reg_enabled: bool,
    layer_ratios: list[float],
) -> str:
    ratios = ", ".join(f"L{i + 1}={value:.4f}" for i, value in enumerate(layer_ratios))
    reg_text = "n/a" if reg_loss is None else f"{reg_loss:.6f}"
    return (
        f"[epoch_summary] experiment={experiment_name} "
        f"epoch={epoch}/{total_epochs} "
        f"total_loss={total_loss:.6f} "
        f"task_loss={task_loss:.6f} "
        f"reg_loss={reg_text} "
        f"reg_active={'yes' if reg_enabled else 'no'} "
        f"lr={learning_rate:.6e} "
        f"train_sec={training_time_sec:.2f} "
        f"eval_sec={'n/a' if eval_time_sec is None else f'{eval_time_sec:.2f}'} "
        f"eval_loss={'n/a' if eval_loss is None else f'{eval_loss:.6f}'} "
        f"analysis_sec={analysis_time_sec:.4f} "
        f"ratios=[{ratios}]"
    )


def format_init_summary(
    *,
    experiment_name: str,
    total_epochs: int,
    learning_rate: float,
    analysis_time_sec: float,
    layer_ratios: list[float],
) -> str:
    ratios = ", ".join(f"L{i + 1}={value:.4f}" for i, value in enumerate(layer_ratios))
    return (
        f"[init_summary] experiment={experiment_name} "
        f"epoch=0/{total_epochs} "
        f"lr={learning_rate:.6e} "
        f"analysis_sec={analysis_time_sec:.4f} "
        f"ratios=[{ratios}]"
    )


def evaluate_model(
    model: BertForMaskedLM,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float | None, float]:
    losses: list[float] = []
    model.eval()
    maybe_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(float(outputs.loss.item()))
    maybe_sync(device)
    duration = time.perf_counter() - start
    return (safe_mean(losses) if losses else None), duration


def train_experiment(
    config: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    training = config["training"]
    dataset_config = config["dataset"]
    model_config = config["model"]
    regularization = config.get("regularization", {})
    instrumentation = config.get("instrumentation", {})

    set_seed(int(config.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer()
    model = build_model(model_config, tokenizer)
    if config.get("initialization", {}).get("query_key") == "symmetric":
        apply_symmetric_query_key_initialization(model)
    model.to(device)

    train_cache_path = Path(dataset_config["resolved_paths"]["tokenized_train_file"])
    test_cache_path = Path(dataset_config["resolved_paths"]["tokenized_test_file"])
    num_workers = resolve_num_workers(training)

    train_cache = load_tokenized_cache(train_cache_path)
    test_cache = load_tokenized_cache(test_cache_path)
    train_dataset = TokenizedDataset(train_cache["examples"])
    test_dataset = TokenizedDataset(test_cache["examples"])
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(training["mlm_probability"]),
    )

    per_device_batch_size = int(training["per_device_batch_size"])
    grad_accum_steps = int(training["gradient_accumulation_steps"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        persistent_workers=num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    total_update_steps = int(training.get("estimated_total_steps") or 0)
    if total_update_steps <= 0:
        total_update_steps = math.ceil(len(train_loader) / grad_accum_steps) * int(training["max_epochs"])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=max(total_update_steps, 1),
    )

    run_paths = prepare_run_paths(output_dir)

    warmup_steps = int(training.get("warmup_steps", 0))
    eval_every_epochs = int(training.get("eval_every_epochs", 1))
    intervals = regularization.get("intervals", {}).get("layers", []) if regularization.get("enabled") else []
    reg_schedule = regularization.get("schedule", [])
    lambda_value = float(regularization.get("lambda", 0.0))
    penalty = str(regularization.get("penalty", "squared_hinge"))

    batch_example = next(iter(train_loader))
    batch_example = {key: value.to(device) for key, value in batch_example.items()}
    task_flops_per_microbatch, flops_method = estimate_model_flops(model, batch_example)
    hidden_size = int(model_config["hidden_size"])
    num_layers = int(model_config["num_hidden_layers"])
    reg_flops_per_microbatch = count_reg_flops(hidden_size, num_layers)
    analysis_flops_per_epoch = count_analysis_flops(hidden_size, num_layers)

    optimizer.zero_grad(set_to_none=True)

    epoch_metrics: list[dict[str, Any]] = []
    ratio_history: list[dict[str, Any]] = []

    global_microbatches = 0
    global_optimizer_steps = 0
    total_training_time = 0.0
    total_eval_time = 0.0
    total_analysis_time = 0.0
    total_task_flops = 0
    total_reg_flops = 0
    total_analysis_flops = 0

    maybe_sync(device)
    init_analysis_start = time.perf_counter()
    initial_layer_ratios = compute_layer_asymmetry_ratios(model)
    maybe_sync(device)
    init_analysis_time = time.perf_counter() - init_analysis_start
    total_analysis_time += init_analysis_time
    total_analysis_flops += analysis_flops_per_epoch

    ratio_history.append(
        {
            "epoch": 0,
            "layer_asymmetry_ratio": [round(value, 8) for value in initial_layer_ratios],
        }
    )
    epoch_metrics.append(
        {
            "epoch": 0,
            "regularization_active": False,
            "avg_task_loss": None,
            "avg_total_loss": None,
            "avg_reg_loss": None,
            "training_time_sec": 0.0,
            "evaluation_time_sec": None,
            "eval_loss": None,
            "analysis_time_sec": round(init_analysis_time, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "layer_asymmetry_ratio": [round(value, 8) for value in initial_layer_ratios],
        }
    )
    print(
        format_init_summary(
            experiment_name=config.get("experiment_name", "remote-lab-train"),
            total_epochs=int(training["max_epochs"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            analysis_time_sec=init_analysis_time,
            layer_ratios=initial_layer_ratios,
        ),
        flush=True,
    )

    use_tqdm = sys.stdout.isatty()
    progress = tqdm(
        range(1, int(training["max_epochs"]) + 1),
        desc=config.get("experiment_name", "remote-lab-train"),
        disable=not use_tqdm,
    )

    for epoch in progress:
        model.train()
        epoch_total_losses: list[float] = []
        epoch_task_losses: list[float] = []
        epoch_reg_losses: list[float] = []
        reg_enabled = bool(regularization.get("enabled")) and regularization_active(epoch, reg_schedule)
        eval_loss: float | None = None
        eval_time_sec: float | None = None

        maybe_sync(device)
        epoch_train_start = time.perf_counter()

        for microbatch_idx, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(**batch)
            task_loss = outputs.loss
            total_loss = task_loss

            reg_loss_value = 0.0
            if reg_enabled:
                reg_loss, _ = compute_reg_loss(model, intervals, penalty=penalty)
                reg_loss_value = float(reg_loss.detach().item())
                total_loss = total_loss + lambda_value * reg_loss
                total_reg_flops += reg_flops_per_microbatch

            scaled_loss = total_loss / grad_accum_steps
            scaled_loss.backward()

            if microbatch_idx % grad_accum_steps == 0 or microbatch_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_optimizer_steps += 1
                if global_optimizer_steps > warmup_steps:
                    scheduler.step()

            global_microbatches += 1
            if task_flops_per_microbatch is not None:
                total_task_flops += task_flops_per_microbatch

            epoch_task_losses.append(float(task_loss.detach().item()))
            epoch_total_losses.append(float(total_loss.detach().item()))
            if reg_enabled:
                epoch_reg_losses.append(reg_loss_value)

        maybe_sync(device)
        epoch_train_time = time.perf_counter() - epoch_train_start
        total_training_time += epoch_train_time

        maybe_sync(device)
        analysis_start = time.perf_counter()
        layer_ratios = compute_layer_asymmetry_ratios(model)
        maybe_sync(device)
        analysis_time = time.perf_counter() - analysis_start
        total_analysis_time += analysis_time
        total_analysis_flops += analysis_flops_per_epoch

        if eval_every_epochs > 0 and epoch % eval_every_epochs == 0:
            eval_loss, eval_duration = evaluate_model(model, test_loader, device)
            total_eval_time += eval_duration
            eval_time_sec = eval_duration

        ratio_history.append(
            {
                "epoch": epoch,
                "layer_asymmetry_ratio": [round(value, 8) for value in layer_ratios],
            }
        )

        epoch_metrics.append(
            {
                "epoch": epoch,
                "regularization_active": reg_enabled,
                "avg_task_loss": round(safe_mean(epoch_task_losses), 8),
                "avg_total_loss": round(safe_mean(epoch_total_losses), 8),
                "avg_reg_loss": round(safe_mean(epoch_reg_losses), 8) if epoch_reg_losses else None,
                "training_time_sec": round(epoch_train_time, 6),
                "evaluation_time_sec": round(eval_time_sec, 6) if eval_time_sec is not None else None,
                "eval_loss": round(eval_loss, 8) if eval_loss is not None else None,
                "analysis_time_sec": round(analysis_time, 6),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "layer_asymmetry_ratio": [round(value, 8) for value in layer_ratios],
            }
        )

        avg_total_loss = safe_mean(epoch_total_losses)
        avg_task_loss = safe_mean(epoch_task_losses)
        avg_reg_loss = safe_mean(epoch_reg_losses) if epoch_reg_losses else None

        if use_tqdm:
            progress.set_postfix(
                loss=f"{avg_total_loss:.4f}",
                reg="on" if reg_enabled else "off",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        print(
            format_epoch_summary(
                experiment_name=config.get("experiment_name", "remote-lab-train"),
                epoch=epoch,
                total_epochs=int(training["max_epochs"]),
                total_loss=avg_total_loss,
                task_loss=avg_task_loss,
                reg_loss=avg_reg_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                training_time_sec=epoch_train_time,
                eval_time_sec=eval_time_sec,
                eval_loss=eval_loss,
                analysis_time_sec=analysis_time,
                reg_enabled=reg_enabled,
                layer_ratios=layer_ratios,
            ),
            flush=True,
        )

    final_eval_loss = next(
        (row["eval_loss"] for row in reversed(epoch_metrics) if row.get("eval_loss") is not None),
        None,
    )
    if final_eval_loss is None:
        final_eval_loss, eval_duration = evaluate_model(model, test_loader, device)
        total_eval_time += eval_duration

    model.save_pretrained(run_paths.model_dir)
    tokenizer.save_pretrained(run_paths.model_dir)

    metrics = {
        "experiment_name": config.get("experiment_name"),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tokenized_train_cache": str(train_cache_path),
        "tokenized_test_cache": str(test_cache_path),
        "epochs": int(training["max_epochs"]),
        "global_microbatches": global_microbatches,
        "global_optimizer_steps": global_optimizer_steps,
        "final_train_loss": epoch_metrics[-1]["avg_total_loss"] if epoch_metrics else None,
        "final_eval_loss": round(float(final_eval_loss), 8) if final_eval_loss is not None else None,
        "training_time_sec": round(total_training_time, 6),
        "evaluation_time_sec": round(total_eval_time, 6),
        "analysis_time_sec": round(total_analysis_time, 6),
        "task_flops": total_task_flops if instrumentation.get("measure_flops", False) else None,
        "reg_flops": total_reg_flops if instrumentation.get("measure_flops", False) else None,
        "analysis_flops": total_analysis_flops if instrumentation.get("measure_flops", False) else None,
        "reported_training_flops": (
            total_task_flops + total_reg_flops if instrumentation.get("exclude_analysis_flops", True) else total_task_flops + total_reg_flops + total_analysis_flops
        )
        if instrumentation.get("measure_flops", False)
        else None,
        "reported_training_time_sec": (
            total_training_time if instrumentation.get("exclude_analysis_time", True) else total_training_time + total_analysis_time
        ),
        "flops_estimation_method": flops_method,
    }

    summary = {
        "config": config,
        "metrics": metrics,
        "epoch_metrics_path": str(run_paths.epoch_metrics_path),
        "ratio_history_path": str(run_paths.ratio_path),
        "model_dir": str(run_paths.model_dir),
    }

    write_json(run_paths.metrics_path, metrics)
    write_json(run_paths.epoch_metrics_path, epoch_metrics)
    write_json(run_paths.ratio_path, ratio_history)
    write_json(run_paths.summary_path, summary)

    return summary
