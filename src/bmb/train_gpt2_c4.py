#!/usr/bin/env python3
"""Train GPT-2 small (124M) on C4 realnewslike with custom attention variants.

Example:
    python -m bmb.train_gpt2_c4 \
        --variant baseline --output-dir runs/gpt2_c4_baseline_50k

    python -m bmb.train_gpt2_c4 \
        --variant lowrank --rank 32 --output-dir runs/gpt2_c4_lowrank_r32_50k

    python -m bmb.train_gpt2_c4 \
        --variant bmbuv --rank 32 --factor-rank 32 \
        --output-dir runs/gpt2_c4_bmbuv_r32s32_50k
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from itertools import chain

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from bmb.attention_accounting import attention_forward_accounting
from bmb.gpt2_attention_variants import replace_gpt2_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 on C4 with attention variants")
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "lowrank", "fullyshared", "bbt", "bmb", "bmbuv", "partialshared"],
        help="Attention variant to use",
    )
    parser.add_argument("--rank", type=int, default=32, help="Rank for lowrank/bmbuv")
    parser.add_argument("--factor-rank", type=int, default=None, help="Factor rank for bmbuv (defaults to rank)")
    parser.add_argument("--shared-dim", type=int, default=728, help="Shared qk dim for partialshared")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/cache/c4-realnewslike-gpt2-1024",
        help="Path to a prepared C4 RealNewsLike GPT-2 cache, or a raw dataset directory.",
    )
    parser.add_argument("--num-proc", type=int, default=8, help="Worker processes for tokenization/preprocessing")
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes during training/eval. Use 0 for safest multi-run stability.",
    )
    parser.add_argument("--max-steps", type=int, default=50000, help="Total training steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--per-device-batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Linear warmup steps")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bf16 mixed precision")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16", help="Disable bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Use gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


def load_or_download_c4(dataset_path: str):
    dataset_path = os.path.expanduser(dataset_path)
    if os.path.exists(dataset_path):
        print(f"Loading C4 from {dataset_path}")
        ds = load_from_disk(dataset_path)
    else:
        print(f"Dataset not found at {dataset_path}. Downloading from HuggingFace...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=False)
        ds.save_to_disk(dataset_path)
        print(f"Saved to {dataset_path}")
    return ds


def group_tokenized_examples(examples: dict, seq_length: int) -> dict:
    """Group tokenized examples into fixed-length contiguous chunks."""
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // seq_length) * seq_length
    if total_length == 0:
        return {k: [] for k in concatenated.keys()}
    return {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated.items()
    }


def is_prepared_cache(ds: Dataset | DatasetDict) -> bool:
    if isinstance(ds, DatasetDict):
        if "train" not in ds:
            return False
        sample_split = ds["train"]
    else:
        sample_split = ds
    cols = set(sample_split.column_names)
    return "input_ids" in cols


def prepare_raw_c4(
    ds: Dataset,
    tokenizer: GPT2Tokenizer,
    seq_length: int,
    seed: int,
    num_proc: int,
) -> DatasetDict:
    ds = ds.train_test_split(test_size=0.005, seed=seed)
    train_ds = ds["train"]
    val_ds = ds["test"]
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    remove_cols = [c for c in ["text", "timestamp", "url"] if c in train_ds.column_names]

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, add_special_tokens=False)

    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=num_proc, remove_columns=remove_cols)
    val_ds = val_ds.map(tokenize_fn, batched=True, num_proc=num_proc, remove_columns=remove_cols)

    def group_fn(examples):
        return group_tokenized_examples(examples, seq_length)

    train_ds = train_ds.map(group_fn, batched=True, num_proc=num_proc)
    val_ds = val_ds.map(group_fn, batched=True, num_proc=num_proc)
    return DatasetDict({"train": train_ds, "val": val_ds})


def load_training_splits(
    dataset_path: str,
    tokenizer: GPT2Tokenizer,
    seq_length: int,
    seed: int,
    num_proc: int,
) -> tuple[Dataset, Dataset]:
    ds = load_or_download_c4(dataset_path)
    if is_prepared_cache(ds):
        print("Detected prepared tokenized cache.")
        if isinstance(ds, DatasetDict):
            train_ds = ds["train"]
            if "val" in ds:
                val_ds = ds["val"]
            elif "validation" in ds:
                val_ds = ds["validation"]
            elif "test" in ds:
                val_ds = ds["test"]
            else:
                raise ValueError("Prepared cache found, but no val/validation/test split exists.")
        else:
            raise ValueError("Prepared cache must be stored as a DatasetDict with train/val splits.")
    else:
        if not isinstance(ds, Dataset):
            if "train" in ds:
                ds = ds["train"]
            else:
                raise ValueError("Raw dataset must contain a train split or be a Dataset.")
        print("Detected raw text dataset. Tokenizing and grouping on the fly.")
        prepared = prepare_raw_c4(ds, tokenizer, seq_length, seed, num_proc)
        train_ds = prepared["train"]
        val_ds = prepared["val"]

    print(f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}")
    return train_ds, val_ds


def compute_parameter_summary(model: GPT2LMHeadModel) -> dict:
    """Count total and attention-specific parameters."""
    total = sum(p.numel() for p in model.parameters())
    attn = 0
    for name, p in model.named_parameters():
        if "attn" in name:
            attn += p.numel()
    return {"total_params": total, "attention_params": attn, "non_attention_params": total - attn}


def compute_attention_theory_summary(args: argparse.Namespace, config: GPT2Config) -> dict:
    summary = attention_forward_accounting(
        str(args.variant),
        model_dim=int(config.n_embd),
        num_heads=int(config.n_head),
        num_layers=int(config.n_layer),
        tokens=int(args.seq_length),
        rank=getattr(args, "rank", None),
        factor_rank=getattr(args, "factor_rank", None),
        shared_dim=getattr(args, "shared_dim", None),
    )
    return {
        "variant": summary["variant"],
        "num_layers": summary["num_layers"],
        "num_heads": summary["num_heads"],
        "model_dim": summary["model_dim"],
        "head_dim": summary["head_dim"],
        "seq_length": summary["tokens"],
        "per_layer_qk_params": summary["qk_params_per_layer"],
        "per_layer_attention_params": summary["attention_params_per_layer"],
        "total_qk_params": summary["qk_params_total"],
        "total_attention_params": summary["attention_params_total"],
        "per_example_qk_flops_per_layer": summary["qk_flops_per_layer"],
        "per_example_attention_flops_per_layer": summary["attention_flops_per_layer"],
        "per_example_qk_flops_total": summary["qk_flops_total"],
        "per_example_attention_flops_total": summary["attention_flops_total"],
    }


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    """Build TrainingArguments while tolerating minor API drift across transformers versions."""
    kwargs = {
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch_fused",
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": "cosine",
        "bf16": args.bf16,
        "fp16": not args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "seed": args.seed,
        "report_to": "none",
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
    }

    sig = inspect.signature(TrainingArguments.__init__).parameters

    if "overwrite_output_dir" in sig:
        kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig:
        kwargs["eval_strategy"] = "steps"

    if "logging_strategy" in sig:
        kwargs["logging_strategy"] = "steps"

    if "save_strategy" in sig:
        kwargs["save_strategy"] = "steps"

    supported = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**supported)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MetricsJsonlCallback(TrainerCallback):
    """Persist train/eval metrics incrementally for mid-run analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_path = os.path.join(output_dir, "metrics_train.jsonl")
        self.eval_path = os.path.join(output_dir, "metrics_eval.jsonl")

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        for path in (self.train_path, self.eval_path):
            with open(path, "a", encoding="utf-8"):
                pass
        return control

    def _write(self, path: str, payload: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        payload = {
            "timestamp_utc": _utc_now_iso(),
            "step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: v for k, v in logs.items()},
        }
        target = self.eval_path if any(k.startswith("eval_") for k in logs) else self.train_path
        self._write(target, payload)
        return control


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds, val_ds = load_training_splits(
        args.dataset_path,
        tokenizer=tokenizer,
        seq_length=args.seq_length,
        seed=args.seed,
        num_proc=args.num_proc,
    )

    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    variant_kwargs = {}
    if args.variant in {"lowrank", "bbt", "bmb"}:
        variant_kwargs = {"rank": args.rank}
    elif args.variant == "bmbuv":
        variant_kwargs = {"rank": args.rank, "factor_rank": args.factor_rank or args.rank}
    elif args.variant == "partialshared":
        variant_kwargs = {"shared_dim": args.shared_dim}

    if args.variant != "baseline":
        replace_gpt2_attention(model, args.variant, **variant_kwargs)
        print(f"Replaced attention with variant: {args.variant}, kwargs: {variant_kwargs}")

    param_info = compute_parameter_summary(model)
    attention_theory_summary = compute_attention_theory_summary(args, config)
    print(f"Total params: {param_info['total_params']:,}")
    print(f"Attention params: {param_info['attention_params']:,}")
    print(
        "Per-layer QK params:",
        f"{attention_theory_summary['per_layer_qk_params']:,}",
        "| Per-example attention FLOPs/layer:",
        f"{attention_theory_summary['per_example_attention_flops_per_layer']:,}",
    )

    effective_batch = args.per_device_batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch}")

    training_args = build_training_arguments(args)

    os.makedirs(args.output_dir, exist_ok=True)
    run_metadata = {
        "timestamp_utc": _utc_now_iso(),
        "variant": args.variant,
        "variant_kwargs": variant_kwargs,
        "param_info": param_info,
        "attention_theory_summary": attention_theory_summary,
        "training_args": training_args.to_dict(),
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=[MetricsJsonlCallback(args.output_dir)],
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")

    train_metrics = dict(train_result.metrics)
    train_runtime = float(train_metrics.get("train_runtime", 0.0) or 0.0)
    train_steps = int(train_metrics.get("global_step", args.max_steps) or args.max_steps)
    tokens_per_step = (
        args.per_device_batch_size
        * args.gradient_accumulation_steps
        * args.seq_length
    )
    total_train_tokens = tokens_per_step * train_steps
    total_eval_tokens = len(val_ds) * args.seq_length

    efficiency_summary = {
        "tokens_per_step": int(tokens_per_step),
        "total_train_tokens": int(total_train_tokens),
        "train_runtime_sec": train_runtime,
        "step_wall_clock_sec": (train_runtime / train_steps) if train_steps > 0 and train_runtime > 0 else None,
        "train_tokens_per_sec": (total_train_tokens / train_runtime) if train_runtime > 0 else None,
        "eval_runtime_sec": float(eval_results.get("eval_runtime", 0.0) or 0.0),
        "eval_tokens_per_sec": (
            total_eval_tokens / float(eval_results.get("eval_runtime", 0.0))
            if float(eval_results.get("eval_runtime", 0.0) or 0.0) > 0
            else None
        ),
        "train_steps_per_sec": train_metrics.get("train_steps_per_second"),
        "train_samples_per_sec": train_metrics.get("train_samples_per_second"),
        "eval_steps_per_sec": eval_results.get("eval_steps_per_second"),
        "eval_samples_per_sec": eval_results.get("eval_samples_per_second"),
        "peak_cuda_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
        ),
        "peak_cuda_memory_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024**2) if torch.cuda.is_available() else None
        ),
    }

    summary = {
        "variant": args.variant,
        "variant_kwargs": variant_kwargs,
        "param_info": param_info,
        "attention_theory_summary": attention_theory_summary,
        "train_metrics": train_metrics,
        "eval_loss": eval_results["eval_loss"],
        "eval_metrics": eval_results,
        "perplexity": perplexity,
        "efficiency_summary": efficiency_summary,
        "training_args": training_args.to_dict(),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
