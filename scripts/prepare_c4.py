#!/usr/bin/env python3
"""Prepare a full GPT-2 token/chunk cache for C4 realnewslike.

This script converts the raw Hugging Face Arrow dataset into a DatasetDict with
train/val splits of contiguous fixed-length token blocks. It is intended to be
run once on the remote machine and then reused by all GPT-2 attention variants.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import GPT2Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a GPT-2 cache from raw C4 realnewslike")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/c4-realnewslike",
        help="Path to the raw save_to_disk C4 dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/cache/c4-realnewslike-gpt2-1024",
        help="Output path for the tokenized/chunked DatasetDict cache",
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name")
    parser.add_argument("--seq-length", type=int, default=1024, help="Block length")
    parser.add_argument("--val-fraction", type=float, default=0.005, help="Validation fraction")
    parser.add_argument("--num-proc", type=int, default=32, help="Worker processes for map()")
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    return parser.parse_args()


def group_tokenized_examples(examples: dict, seq_length: int) -> dict:
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // seq_length) * seq_length
    if total_length == 0:
        return {k: [] for k in concatenated.keys()}
    return {
        k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
        for k, t in concatenated.items()
    }


def main() -> int:
    args = parse_args()
    input_path = os.path.expanduser(args.input_path)
    output_path = os.path.expanduser(args.output_path)

    if os.path.exists(output_path):
        if not args.overwrite:
            raise SystemExit(f"Output path already exists: {output_path}. Use --overwrite to replace it.")
        import shutil

        shutil.rmtree(output_path)

    print(f"[load] raw dataset from {input_path}")
    ds = load_from_disk(input_path)
    if isinstance(ds, DatasetDict):
        if "train" not in ds:
            raise SystemExit("Expected raw DatasetDict to contain a train split.")
        ds = ds["train"]
    if not isinstance(ds, Dataset):
        raise SystemExit("Expected a raw Dataset.")

    print(f"[raw] rows={len(ds):,} columns={ds.column_names}")
    print(f"[split] val_fraction={args.val_fraction}")
    split = ds.train_test_split(test_size=args.val_fraction, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    print(f"[split] train={len(train_ds):,} val={len(val_ds):,}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    remove_cols = [c for c in ["text", "timestamp", "url"] if c in train_ds.column_names]

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=False, add_special_tokens=False)

    print(f"[tokenize] num_proc={args.num_proc}")
    train_ds = train_ds.map(tokenize_fn, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)
    val_ds = val_ds.map(tokenize_fn, batched=True, num_proc=args.num_proc, remove_columns=remove_cols)

    def group_fn(examples):
        return group_tokenized_examples(examples, args.seq_length)

    print(f"[group] seq_length={args.seq_length}")
    train_ds = train_ds.map(group_fn, batched=True, num_proc=args.num_proc)
    val_ds = val_ds.map(group_fn, batched=True, num_proc=args.num_proc)

    prepared = DatasetDict({"train": train_ds, "val": val_ds})
    print(f"[save] output={output_path}")
    prepared.save_to_disk(output_path)

    meta = {
        "input_path": input_path,
        "output_path": output_path,
        "tokenizer": args.tokenizer,
        "seq_length": args.seq_length,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "num_proc": args.num_proc,
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "columns": train_ds.column_names,
    }
    Path(output_path, "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("[done]")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
