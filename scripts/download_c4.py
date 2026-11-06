#!/usr/bin/env python3
"""Download C4 RealNewsLike and save the raw Hugging Face dataset locally."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        default="data/raw/c4-realnewslike",
        help="Directory passed to Dataset.save_to_disk",
    )
    parser.add_argument("--dataset", default="allenai/c4", help="Hugging Face dataset repository")
    parser.add_argument("--config", default="realnewslike", help="Dataset configuration")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Optional worker count for save_to_disk",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(os.path.expanduser(args.output_path))
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output path already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {args.dataset} config={args.config} split={args.split}")
    dataset = load_dataset(args.dataset, args.config, split=args.split)
    print(f"[save] rows={len(dataset):,} output={output_path}")
    dataset.save_to_disk(str(output_path), num_proc=args.num_proc)

    metadata = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "rows": len(dataset),
        "columns": dataset.column_names,
    }
    output_path.joinpath("download_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
