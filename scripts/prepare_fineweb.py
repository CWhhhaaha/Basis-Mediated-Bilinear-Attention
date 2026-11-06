"""Tokenize FineWeb-Edu into the binary layout used by the TPA recipe.

The first 100M-token shard is validation; subsequent shards are training data.
Each document starts with GPT-2 <|endoftext|>. Shards contain a 256-int32 header
followed by uint16 tokens.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def write_shard(path: Path, tokens: np.ndarray):
    header = np.zeros(256, dtype=np.int32)
    header[:3] = [20240520, 1, len(tokens)]
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(tokens.astype(np.uint16).tobytes())


def tokenize(document):
    encoder = tiktoken.get_encoding("gpt2")
    tokens = [encoder.eot_token]
    tokens.extend(encoder.encode_ordinary(document["text"]))
    return np.asarray(tokens, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/fineweb-edu100B")
    parser.add_argument("--subset", default="sample-100BT", choices=["sample-10BT", "sample-100BT", "sample-350BT"])
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    parser.add_argument("--max-documents", type=int, help="Optional small data-pipeline smoke test")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name=args.subset, split="train", streaming=True
    )
    if args.max_documents:
        dataset = dataset.take(args.max_documents)

    shard = np.empty(args.shard_size, dtype=np.uint16)
    count = 0
    shard_index = 0
    progress = tqdm(total=args.shard_size, unit="tokens", desc="shard 0")
    with mp.Pool(args.workers) as pool:
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            position = 0
            while position < len(tokens):
                space = args.shard_size - count
                take = min(space, len(tokens) - position)
                shard[count:count + take] = tokens[position:position + take]
                count += take
                position += take
                progress.update(take)
                if count == args.shard_size:
                    split = "val" if shard_index == 0 else "train"
                    write_shard(output / f"fineweb_{split}_{shard_index:06d}.bin", shard)
                    shard_index += 1
                    count = 0
                    progress.close()
                    progress = tqdm(total=args.shard_size, unit="tokens", desc=f"shard {shard_index}")
    if count:
        split = "val" if shard_index == 0 else "train"
        write_shard(output / f"fineweb_{split}_{shard_index:06d}.bin", shard[:count])
    progress.close()
    with (output / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": 50304, "tokenizer": "gpt2"}, handle)


if __name__ == "__main__":
    main()
