"""Evaluate a saved checkpoint on the public TPA nine-task suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bmb.transformers_integration import register_bmb


TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "mmlu",
    "sciq",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-fewshot", type=int, choices=[0, 2], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--limit", type=float, help="Optional smoke-test limit")
    args = parser.parse_args()

    register_bmb()
    from lm_eval import simple_evaluate

    results = simple_evaluate(
        model="hf",
        model_args={
            "pretrained": args.checkpoint,
            "device": args.device,
            "dtype": "bfloat16" if args.device.startswith("cuda") else "float32",
        },
        tasks=TASKS,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=str) + "\n")
    scores = []
    for task in TASKS:
        task_result = results["results"][task]
        accuracy = task_result.get("acc_norm,none", task_result.get("acc,none"))
        if accuracy is not None:
            scores.append(accuracy)
    print(f"{args.num_fewshot}-shot unweighted mean: {100 * sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
