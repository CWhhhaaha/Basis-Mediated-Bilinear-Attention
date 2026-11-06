# FineWeb-Edu training with the public TPA recipe

## What comes from TPA, and what is BMB?

The FineWeb-Edu experiments follow the public TPA data and optimization protocol. TPA and GQA results are external reference values reported in the paper; this repository provides our GQA and GQA+BMB-UV implementations and training entry points.

The attention replacement is localized:

$$
Z=XB,\qquad Q_h=ZU_h^Q,\qquad K_j=ZU_j^K,\qquad V_j=XW_j^V.
$$

Here $h$ indexes query heads and $j$ indexes shared KV heads.

Everything outside QK follows the Llama-style TPA topology: tied GPT-2 embeddings, RoPE, parameter-free RMSNorm, SwiGLU, no biases, and `o_proj` input width `Hq*dv`.

## Exact configurations

| Setting | Small | XL |
|---|---:|---:|
| Config | `configs/fineweb_small.json` | `configs/fineweb_xl.json` |
| Layers | 12 | 48 |
| Model width | 768 | 1,600 |
| Query heads / KV heads | 22 / 2 | 48 / 2 |
| Latent width $r$ | 64 | 128 |
| Q/K width $s$ | 32 | 32 |
| Value width $d_v$ | 64 | 64 |
| MLP hidden width | 2,048 | 4,266 |
| Total parameters | 110,592,000 | 1,328,793,600 |
| Context | 1,024 | 1,024 |
| GPUs in paper recipe | 4×A800 | 24×A800 |
| Micro-batch per GPU | 24 | 6 |
| Gradient accumulation | 5 | 10 |
| Global batch (sequences) | 480 | 1,440 |
| Optimizer updates | 100,000 | 33,334 |
| Training tokens | 49,152,000,000 | 49,152,983,040 |
| Peak LR | 6e-4 | 1e-4 |
| Minimum LR | 3e-5 | 1e-5 |
| Warmup updates | 2,000 | 667 |

Common settings: AdamW, weight decay 0.1, betas `(0.9, 0.95)`, epsilon 1e-8, gradient clipping 1.0, cosine schedule, BF16, and seed 5000. Evaluation and checkpointing run every 1,000 updates; progress is printed every 10 updates. Configs enable `torch.compile` on CUDA.

Small matches the public TPA batch construction. XL is a **token-budget-matched external comparison**, not an identical optimizer trajectory to TPA XL: it uses a 3× larger global batch and one third as many updates.

The token calculation is

$$
N_{\mathrm{tokens}}=
N_{\mathrm{GPUs}}\times B_{\mathrm{micro}}\times
N_{\mathrm{accumulation}}\times T\times N_{\mathrm{updates}}.
$$

Changing GPU count without adjusting the other terms changes tokens per update and no longer reproduces the stated recipe.

## Prepare data

```bash
bash scripts/prepare_fineweb.sh
```

The script streams FineWeb-Edu `sample-100BT`, adds GPT-2 `<|endoftext|>` before each document, and writes uint16 shards. See [data preparation](data_preparation.md) for the exact layout.

## Train Small

On one four-GPU node:

```bash
SCALE=small NPROC_PER_NODE=4 bash scripts/train_fineweb.sh
```

Equivalent direct entry point:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=4 \
  -m bmb.train_fineweb --config configs/fineweb_small.json
```

## Train XL on 24 GPUs

For three 8-GPU nodes, run the following on each node with `NODE_RANK=0`, `1`, and `2`, respectively:

```bash
SCALE=xl \
NNODES=3 NPROC_PER_NODE=8 NODE_RANK=0 \
MASTER_ADDR=<rank0-address> MASTER_PORT=29500 \
bash scripts/train_fineweb.sh
```

`MASTER_ADDR` and `MASTER_PORT` must be identical across nodes. `NODE_RANK` must be unique. The launcher uses `torchrun` and the training program uses NCCL on CUDA.

On a 24-GPU node, `SCALE=xl bash scripts/train_fineweb.sh` defaults to 24 local workers. For any other topology, set `NPROC_PER_NODE` explicitly.

## Matched GQA controls

`configs/fineweb_gqa_small.json` and `configs/fineweb_gqa_xl.json` select `GQAAttention` while retaining the same non-QK topology, data, and optimization settings. Their exact parameter counts are 123,568,128 and 1,554,892,800. Launch them with `SCALE=gqa_small` or `SCALE=gqa_xl` and the same world size as the corresponding BMB-UV run.

## Overrides and resume

Useful overrides:

```bash
SCALE=small NPROC_PER_NODE=1 bash scripts/train_fineweb.sh \
  --data-path /path/to/shards \
  --output-dir runs/smoke \
  --max-steps 2 \
  --no-compile
```

Resume from a checkpoint directory containing both model weights and `optimizer.pt`:

```bash
SCALE=small NPROC_PER_NODE=4 bash scripts/train_fineweb.sh \
  --resume runs/fineweb_bmbuv_small/checkpoint-50000
```

The resume path restores model weights, optimizer state, update index, and best validation loss. Checkpoint directories also contain `recipe.json`. `best` and `final` use the same format.
