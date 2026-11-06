# Downstream evaluation and systems measurements

## Nine-task downstream evaluation

Install the dependencies:

```bash
python -m pip install -e '.[train,eval]'
```

The evaluation covers ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande, MMLU, and SciQ. Run both shot settings:

```bash
for shot in 0 2; do
  PYTHONPATH=src python -m bmb.evaluate_downstream \
    --checkpoint runs/fineweb_bmbuv_xl/final \
    --num-fewshot "${shot}" \
    --output "results/bmbuv_xl_${shot}shot.json"
done
```

The program registers the custom model with Transformers, calls `lm_eval.simple_evaluate`, saves the result JSON, and prints the unweighted nine-task mean. It prefers `acc_norm,none` and otherwise uses `acc,none`. `--device cuda` uses BF16; CPU uses float32. `--batch-size` defaults to `1`.

Use `--limit` for a short pipeline check:

```bash
PYTHONPATH=src python -m bmb.evaluate_downstream \
  --checkpoint runs/fineweb_bmbuv_small/final \
  --num-fewshot 0 \
  --limit 0.01 \
  --output results/smoke.json
```

## Full-model systems benchmark

`src/bmb/benchmark_systems.py` measures the complete model, including projections and cache writes. It accepts a trained checkpoint or an initialized structural control.

PyTorch SDPA is the default. For the FA4 path used in the paper's systems comparison, install FA4 and pass `--backend fa4` to both architectures:

```bash
python -m pip install -e '.[fa4]'
python -m pip install --pre 'flash-attn-4[cu13]'  # CUDA 13
```

### GQA+BMB-UV XL

```bash
PYTHONPATH=src python -m bmb.benchmark_systems \
  --config configs/fineweb_xl.json \
  --checkpoint runs/fineweb_bmbuv_xl/final \
  --prefill-length 16384 \
  --cache-length 1000000 \
  --decode-steps 128 \
  --warmup 3 \
  --backend fa4 \
  --output results/bmbuv_xl_systems.json
```

### Matched GQA control

```bash
PYTHONPATH=src python -m bmb.benchmark_systems \
  --config configs/fineweb_gqa_xl.json \
  --checkpoint runs/fineweb_gqa_xl/final \
  --prefill-length 16384 \
  --cache-length 1000000 \
  --decode-steps 128 \
  --warmup 3 \
  --backend fa4 \
  --output results/gqa_xl_systems.json
```

Use the same GPU, input lengths, warmup, backend, and step counts for both runs. `--skip-prefill` and `--skip-long-cache` allow separate memory-constrained runs.

## Measurement protocol

- BF16, batch 1, explicit K/V caches.
- 16K prefill executes the full projection/cache-write path and computes only the final LM logits.
- Short-cache decode uses the prefill state copied into preallocated static K/V tensors.
- Long-cache decoding uses synthetic preallocated K/V tensors and 128 timed full-model decode steps after warmup.
- Model/cache initialization is excluded from long-cache timing and from peak allocation measured after cache allocation.
- Logical KV is shape-derived BF16 capacity and is reported separately from measured peak allocation.
- The 1M test measures decoding over an existing cache, not 1M-token prefill.

FA4 requires CUDA FP16/BF16, zero attention dropout, and no padding or custom mask in this adapter. The default SDPA path remains available for other configurations.

## Paper-reported XL reference points

| Measurement | GQA | GQA+BMB-UV |
|---|---:|---:|
| Logical KV at 1M positions | 22.89 GiB | 17.17 GiB |
| 16K prefill | 16.30K tok/s | 20.46K tok/s |
| 16K decode | 34.91 ms/tok | 35.62 ms/tok |
| 1M-cache decode | 1,753.87 ms/tok | 1,334.64 ms/tok |
| Peak allocation at 1M | 25.93 GiB | 19.79 GiB |

These measurements use one RTX 6000D, BF16, batch 1, FA4, and static caches.
