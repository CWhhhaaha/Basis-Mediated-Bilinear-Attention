# Reproduction guide

The repository includes the Attention API, model configurations, data-preparation scripts, training launchers, downstream evaluation, and systems benchmarks used for the paper.

## Local checks

```bash
python -m pip install -e '.[train,test]'
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python examples/plug_and_play.py
PYTHONPATH=src python examples/functional_attention.py
PYTHONPATH=src python examples/cache_savings.py
```

The complete suite checks attention outputs, gradients, masks, RoPE, cache decoding, chunked prefill, GPT-2 integration, and Hugging Face generation. FA4 adapter tests use a mocked entry point; FA4 execution requires a compatible CUDA GPU and package installation.

## Paper experiments

### GPT-2/C4

```bash
python scripts/download_c4.py
python scripts/prepare_c4.py
METHOD=bmbuv32 GPU_ID=0 bash scripts/run_gpt2_c4.sh
```

Set `METHOD` to `baseline`, `fullyshared`, `partshr728`, `lowrank32`, `bbt64`, `bmbuv32`, `bmbuv64`, or `bmb64`. The shared settings are in `configs/gpt2_c4.yaml`; `scripts/launch_gpt2_c4_8way.sh` launches the eight-method comparison on eight GPUs.

### ViT/ImageNet-1K

Prepare the official ILSVRC2012 data as described in [data_preparation.md](data_preparation.md), then run:

```bash
METHOD=bmbuv64 GPU_ID=0 bash scripts/run_imagenet.sh
```

Per-method JSON files are in `configs/experiments/`. `scripts/launch_imagenet_8way.sh` launches the eight-method comparison on eight GPUs.

### FineWeb-Edu Small and XL

```bash
bash scripts/download_fineweb.sh
SCALE=small NPROC_PER_NODE=4 bash scripts/train_fineweb.sh
```

For the XL recipe, use `SCALE=xl` and the distributed settings in [tpa_fineweb_recipe.md](tpa_fineweb_recipe.md). Matched GQA controls are selected with `SCALE=gqa_small` or `SCALE=gqa_xl`. Model and optimization settings are stored in `configs/fineweb_*.json`.

### Downstream evaluation

```bash
python -m pip install -e '.[train,eval]'
PYTHONPATH=src python -m bmb.evaluate_downstream \
  --checkpoint runs/fineweb_bmbuv_xl/final \
  --num-fewshot 0 \
  --output results/downstream_0shot.json
```

Use `--num-fewshot 2` for the second evaluation setting. The script evaluates ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande, MMLU, and SciQ, then prints their unweighted mean.

### Systems benchmark

```bash
PYTHONPATH=src python -m bmb.benchmark_systems \
  --config configs/fineweb_xl.json \
  --checkpoint runs/fineweb_bmbuv_xl/final \
  --prefill-length 16384 \
  --cache-length 1000000 \
  --decode-steps 128 \
  --backend fa4 \
  --output results/systems_xl.json
```

Run the matched GQA configuration with the same backend and measurement settings. Omit `--backend fa4` to use PyTorch SDPA. See [evaluation_and_systems.md](evaluation_and_systems.md) for the cache protocol.

## Accounting convention

`src/bmb/attention_accounting.py` implements the paper's symbolic QK parameter and FLOP definitions for the eight-method comparisons. Matrix multiplication and accumulation count as two FLOPs; bias operations, softmax, masks, dropout, nonlinearities, and reshapes are excluded. These symbolic counts are separate from device runtime and profiler measurements.
