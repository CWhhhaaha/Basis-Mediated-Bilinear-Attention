#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
METHOD="${METHOD:-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

case "${METHOD}" in
  baseline)
    CONFIG="configs/experiments/imagenet1k_vit12_baseline_recipe_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_baseline_recipe_30ep"
    ;;
  fullyshared)
    CONFIG="configs/experiments/imagenet1k_vit12_fullyshared_recipe_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_fullyshared_recipe_30ep"
    ;;
  partshr728)
    CONFIG="configs/experiments/imagenet1k_vit12_partshr_orig_r728_recipe_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_partshr_orig_r728_recipe_30ep"
    ;;
  lowrank32)
    CONFIG="configs/experiments/imagenet1k_vit12_lowrank_r32_recipe_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_lowrank_r32_recipe_30ep"
    ;;
  bbt64)
    CONFIG="configs/experiments/imagenet1k_vit12_bbt_recipe_r64_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_bbt_recipe_r64_30ep"
    ;;
  bmbuv32)
    CONFIG="configs/experiments/imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_bmbuv_recipe_r32_s32_30ep"
    ;;
  bmbuv64)
    CONFIG="configs/experiments/imagenet1k_vit12_bmbuv_recipe_r64_s64_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_bmbuv_recipe_r64_s64_30ep"
    ;;
  bmb64)
    CONFIG="configs/experiments/imagenet1k_vit12_bmb_recipe_r64_30ep_v1.json"
    RUN_STEM="imagenet1k_vit12_bmb_recipe_r64_30ep"
    ;;
  *)
    echo "Unsupported METHOD=${METHOD}"
    exit 1
    ;;
esac

RUN_NAME="${RUN_STEM}_gpu${GPU_ID}"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/train.log"

mkdir -p "${OUTPUT_DIR}"

nohup bash -lc "
cd '${PROJECT_ROOT}'
export CUDA_VISIBLE_DEVICES='${GPU_ID}'
export PYTHONPATH='${PROJECT_ROOT}/src':\"\${PYTHONPATH:-}\"
'${PYTHON_BIN}' -m bmb.cli \
  --config '${CONFIG}' \
  --output-dir 'runs/${RUN_NAME}'
" > "${LOG_PATH}" 2>&1 &

echo "gpu_id=${GPU_ID}"
echo "method=${METHOD}"
echo "run_name=${RUN_NAME}"
echo "log_path=${LOG_PATH}"
