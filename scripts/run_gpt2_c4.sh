#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
METHOD="${METHOD:-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/data/cache/c4-realnewslike-gpt2-1024}"
MAX_STEPS="${MAX_STEPS:-50000}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"

case "${METHOD}" in
  baseline)
    VARIANT="baseline"
    RUN_STEM="gpt2_c4_baseline_50k"
    EXTRA_ARGS=""
    ;;
  fullyshared)
    VARIANT="fullyshared"
    RUN_STEM="gpt2_c4_fullshr_50k"
    EXTRA_ARGS=""
    ;;
  partshr728)
    VARIANT="partialshared"
    RUN_STEM="gpt2_c4_partshr728_50k"
    EXTRA_ARGS="--shared-dim 728"
    ;;
  lowrank32)
    VARIANT="lowrank"
    RUN_STEM="gpt2_c4_lowrank_r32_50k"
    EXTRA_ARGS="--rank 32"
    ;;
  bbt64)
    VARIANT="bbt"
    RUN_STEM="gpt2_c4_bbt_r64_50k"
    EXTRA_ARGS="--rank 64"
    ;;
  bmbuv32)
    VARIANT="bmbuv"
    RUN_STEM="gpt2_c4_bmbuv_r32s32_50k"
    EXTRA_ARGS="--rank 32 --factor-rank 32"
    ;;
  bmbuv64)
    VARIANT="bmbuv"
    RUN_STEM="gpt2_c4_bmbuv_r64s64_50k"
    EXTRA_ARGS="--rank 64 --factor-rank 64"
    ;;
  bmb64)
    VARIANT="bmb"
    RUN_STEM="gpt2_c4_bmb_r64_50k"
    EXTRA_ARGS="--rank 64"
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
'${PYTHON_BIN}' -m bmb.train_gpt2_c4 \
  --variant '${VARIANT}' \
  --dataset-path '${DATASET_PATH}' \
  --output-dir '${OUTPUT_DIR}' \
  --max-steps ${MAX_STEPS} \
  --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
  --gradient-accumulation-steps ${GRAD_ACCUM} \
  --learning-rate ${LEARNING_RATE} \
  --warmup-steps ${WARMUP_STEPS} \
  --eval-steps ${EVAL_STEPS} \
  --save-steps ${SAVE_STEPS} \
  --logging-steps ${LOGGING_STEPS} \
  --bf16 \
  --gradient-checkpointing \
  ${EXTRA_ARGS}
" > "${LOG_PATH}" 2>&1 &

echo "gpu_id=${GPU_ID}"
echo "method=${METHOD}"
echo "run_name=${RUN_NAME}"
echo "dataset_path=${DATASET_PATH}"
echo "log_path=${LOG_PATH}"
