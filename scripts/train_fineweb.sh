#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALE="${SCALE:-small}"
if [[ -z "${NPROC_PER_NODE+x}" ]]; then
  if [[ "${SCALE}" == "xl" || "${SCALE}" == "gqa_xl" ]]; then
    NPROC_PER_NODE=24
  else
    NPROC_PER_NODE=4
  fi
fi
CONFIG="${CONFIG:-${ROOT}/configs/fineweb_${SCALE}.json}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
NNODES="${NNODES:-1}"
if [[ "${NNODES}" == "1" ]]; then
  TORCHRUN_TOPOLOGY=(--standalone)
else
  : "${NODE_RANK:?Set NODE_RANK for multi-node training}"
  : "${MASTER_ADDR:?Set MASTER_ADDR for multi-node training}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  TORCHRUN_TOPOLOGY=(
    --nnodes="${NNODES}"
    --node_rank="${NODE_RANK}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
  )
fi
exec torchrun "${TORCHRUN_TOPOLOGY[@]}" --nproc_per_node="${NPROC_PER_NODE}" \
  -m bmb.train_fineweb --config "${CONFIG}" "$@"
