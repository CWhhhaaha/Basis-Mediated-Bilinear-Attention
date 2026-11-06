#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_FULLSHR="${GPU_FULLSHR:-1}"
GPU_PARTSHR728="${GPU_PARTSHR728:-2}"
GPU_BMB64="${GPU_BMB64:-3}"
GPU_BMBUV64="${GPU_BMBUV64:-4}"
GPU_BBT64="${GPU_BBT64:-5}"
GPU_LOWRANK32="${GPU_LOWRANK32:-6}"
GPU_BMBUV32="${GPU_BMBUV32:-7}"

launch_one() {
  local gpu_id="$1"
  local method="$2"
  GPU_ID="${gpu_id}" METHOD="${method}" \
    bash "${PROJECT_ROOT}/scripts/run_imagenet.sh"
}

launch_one "${GPU_BASELINE}" baseline
launch_one "${GPU_FULLSHR}" fullyshared
launch_one "${GPU_PARTSHR728}" partshr728
launch_one "${GPU_BMB64}" bmb64
launch_one "${GPU_BMBUV64}" bmbuv64
launch_one "${GPU_BBT64}" bbt64
launch_one "${GPU_LOWRANK32}" lowrank32
launch_one "${GPU_BMBUV32}" bmbuv32

echo ""
echo "All ImageNet-1K runs launched."
