#!/usr/bin/env bash
set -euo pipefail

# One-sample Claude Code inference for the preinstalled newstruct environment.
# YuanRong credentials are expected to already be exported by the login shell
# (for example from ~/.bashrc). Python/Ray/verl/vLLM dependencies are expected
# to already be installed in the active environment.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATA_PATH="/mnt/share/z00876269/datasets/uniagent_0901/swe_bench_verified_53_47.parquet"
MODEL_PATH="/mnt/share/z00876269/models/Qwen3.5-35B-A3B"
TASK_CONFIG="${REPO_ROOT}/examples/quickstart/training/task_config_claude_code_openyuanrong.yaml"
LOG_DIR="/mnt/share/z00876269/logs/cc-yuanrong-single"
RESULT_PATH="${LOG_DIR}/result.json"

# The defaults below assume one node with eight accelerator resources. Change
# these three values in this file if the Ray cluster topology is different.
TP=8
NNODES=1
N_DEVICES_PER_NODE=8

for name in OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN; do
    if [[ -z "${!name:-}" ]]; then
        echo "${name} is not visible to this shell; run 'source ~/.bashrc' first" >&2
        exit 2
    fi
done

for path in "${DATA_PATH}" "${MODEL_PATH}/config.json" "${TASK_CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

python3 - <<'PY'
import importlib

for name in ("ray", "verl", "uni_agent", "vllm", "datasets"):
    module = importlib.import_module(name)
    print(f"dependency ok: {name} -> {getattr(module, '__file__', '<namespace>')}")
PY

exec python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "Qwen3.5-35B-A3B" \
    --task-config "${TASK_CONFIG}" \
    --tool-parser qwen3_coder \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_DEVICES_PER_NODE}" \
    --gateway-count 1 \
    --concurrency 1 \
    --limit 1 \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"
