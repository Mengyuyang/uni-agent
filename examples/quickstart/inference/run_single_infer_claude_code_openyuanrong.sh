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

# Qwen3.5-35B-A3B on one 16-NPU node: use two TP8 rollout groups. This matches
# the validated verl Ascend topology (16 devices per node, GEN_TP=8).
TP=8
NNODES=1
N_DEVICES_PER_NODE=16

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

if ! ray status >/dev/null 2>&1; then
    echo "No running Ray cluster was found." >&2
    echo "Start the 16-NPU Ray head first, then rerun this script." >&2
    exit 2
fi

python3 - <<'PY'
import importlib

for name in ("ray", "verl", "uni_agent", "vllm", "datasets"):
    module = importlib.import_module(name)
    print(f"dependency ok: {name} -> {getattr(module, '__file__', '<namespace>')}")

from vllm import LLM

print(f"dependency ok: vllm.LLM -> {LLM}")
PY

exec env RAY_ADDRESS="${RAY_ADDRESS:-auto}" python3 examples/inference/parallel_infer_verl.py \
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
