#!/usr/bin/env bash
set -euo pipefail

# Run this script from a Linux A3 node. It keeps OpenYuanrong credentials out of
# the tracked Runtime Environment YAML and forwards them to Ray workers.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-/mnt/share/z00876269/code/verl}"
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/Qwen3.5-9B}"
cd "${REPO_ROOT}"

OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-${AKERNEL_SERVER_ADDRESS:-}}"
OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-${AKERNEL_TOKEN:-}}"
OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-${AKERNEL_TUNNEL_SSL_VERIFY:-0}}"

: "${OPENYUANRONG_SERVER_ADDRESS:?Export OPENYUANRONG_SERVER_ADDRESS or AKERNEL_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Export OPENYUANRONG_TOKEN or AKERNEL_TOKEN}"
: "${MODEL_PATH:?Set MODEL_PATH to a local path or Hugging Face model id}"

export REPO_ROOT VERL_ROOT
export OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN OPENYUANRONG_TUNNEL_SSL_VERIFY

DATA_PATH="${DATA_PATH:-${VAL_DATA:-/mnt/share/z00876269/datasets/uni-agent_old/swe_bench_verified_yuanrong.parquet}}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/inference/task_config_claude_code_openyuanrong.yaml}"
ENGINE="${ENGINE:-vllm}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TP="${TP:-${GEN_TP:-4}}"
NNODES="${NNODES:-${ROLLOUT_NNODES:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${ROLLOUT_NGPUS_PER_NODE:-8}}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-${MAX_CONCURRENT_SESSIONS:-1}}"
N="${N:-1}"
LIMIT="${LIMIT:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEM_UTIL:-0.7}}"
LOG_DIR="${LOG_DIR:-/mnt/share/z00876269/logs/openyuanrong-claude-code-smoke}"
RESULT_PATH="${RESULT_PATH:-/mnt/share/z00876269/logs/openyuanrong-claude-code-smoke.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAY_AUTO_START="${RAY_AUTO_START:-1}"
RAY_NPU_RESOURCES="${RAY_NPU_RESOURCES:-${N_GPUS_PER_NODE}}"

export USE_OPENYUANRONG_SDK="${USE_OPENYUANRONG_SDK:-0}"

ensure_ray() {
    # Prefer an explicitly configured, healthy cluster. If RAY_ADDRESS is stale,
    # retry local discovery without it before starting a single-node A3 cluster.
    if ray status >/dev/null 2>&1; then
        return
    fi
    if (unset RAY_ADDRESS; ray status >/dev/null 2>&1); then
        unset RAY_ADDRESS
        return
    fi
    if [[ "${RAY_AUTO_START}" != "1" ]]; then
        echo "No reachable Ray cluster. Start Ray or set RAY_AUTO_START=1." >&2
        exit 1
    fi
    if [[ "${NNODES}" != "1" ]]; then
        echo "Ray auto-start only supports NNODES=1; start the multi-node cluster first." >&2
        exit 1
    fi

    echo "No reachable Ray cluster; starting a local head with NPU=${RAY_NPU_RESOURCES}."
    unset RAY_ADDRESS
    ray start --head \
        --resources="{\"NPU\": ${RAY_NPU_RESOURCES}}" \
        --disable-usage-stats
}

ensure_ray

RUNTIME_ENV_JSON="$("${PYTHON_BIN}" -c '
import json
import os

keys = (
    "OPENYUANRONG_SERVER_ADDRESS",
    "OPENYUANRONG_TOKEN",
    "OPENYUANRONG_TUNNEL_SSL_VERIFY",
    "USE_OPENYUANRONG_SDK",
)
env_vars = {key: os.environ[key] for key in keys}
python_paths = [
    os.environ["VERL_ROOT"],
    os.environ["REPO_ROOT"],
    os.path.join(os.environ["REPO_ROOT"], "verl"),
]
python_paths.extend(filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)))
env_vars.update(
    {
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_paths)),
        "VLLM_DISABLE_COMPILE_CACHE": "1",
    }
)
print(json.dumps({"env_vars": env_vars}))
')"

ray job submit \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --task-config "${TASK_CONFIG}" \
    --engine "${ENGINE}" \
    --tool-parser "${TOOL_PARSER}" \
    --tensor-parallel-size "${TP}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --n "${N}" \
    --limit "${LIMIT}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"
