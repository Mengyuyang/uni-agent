#!/usr/bin/env bash
set -euo pipefail

# Run this script from a Linux Ray head node. It keeps OpenYuanrong credentials
# out of the tracked Runtime Environment YAML and forwards them to Ray workers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${REPO_ROOT}"

: "${OPENYUANRONG_SERVER_ADDRESS:?Set OPENYUANRONG_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Set OPENYUANRONG_TOKEN}"
: "${MODEL_PATH:?Set MODEL_PATH to a local path or Hugging Face model id}"

export OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN

DATA_PATH="${DATA_PATH:-${HOME}/data/swe_agent/swe_bench_verified.parquet}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/inference/task_config_claude_code_openyuanrong.yaml}"
ENGINE="${ENGINE:-vllm}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TP="${TP:-4}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
N="${N:-1}"
LIMIT="${LIMIT:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEM_UTIL:-0.7}}"
LOG_DIR="${LOG_DIR:-/mnt/shared/uni_agent_logs/openyuanrong-claude-code-smoke}"
RESULT_PATH="${RESULT_PATH:-/mnt/shared/results/openyuanrong-claude-code-smoke.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-0}"
export USE_OPENYUANRONG_SDK="${USE_OPENYUANRONG_SDK:-0}"

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
env_vars.update(
    {
        "PYTHONPATH": "verl",
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
