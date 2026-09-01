#!/usr/bin/env bash
set -euo pipefail

# One-sample end-to-end smoke test for:
# Claude Code sidecar -> OpenYuanrong reverse tunnel -> Uni-Agent Gateway -> model.
# Run from the repository root after filling a private runtime-env YAML.

: "${DATA_PATH:?Set DATA_PATH to a provider-agnostic SWE parquet file}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model checkpoint}"
: "${RUNTIME_ENV:?Set RUNTIME_ENV to a private OpenYuanrong runtime-env YAML}"

TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/training/task_config_claude_code_openyuanrong.yaml}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
TP="${TP:-4}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
LIMIT="${LIMIT:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
LOG_DIR="${LOG_DIR:-/tmp/uni_agent_cc_openyuanrong_smoke}"
RESULT_PATH="${RESULT_PATH:-${LOG_DIR}/result.json}"

ray job submit \
    --runtime-env "${RUNTIME_ENV}" \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}" \
    --task-config "${TASK_CONFIG}" \
    --tool-parser "${TOOL_PARSER}" \
    --tensor-parallel-size "${TP}" \
    --nnodes "${NNODES}" \
    --n-gpus-per-node "${N_GPUS_PER_NODE}" \
    --gateway-count "${GATEWAY_COUNT}" \
    --concurrency "${CONCURRENCY}" \
    --limit "${LIMIT}" \
    --log-dir "${LOG_DIR}" \
    --result-path "${RESULT_PATH}"
