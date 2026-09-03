#!/usr/bin/env bash
# One GRPO training-step smoke test for Claude Code on OpenYuanrong.
#
# This is deliberately a *training* command, not the standalone inference
# runner: it collects two Claude Code rollouts for one SWE-Rebench prompt,
# runs the task verifier, reports the resulting rewards to verl, and performs
# one policy update.  Increase the environment-variable overrides below only
# after this completes successfully.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/Qwen3.5-4B}"
TRAIN_FILE="${TRAIN_FILE:-/mnt/share/z00876269/datasets/uniagent_0901/swe_rebench_filtered_5k_47_208.parquet}"
TEST_FILE="${TEST_FILE:-/mnt/share/z00876269/datasets/uniagent_0901/swe_bench_verified_53_47.parquet}"
TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/examples/quickstart/training/task_config_claude_code_openyuanrong.yaml}"

RUN_ROOT="${RUN_ROOT:-/mnt/share/z00876269/outputs/cc-yuanrong-qwen3p5-4b}"
EXP_NAME="${EXP_NAME:-reward-smoke-$(date +%Y%m%d-%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-cc-yuanrong-qwen3p5-4b}"
CKPTS_DIR="${CKPTS_DIR:-${RUN_ROOT}/checkpoints/${EXP_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${RUN_ROOT}/logs/${EXP_NAME}}"

# Leave empty to use the already installed cluster environment.  If your Ray
# cluster needs a job runtime-env YAML, set RUNTIME_ENV to that file before
# launching this script.
RUNTIME_ENV="${RUNTIME_ENV:-}"

for name in OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN; do
    if [[ -z "${!name:-}" ]]; then
        echo "${name} is not visible; run 'source ~/.bashrc' in this shell first." >&2
        exit 2
    fi
done

for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 2
    fi
done

if ! ray status >/dev/null 2>&1; then
    echo "No running Ray cluster was found. Start the Ray head before launching training." >&2
    exit 2
fi

mkdir -p "${CKPTS_DIR}" "${AGENT_LOG_DIR}"
cd "${REPO_ROOT}"

exec env \
    MODEL_PATH="${MODEL_PATH}" \
    TRAIN_FILE="${TRAIN_FILE}" \
    TEST_FILE="${TEST_FILE}" \
    TASK_CONFIG="${TASK_CONFIG}" \
    RUNTIME_ENV="${RUNTIME_ENV}" \
    PROJECT_NAME="${PROJECT_NAME}" \
    EXP_NAME="${EXP_NAME}" \
    CKPTS_DIR="${CKPTS_DIR}" \
    AGENT_LOG_DIR="${AGENT_LOG_DIR}" \
    NNODES="${NNODES:-1}" \
    NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}" \
    GEN_TP="${GEN_TP:-2}" \
    TP="${TP:-4}" \
    PP="${PP:-1}" \
    CP="${CP:-2}" \
    TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-1}" \
    N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-2}" \
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}" \
    GATEWAY_COUNT="${GATEWAY_COUNT:-1}" \
    CONCURRENCY="${CONCURRENCY:-2}" \
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}" \
    SAVE_FREQ="${SAVE_FREQ:-1}" \
    TEST_FREQ="${TEST_FREQ:--1}" \
    MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-True}" \
    bash examples/quickstart/training/train_qwen3p5_dense.sh
