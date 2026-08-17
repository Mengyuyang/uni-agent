#!/usr/bin/env bash
# Qwen3.5-9B + Megatron V1 separate-async training on one Ascend A3 node.
# The trainer and rollout pools each use 8 of the node's 16 logical NPUs.
# Rollouts use the canonical Task -> Claude Code -> OpenYuanrong path.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
VERL_ROOT="${VERL_ROOT:-/mnt/share/z00876269/code/verl}"
cd "${REPO_ROOT}"

# Paths from the working A3 Claude Code recipe.
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/Qwen3.5-9B}"
TRAIN_DATA="${TRAIN_DATA:-/mnt/share/z00876269/datasets/uni-agent_old/swe_rebench_filtered_yuanrong.parquet}"
VAL_DATA="${VAL_DATA:-/mnt/share/z00876269/datasets/uni-agent_old/swe_bench_verified_yuanrong.parquet}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/training/task_config_claude_code_openyuanrong.yaml}"

# OpenYuanrong credentials stay in the inherited shell environment. Both the
# current OPENYUANRONG_* names and the legacy AKERNEL_* names are accepted.
OPENYUANRONG_SERVER_ADDRESS="${OPENYUANRONG_SERVER_ADDRESS:-${AKERNEL_SERVER_ADDRESS:-}}"
OPENYUANRONG_TOKEN="${OPENYUANRONG_TOKEN:-${AKERNEL_TOKEN:-}}"
OPENYUANRONG_TUNNEL_SSL_VERIFY="${OPENYUANRONG_TUNNEL_SSL_VERIFY:-${AKERNEL_TUNNEL_SSL_VERIFY:-0}}"
USE_OPENYUANRONG_SDK="${USE_OPENYUANRONG_SDK:-0}"

: "${OPENYUANRONG_SERVER_ADDRESS:?Export OPENYUANRONG_SERVER_ADDRESS or AKERNEL_SERVER_ADDRESS}"
: "${OPENYUANRONG_TOKEN:?Export OPENYUANRONG_TOKEN or AKERNEL_TOKEN}"
[[ -d "${VERL_ROOT}" ]] || { echo "VERL_ROOT does not exist: ${VERL_ROOT}" >&2; exit 1; }
[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${TRAIN_DATA}" ]] || { echo "TRAIN_DATA does not exist: ${TRAIN_DATA}" >&2; exit 1; }
[[ -f "${VAL_DATA}" ]] || { echo "VAL_DATA does not exist: ${VAL_DATA}" >&2; exit 1; }
[[ -f "${TASK_CONFIG}" ]] || { echo "TASK_CONFIG does not exist: ${TASK_CONFIG}" >&2; exit 1; }

export REPO_ROOT VERL_ROOT
export OPENYUANRONG_SERVER_ADDRESS OPENYUANRONG_TOKEN OPENYUANRONG_TUNNEL_SSL_VERIFY
export USE_OPENYUANRONG_SDK
export PYTHONPATH="${VERL_ROOT}:${REPO_ROOT}:${REPO_ROOT}/verl:${PYTHONPATH:-}"

# A3 single-node 8 + 8 layout.
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"
TOTAL_NPUS=$((NNODES * N_GPUS_PER_NODE + ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
if [[ "${NNODES}" != "1" || "${ROLLOUT_NNODES}" != "1" || "${TOTAL_NPUS}" != "16" ]]; then
    echo "This recipe requires one A3 node with trainer=8 and rollout=8 NPUs." >&2
    exit 1
fi

# V1 separate-async pipeline.
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
GEN_TP="${GEN_TP:-4}"
TRAIN_TP="${TRAIN_TP:-4}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-1}"
N="${N:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-16}"

# Rollout and optimization defaults copied from the working recipe.
PROMPT_LENGTH="${PROMPT_LENGTH:-4096}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-75536}"
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"
UPDATE_WEIGHTS_BUCKET_MB="${UPDATE_WEIGHTS_BUCKET_MB:-2048}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
OFFLOAD="${OFFLOAD:-True}"
OPTIMIZER_OFFLOAD_FRACTION="${OFFLOAD_FRACTION:-1.0}"
USE_MBRIDGE="${USE_MBRIDGE:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-False}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-True}"
USE_NAIVE_L2NORM="${USE_NAIVE_L2NORM:-True}"
PAD_BSHD_TO_MINIBATCH_MAX="${PAD_BSHD_TO_MINIBATCH_MAX:-True}"
RECOMPUTE_GRANULARITY="${RECOMPUTE_GRANULARITY:-full}"
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"

# Canonical Task/Agent framework settings.
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
MAX_CONCURRENT_SESSIONS="${MAX_CONCURRENT_SESSIONS:-32}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

# Run metadata and outputs.
PROJECT_NAME="${PROJECT_NAME:-claude_code_task_a3}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-claude_code_task_a3_$(date +%Y%m%d_%H%M)}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
CKPTS_DIR="${CKPTS_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-/mnt/share/z00876269/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${LOG_DIR}/agents}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"
mkdir -p "${CKPTS_DIR}" "${AGENT_LOG_DIR}" "${LOG_DIR}"

ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
INFER_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))

echo "=== Task/Claude Code A3 Training ==="
echo "Repository:  ${REPO_ROOT}"
echo "Model:       ${MODEL_PATH}"
echo "Train data:  ${TRAIN_DATA}"
echo "Val data:    ${VAL_DATA}"
echo "Task config: ${TASK_CONFIG}"
echo "Resources:   trainer=1x${N_GPUS_PER_NODE}, rollout=1x${ROLLOUT_NGPUS_PER_NODE}"
echo "Parallelism: train TP/PP/CP=${TRAIN_TP}/${TRAIN_PP}/${TRAIN_CP}, rollout TP=${GEN_TP}"
echo "Batch:       prompts=${TRAIN_BATCH_SIZE}, n=${N}, mini=${PPO_MINI_BATCH_SIZE}"
echo "Outputs:     ${LOG_DIR}"
echo "Trajectories:${AGENT_LOG_DIR}/step_<step>/session-*"
echo "===================================="

# This full-node training recipe needs a fresh Ray head registered with all 16
# NPU resources. It never registers CUDA/GPU resources.
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export TOTAL_NPUS

python3 - <<'PY_CHECK_NPU'
import os

import torch
import torch_npu  # noqa: F401

expected = int(os.environ["TOTAL_NPUS"])
count = torch.npu.device_count()
print("ASCEND_RT_VISIBLE_DEVICES:", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
print("torch.npu.device_count():", count)
if count < expected:
    raise RuntimeError(f"Current Python sees {count} NPUs; this recipe requires {expected}")
PY_CHECK_NPU

echo "Restarting Ray for exclusive 16-NPU training..."
ray stop --force || true
sleep 2
unset RAY_ADDRESS
ray start --head \
    --resources="{\"NPU\": ${TOTAL_NPUS}}" \
    --disable-usage-stats

python3 - <<'PY_CHECK_RAY'
import os

import ray

expected = int(os.environ["TOTAL_NPUS"])
ray.init(address="auto", ignore_reinit_error=True)
total = ray.cluster_resources()
available = ray.available_resources()
print("Ray total resources:", total)
print("Ray available resources:", available)
if int(total.get("NPU", 0)) != expected:
    raise RuntimeError(f"Ray registered NPU={total.get('NPU', 0)}; expected {expected}")
ray.shutdown()
PY_CHECK_RAY

# Forward only names and inherited values; credentials are not printed or stored
# in the tracked Task YAML.
RUNTIME_ENV_JSON="$(python3 -c '
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
        "TRANSFER_QUEUE_ENABLE": "1",
        "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
        "RAY_DEDUP_LOGS": "0",
        "PYTHONUNBUFFERED": "1",
        "VERL_LOGGING_LEVEL": "INFO",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "SANDBOX_NAME_PREFIX": "claude-code-task-a3-",
    }
)
print(json.dumps({"env_vars": env_vars}))
')"

MAIN_CMD=(
    python3 -m verl.trainer.main_ppo
    --config-name=ppo_megatron_trainer
    +ray_kwargs.ray_init.address=auto
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}"
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}"
    transfer_queue.enable=True
    transfer_queue.metrics.enabled=True
    "data.train_files=['${TRAIN_DATA}']"
    "data.val_files=['${VAL_DATA}']"
    data.prompt_key=prompt
    data.truncation=left
    data.max_prompt_length="${PROMPT_LENGTH}"
    data.max_response_length="${RESPONSE_LENGTH}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.gen_batch_size="${TRAIN_BATCH_SIZE}"
    data.train_max_samples="${TRAIN_MAX_SAMPLES}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.return_raw_chat=True
    data.trust_remote_code=True
    data.dataloader_num_workers=0
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}"
    actor_rollout_ref.hybrid_engine=True
    actor_rollout_ref.nccl_timeout=9600
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}"
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}"
    actor_rollout_ref.rollout.n="${N}"
    actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}"
    actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}"
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${TOP_P}"
    actor_rollout_ref.rollout.top_k="${TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.temperature="${TEMPERATURE}"
    actor_rollout_ref.rollout.val_kwargs.top_p="${TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}"
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.disable_log_stats=False
    +actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}"
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}"
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}"
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}"
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${MAX_CONCURRENT_SESSIONS}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}"
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"'
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    algorithm.rollout_correction.bypass_mode=True
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_dynamic_bsz=True
    +actor_rollout_ref.actor.use_rollout_log_probs=True
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}"
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction="${OPTIMIZER_OFFLOAD_FRACTION}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.param_offload="${OFFLOAD}"
    actor_rollout_ref.actor.megatron.grad_offload="${OFFLOAD}"
    actor_rollout_ref.actor.megatron.optimizer_offload="${OFFLOAD}"
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}"
    actor_rollout_ref.actor.megatron.use_mbridge="${USE_MBRIDGE}"
    actor_rollout_ref.actor.megatron.use_remove_padding="${USE_REMOVE_PADDING}"
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max="${PAD_BSHD_TO_MINIBATCH_MAX}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn="${USE_FLASH_ATTN}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm="${USE_NAIVE_L2NORM}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity="${RECOMPUTE_GRANULARITY}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method="${RECOMPUTE_METHOD}"
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers="${RECOMPUTE_NUM_LAYERS}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}"
    actor_rollout_ref.ref.megatron.param_offload="${OFFLOAD}"
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}"
    actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}"
    actor_rollout_ref.ref.megatron.use_remove_padding="${USE_REMOVE_PADDING}"
    actor_rollout_ref.ref.megatron.pad_bshd_to_minibatch_max="${PAD_BSHD_TO_MINIBATCH_MAX}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn="${USE_FLASH_ATTN}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.use_naive_l2norm="${USE_NAIVE_L2NORM}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.recompute_granularity="${RECOMPUTE_GRANULARITY}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.recompute_method="${RECOMPUTE_METHOD}"
    ++actor_rollout_ref.ref.megatron.override_transformer_config.recompute_num_layers="${RECOMPUTE_NUM_LAYERS}"
    reward.reward_manager.name=dapo
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.val_before_train=False
    trainer.device=npu
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.resume_mode=auto
    trainer.log_val_generations=2
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    MAIN_CMD+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi
MAIN_CMD+=("$@")

ray job submit \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    --working-dir . \
    -- "${MAIN_CMD[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
