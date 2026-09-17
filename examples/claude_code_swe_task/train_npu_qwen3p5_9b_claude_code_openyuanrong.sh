#!/usr/bin/env bash
# Single-node Ascend A3 GSPO recipe for Qwen3.5-9B Dense + Claude Code/OpenYuanrong.
# Target topology: 1 node x 16 NPUs, split into 8 trainer NPUs and 8 rollout NPUs.
# Start the Ray head first, then run this script on that node.
# No-fla_npu compatibility version: do not force AscendC/fla_npu GDN.
# Qwen3.5 Megatron stays in BSHD: remove_padding and dynamic_bsz are disabled.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

RUNTIME_DIR="${RUNTIME_DIR:-/mnt/share/z00876269}"
VERL_ROOT="${VERL_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)/verl}"
MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/Qwen3.5-9B}"
TRAIN_FILE="${TRAIN_FILE:-${RUNTIME_DIR}/datasets/uniagent_0901/swe_rebench_filtered_5k_47_208.parquet}"
TEST_FILE="${TEST_FILE:-${RUNTIME_DIR}/datasets/uniagent_0901/swe_bench_verified_53_47.parquet}"
RUNTIME_ENV="${RUNTIME_ENV:-/mnt/share/z00876269/code/newstruct/runtime_env_openyuanrong.yaml}"
TASK_CONFIG="${TASK_CONFIG:-examples/claude_code_swe_task/task_config_claude_code_openyuanrong.yaml}"
RAY_JOB_ADDRESS="${RAY_JOB_ADDRESS:-http://127.0.0.1:28268}"

PROJECT_NAME="${PROJECT_NAME:-cc-yuanrong-qwen3p5-9b-gspo}"
EXP_NAME="${EXP_NAME:-1node-qwen35-9b-gspo-$(date +%Y%m%d-%H%M)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_DIR}}"
CKPTS_DIR="${CKPTS_DIR:-${OUTPUT_ROOT}/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${OUTPUT_ROOT}/logs/${PROJECT_NAME}/${EXP_NAME}}"
CONSOLE_LOG_DIR="${CONSOLE_LOG_DIR:-${OUTPUT_ROOT}/logs/qwen3p5-9b-uniagent}"
CONSOLE_LOG_FILE="${CONSOLE_LOG_FILE:-${CONSOLE_LOG_DIR}/${EXP_NAME}.log}"

# Keep the full Ray console in one append-only file while still showing it in
# the launching terminal.
mkdir -p "${CONSOLE_LOG_DIR}"
exec > >(tee -a "${CONSOLE_LOG_FILE}") 2>&1
echo "Streaming Ray console logs to ${CONSOLE_LOG_FILE}"

export PYTHONPATH="/mnt/share/t00986241/recipe/Megatron-Bridge/src:${VERL_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# -------- Ascend runtime / vLLM optimizations --------
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE="${VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE:-1}"

# -------- Qwen3.5 GDN backend --------
# No fla_npu in this environment: do not force use_ascend_gdn/use_triton_gdn.
# Let the same default GDN path used by the working recipe take effect.

# Follow the mini-swe separate-async layout on one A3 node. Actor/ref use one
# 8-NPU pool and vLLM uses a disjoint 8-NPU pool.
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
ROLLOUT_NNODES="${ROLLOUT_NNODES:-1}"
ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"
PHYSICAL_NPUS="${PHYSICAL_NPUS:-16}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-2}"

# The physical 16-NPU node is split into one 8-NPU trainer pool and one 8-NPU
# TP8 rollout engine. Sixteen prompts keep the PPO mini-batch at the proven
# eight-prompt size while producing 128 trajectories in two 64-session waves.
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-16}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

# Retain the tested 136K context envelope. The dedicated TP8 rollout pool uses
# the same per-engine token budget as the mini-swe recipe.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128000}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_MODEL_LEN}}"

# Dense trainer: TP(2) x PP(1) x CP(4) consumes the 8-NPU trainer pool.
# Rollout follows mini-swe and spans the complete, separate 8-NPU pool.
ROLLOUT_TP="${ROLLOUT_TP:-${ROLLOUT_NGPUS_PER_NODE}}"
TRAIN_TP="${TRAIN_TP:-2}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-4}"

GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
CONCURRENCY="${CONCURRENCY:-64}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-32}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-longest}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"

# -------- GSPO algorithm settings --------
# Match the mini-swe-agent recipe: GRPO advantage estimation with the GSPO
# sequence-level policy loss and its narrow clipping range.
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
USE_KL_IN_REWARD="${USE_KL_IN_REWARD:-False}"
KL_COEF="${KL_COEF:-0.0}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-4e-4}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"
CLIP_RATIO_C="${CLIP_RATIO_C:-10.0}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
BYPASS_MODE="${BYPASS_MODE:-True}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
LOSS_MODE="${LOSS_MODE:-gspo}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:-10}"

ACTOR_PPO_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
LOG_PROB_MAX_TOKEN_LEN=$((MAX_MODEL_LEN / TRAIN_CP))
TRAIN_NPUS=$((NNODES * NGPUS_PER_NODE))
ROLLOUT_NPUS=$((ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
TOTAL_NPUS=$((TRAIN_NPUS + ROLLOUT_NPUS))
TRAIN_MODEL_PARALLEL_SIZE=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))

if (( TOTAL_NPUS != PHYSICAL_NPUS )); then
    echo "Trainer + rollout request ${TOTAL_NPUS} NPUs, but PHYSICAL_NPUS=${PHYSICAL_NPUS}" >&2
    exit 1
fi
if (( TRAIN_NPUS % TRAIN_MODEL_PARALLEL_SIZE != 0 )); then
    echo "Trainer NPUs (${TRAIN_NPUS}) must be divisible by TP*PP*CP (${TRAIN_MODEL_PARALLEL_SIZE})" >&2
    exit 1
fi
if (( ROLLOUT_NPUS % ROLLOUT_TP != 0 )); then
    echo "Rollout NPUs (${ROLLOUT_NPUS}) must be divisible by ROLLOUT_TP (${ROLLOUT_TP})" >&2
    exit 1
fi
if (( TRAIN_PROMPT_BSZ != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_PROMPT_BSZ must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE for separate_async" >&2
    exit 1
fi
if [[ ! -r "${TASK_CONFIG}" ]]; then
    echo "Task config does not exist or is not readable: ${TASK_CONFIG}" >&2
    exit 1
fi

echo "===== Single-node Qwen3.5-9B Dense training topology ====="
echo "MODEL_PATH=${MODEL_PATH}"
echo "Resources: trainer=${NNODES}x${NGPUS_PER_NODE}, rollout=${ROLLOUT_NNODES}x${ROLLOUT_NGPUS_PER_NODE}, total_npu=${TOTAL_NPUS}"
echo "TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}, N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT}, trajectories_per_step=$((TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT))"
echo "CONCURRENCY=${CONCURRENCY}, NUM_AGENT_WORKERS=${NUM_AGENT_WORKERS}, GATEWAY_COUNT=${GATEWAY_COUNT}"
echo "TRAIN TP/PP/CP=${TRAIN_TP}/${TRAIN_PP}/${TRAIN_CP}, DP=$((TRAIN_NPUS / TRAIN_MODEL_PARALLEL_SIZE)), ROLLOUT_TP=${ROLLOUT_TP}"
echo "Trainer mode=separate_async, parameter_sync_step=${PARAMETER_SYNC_STEP}"
echo "Algorithm: adv_estimator=${ADV_ESTIMATOR}, loss_mode=${LOSS_MODE}, clip=${CLIP_RATIO_LOW}/${CLIP_RATIO_HIGH}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "Ray job address=${RAY_JOB_ADDRESS}"
echo "GDN backend: old/default path (no fla_npu forced)"

ray job submit --address="${RAY_JOB_ADDRESS}" --runtime-env "${RUNTIME_ENV}" --working-dir "${REPO_ROOT}" \
    -- env PYTHONPATH="${PYTHONPATH}" PYTHONUNBUFFERED=1 RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
    python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    transfer_queue.enable=True \
    transfer_queue.metrics.enabled=True \
    trainer.device=npu \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${MAX_MODEL_LEN}" \
    "data.train_files=['${TRAIN_FILE}']" \
    "data.val_files=['${TEST_FILE}']" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.dataloader_num_workers=0 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    data.val_batch_size=1 \
    data.gen_batch_size="${TRAIN_PROMPT_BSZ}" \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
    actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.enable_sleep_mode=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
    actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection="${TRAJECTORY_SELECTION}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode="${MASK_UNFINISHED_EPISODE}" \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_cpu_binding=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=true \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    algorithm.rollout_correction.bypass_mode="${BYPASS_MODE}" \
    actor_rollout_ref.actor.checkpoint.strict=False \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c="${CLIP_RATIO_C}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
    '+actor_rollout_ref.actor.megatron.override_ddp_config={grad_reduce_in_fp32: false, overlap_grad_reduce: true, bucket_size: 50000000}' \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.use_remove_padding=False \
    actor_rollout_ref.actor.megatron.pad_bshd_to_minibatch_max=True \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=auto \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_naive_l2norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.ref.megatron.use_remove_padding=False \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    'trainer.logger=["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
