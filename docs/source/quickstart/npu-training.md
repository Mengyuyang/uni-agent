# Run Training on NPU

Uni-Agent runs the same Agent workflow on Ascend NPUs. The distributed Qwen3-Coder
recipe uses VeOmni, while the single-node dense Qwen3.5 recipe below uses Megatron V1
with separate trainer and rollout pools.

The distributed recipe below trains `Qwen3-Coder-30B-A3B-Instruct` with the
white-box `ReAct Agent`.

## Prerequisites

- An Ascend NPU environment with `torch_npu` and `vllm_ascend`. See the [verl Ascend Tutorial](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/README.md) for image builds and installation.
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni), which provides the training engine this recipe runs on:

    ```bash
    pip3 install --no-deps "git+https://github.com/ByteDance-Seed/VeOmni.git@main"
    ```

- Datasets, Task Config, and Ray Runtime Environment prepared as described in [Run Agent RL Training](rl-training.md). 

## Launch Training

Set the shared data and runtime root, then launch from the repository root:

```bash
RAY_DATA_HOME=/path/to/data \
NNODES=16 \
NGPUS_PER_NODE=16 \
USP_SIZE=16 \
EXPERT_SIZE=8 \
GEN_TP=4 \
CONCURRENCY=1024 \
TRAIN_PROMPT_BSZ=64 \
N_RESP_PER_PROMPT=16 \
PPO_MINI_BATCH_SIZE=16 \
TASK_CONFIG=examples/quickstart/training/task_config_react.yaml \
EXP_NAME=react_qwen3_coder_30b_npu_gspo \
ADV_ESTIMATOR=grpo \
LOSS_MODE=gspo \
ROLLOUT_IS=token \
TEST_FREQ=-1 \
bash examples/quickstart/training/train_npu_qwen3_moe.sh
```

The default layout is:

```text
<RAY_DATA_HOME>/
├── models/Qwen3-Coder-30B-A3B-Instruct/
├── data/uni_agent/
│   ├── swe_rebench_filtered_1150.parquet
│   ├── swe_bench_verified.parquet
│   └── runtime_env.yaml
├── ckpts/
└── logs/
```

Override `MODEL_PATH`, `TRAIN_FILE`, `TEST_FILE`, `RUNTIME_ENV`, or `TASK_CONFIG` when your layout differs.

The recipe sets `trainer.device=npu`, runs the actor with `model_engine=veomni` (full-shard FSDP with `USP_SIZE` Ulysses and `EXPERT_SIZE` expert parallelism), and selects the NPU kernel backends `moe_implementation=fused_npu`, `rms_norm_implementation=npu`, and `rotary_pos_emb_implementation=npu`. 

## Monitor the Run

Checkpoints and per-session Agent logs are written under:

```text
<RAY_DATA_HOME>/ckpts/Uni-Agent-Qwen3-Coder-30B-veomni-npu-colocate/<EXP_NAME>/
<RAY_DATA_HOME>/logs/Uni-Agent-Qwen3-Coder-30B-veomni-npu-colocate/<EXP_NAME>/
```

## Results

_To be added._

## Claude Code + OpenYuanrong on One A3 Node

This recipe is the training counterpart of the OpenYuanrong Claude Code inference
smoke test. It uses the canonical Task path end to end:

```text
verl.trainer.main_ppo
  -> V1 separate_async (trainer 8 NPUs + rollout 8 NPUs)
  -> AgentFrameworkRolloutAdapter
  -> task_runner.run_task
  -> SWE-reBench/SWE-Bench Task
  -> Claude Code in OpenYuanrong
  -> Task reward -> TransferQueue -> PPO update
```

The defaults match the A3 environment used by the existing Claude Code recipe:

- model: `/mnt/share/weights/Qwen3.5-9B`
- training data: `/mnt/share/z00876269/datasets/uni-agent_old/swe_rebench_filtered_yuanrong.parquet`
- validation data: `/mnt/share/z00876269/datasets/uni-agent_old/swe_bench_verified_yuanrong.parquet`
- external verl source: `/mnt/share/z00876269/code/verl`

Legacy `env` / `reward` rows are normalized to Task Configs inside `run_task`, so
these parquet files do not need to be regenerated. The launcher accepts exported
`OPENYUANRONG_*` or legacy `AKERNEL_*` credentials from the current shell and does
not store them in the Task YAML.

Run from the repository checkout:

```bash
bash examples/quickstart/training/train_npu_qwen3p5_claude_code_openyuanrong.sh
```

The script restarts the local Ray cluster and registers exactly 16 `NPU` resources,
so launch it only when the A3 node is available exclusively for this training job.
It streams the Ray job until completion. Checkpoints are written under
`checkpoints/claude_code_task_a3/`; the console and per-session Agent logs are
written under `/mnt/share/z00876269/logs/claude_code_task_a3/`.
