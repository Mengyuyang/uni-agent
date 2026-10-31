# Claude Code SWE Training Recipe

This example trains Qwen3.5-9B with GRPO while Claude Code solves SWE-bench
and SWE-reBench tasks inside OpenYuanrong sandboxes. The Claude Code executable
is supplied by a sidecar image and exposed as `/usr/bin/claude` through the
provider-independent `sandbox.executable_paths` setting.

## Files

- `Dockerfile.claude-code-tool`: builds the Claude Code sidecar image.
- `build_tool.sh`: builds and optionally pushes the sidecar image.
- `task_config_claude_code_openyuanrong.yaml`: configures the agent and the
  OpenYuanrong sandboxes.
- `run_train.sh`: launches the two-node Ascend Qwen3.5-9B GRPO recipe.

## Build the tool image

```bash
bash examples/claude_code_swe_task/build_tool.sh \
    --registry swr.cn-east-3.myhuaweicloud.com/openyuanrong
```

Keep the pushed image tag synchronized with
`sandbox.image_mounts[].image` in the task configuration.

## Launch training

Start the two-node Ray cluster first. The default recipe expects 16 NPUs per
node. Then run the launcher from the repository root:

```bash
DATA_DIR=/path/to/shared/data \
RUNTIME_DIR=/path/to/shared/runtime \
bash examples/claude_code_swe_task/run_train.sh
```

`DATA_DIR` supplies the default model and dataset paths. `RUNTIME_DIR` supplies
the Ray runtime environment, checkpoint directory, and agent log directory.
All paths can also be overridden directly with `MODEL_PATH`, `TRAIN_FILE`,
`TEST_FILE`, `RUNTIME_ENV`, `CKPTS_DIR`, and `AGENT_LOG_DIR`.

The Ray runtime environment must provide the OpenYuanrong client dependencies
and credentials required by the sandbox provider. See
`task_config_claude_code_openyuanrong.yaml` for the sandbox images, mounted tool
image, startup command, and agent timeouts.

The launcher defaults to the validated two-node configuration: 64 prompts per
step, 8 responses per prompt, TP2/PP2/CP8 for training, and TP2 rollout replicas.
These defaults remain overridable through the environment variables defined at
the top of `run_train.sh`.

## Claude CLI model name and stream timeout

Set `cli_model_name` in the task's `agent` configuration when Claude Code needs
a model name or alias different from the policy model name:

```yaml
agent:
  name: claude_code
  model:
    model_name: Qwen3.5-9B
  cli_model_name: sonnet
  run_timeout: 4800
```

This example assumes the configured gateway accepts the `sonnet` alias and
routes it to the intended policy. The override sets the CLI's `--model`,
`ANTHROPIC_MODEL`, and, when subagents are enabled, `CLAUDE_CODE_SUBAGENT_MODEL`.
It does not change `agent.model.model_name` or load a different policy model.
If the override is omitted or empty, all three use `model.model_name` as before.

The agent also sets `CLAUDE_STREAM_IDLE_TIMEOUT_MS` to `run_timeout * 1000`
(integer milliseconds), aligning the CLI stream watchdog with the outer
execution budget. For the example above, the value is `4800000`.
To use a different CLI idle timeout, add this under `agent`:

```yaml
extra_env:
  CLAUDE_STREAM_IDLE_TIMEOUT_MS: "1200000"
```

This override leaves the outer sandbox execution timeout at `run_timeout`.
The setting is passed to the Claude Code process; whether it is honored depends
on the installed CLI version. It does not configure the OpenYuanrong tunnel
client/server timeout (`YR_TUNNEL_HTTP_TIMEOUT`) or the task evaluation timeout.
