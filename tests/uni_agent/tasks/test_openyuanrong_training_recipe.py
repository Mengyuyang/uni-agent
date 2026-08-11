import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from uni_agent.tasks import TaskConfigResolver, get_task

REPO_ROOT = Path(__file__).parents[3]
TASK_CONFIG = REPO_ROOT / "examples/quickstart/training/task_config_claude_code_openyuanrong.yaml"
LAUNCHER = REPO_ROOT / "examples/quickstart/training/train_npu_qwen3p5_claude_code_openyuanrong.sh"


@pytest.mark.parametrize("task_name", ["swe_bench", "swe_rebench"])
def test_openyuanrong_training_task_config_routes_both_swe_tasks(task_name):
    resolver = TaskConfigResolver.from_file(str(TASK_CONFIG))
    resolved = resolver.resolve(
        {
            "name": task_name,
            "sandbox": {"image": "swebench/example:latest"},
            "prompt": [{"role": "user", "content": "fix it"}],
            "metadata": {"instance_id": "example"},
        },
        runtime_model={
            "base_url": "http://gateway.example/sessions/1/v1",
            "model_name": "Qwen3.5-9B",
        },
    )

    task = get_task(resolved)

    assert task.config.eval_timeout == 600
    assert task.config.sandbox.provider == "openyuanrong"
    assert task.config.sandbox.image == "swebench/example:latest"
    assert task.config.sandbox.sandbox_kwargs["proxy_port"] == 38197
    assert task.config.sandbox.sandbox_kwargs["mounts"] == [
        {
            "target": "/opt/claude-code",
            "image_url": "7.227.53.47:8091/openyuanrong/claude-code-tool:latest",
        }
    ]
    assert task.config.agent.name == "claude_code"
    assert task.config.agent.max_turns == 40
    assert task.config.agent.run_timeout == 7200


def test_a3_training_launcher_uses_only_the_canonical_task_runner():
    launcher = LAUNCHER.read_text()

    assert "trainer.v1.trainer_mode=separate_async" in launcher
    assert "trainer.device=npu" in launcher
    assert 'N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"' in launcher
    assert 'ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE:-8}"' in launcher
    assert '--resources="{\\"NPU\\": ${TOTAL_NPUS}}"' in launcher
    assert "--num-gpus" not in launcher
    assert "uni_agent.framework.entry.AgentFrameworkRolloutAdapter" in launcher
    assert "agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task" in launcher
    assert "agent_runners.claude_code" not in launcher
    assert "examples.blackbox_recipes.claude_code" not in launcher
    assert "use_reward_loop_worker=False" in launcher
    assert "+actor_rollout_ref.rollout.enable_sleep_mode=True" in launcher
    assert "+actor_rollout_ref.actor.use_rollout_log_probs=True" in launcher
    assert "NCCL_P2P_DISABLE" not in launcher
    assert "NCCL_SHM_DISABLE" not in launcher


def test_swe_rebench_task_forwards_training_eval_timeout(monkeypatch):
    from uni_agent.tasks.swe_rebench.task import SWEREBenchTask, SWEREBenchTaskConfig

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def exec_shell(self, command, *, workdir=None):
            return SimpleNamespace(exit_code=0)

    class FakeAgent:
        async def run(self, *, sandbox, messages):
            return SimpleNamespace(finished=True)

    captured = {}

    async def fake_compute_reward(metadata, sandbox, eval_timeout):
        captured.update(metadata=metadata, sandbox=sandbox, eval_timeout=eval_timeout)
        return {"resolved": False}

    reward_module = ModuleType("uni_agent.tasks.swe_rebench.reward")
    reward_module.compute_reward = fake_compute_reward
    monkeypatch.setitem(sys.modules, reward_module.__name__, reward_module)

    metadata = {"instance_id": "example"}
    task = SWEREBenchTask(
        SWEREBenchTaskConfig(
            sandbox={"provider": "local"},
            agent={"name": "claude_code"},
            prompt=[{"role": "user", "content": "fix it"}],
            metadata=metadata,
            eval_timeout=600,
        )
    )
    sandbox = FakeSandbox()
    monkeypatch.setattr(task, "build_sandbox", lambda: sandbox)
    monkeypatch.setattr(task, "build_agent", FakeAgent)

    result = asyncio.run(task.run())

    assert captured == {"metadata": metadata, "sandbox": sandbox, "eval_timeout": 600}
    assert result.reward == 0.0
    assert result.finished is True
