import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import uni_agent.framework.task_runner as task_runner
from uni_agent.framework.task_runner import _reward_info_from_result
from uni_agent.tasks import TaskResult


def test_task_result_positional_field_order():
    result = TaskResult(0.5, 1.0, False, {"reason": "limit"})

    assert result.reward == 0.5
    assert result.accuracy == 1.0
    assert result.finished is False
    assert result.extra_info == {"reason": "limit"}


def test_reward_info_omits_unknown_agent_completion():
    result = TaskResult(reward=0.5, accuracy=1.0)

    assert _reward_info_from_result(result) == {
        "reward": 0.5,
        "acc": 1.0,
    }


@pytest.mark.parametrize("finished", [True, False])
def test_reward_info_forwards_agent_completion(finished):
    result = TaskResult(reward=0.0, finished=finished)

    assert _reward_info_from_result(result) == {
        "reward": 0.0,
        "finished": finished,
    }


def test_reward_info_rejects_non_boolean_agent_completion():
    result = TaskResult(reward=0.0, finished=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finished must be a bool or None"):
        _reward_info_from_result(result)


def test_run_task_binds_openyuanrong_gateway_before_task_start(monkeypatch):
    sample_task = {
        "name": "swe_bench",
        "sandbox": {
            "provider": "openyuanrong",
            "image": "swebench/example:latest",
            "sandbox_kwargs": {
                "proxy_port": 39001,
                "upstream": "stale.example:1234",
            },
        },
        "metadata": {"instance_id": "example"},
    }
    original_task = deepcopy(sample_task)
    captured: dict = {}

    class _FakeTask:
        async def run(self):
            return TaskResult(reward=1.0, finished=True)

    def _capture_task(config):
        captured.update(config)
        return _FakeTask()

    monkeypatch.setattr(task_runner, "get_task", _capture_task)
    session = SimpleNamespace(
        base_url="http://10.0.0.8:42317/sessions/session-1/v1",
        reward_info_url=None,
    )

    result = asyncio.run(
        task_runner.run_task(
            session=session,
            tools_kwargs={"task": sample_task},
            model_name="policy",
        )
    )

    assert result.reward == 1.0
    assert sample_task == original_task
    assert captured["sandbox"]["sandbox_kwargs"] == {
        "proxy_port": 39001,
        "upstream": "10.0.0.8:42317",
    }
    assert captured["agent"]["model"]["base_url"] == ("http://127.0.0.1:39001/sessions/session-1/v1")
    assert captured["agent"]["model"]["model_name"] == "policy"


def test_run_task_normalizes_legacy_claude_code_recipe_rows(monkeypatch):
    raw_prompt = [{"role": "user", "content": "fix the bug"}]
    tools_kwargs = {
        "env": {"deployment": {"image": "swebench/example:latest"}},
        "reward": {
            "name": "swe_bench",
            "metadata": {
                "ground_truth": {
                    "instance_id": "example",
                    "repo": "example/repo",
                }
            },
        },
    }
    original_tools_kwargs = deepcopy(tools_kwargs)
    captured: dict = {}

    class _FakeTask:
        async def run(self):
            return TaskResult(reward=0.0, finished=True)

    def _capture_task(config):
        captured.update(config)
        return _FakeTask()

    monkeypatch.setattr(task_runner, "get_task", _capture_task)
    session = SimpleNamespace(
        base_url="http://10.0.0.8:42317/sessions/session-1/v1",
        reward_info_url=None,
    )
    repo_root = Path(__file__).parents[3]
    task_config = repo_root / "examples/quickstart/training/task_config_claude_code_openyuanrong.yaml"

    result = asyncio.run(
        task_runner.run_task(
            session=session,
            raw_prompt=raw_prompt,
            tools_kwargs=tools_kwargs,
            task_config_path=str(task_config),
            model_name="policy",
        )
    )

    assert result.finished is True
    assert tools_kwargs == original_tools_kwargs
    assert captured["name"] == "swe_bench"
    assert captured["prompt"] == raw_prompt
    assert captured["metadata"] == {
        "instance_id": "example",
        "repo": "example/repo",
    }
    assert captured["sandbox"]["image"] == "swebench/example:latest"
    assert captured["sandbox"]["provider"] == "openyuanrong"
    assert captured["agent"]["name"] == "claude_code"
