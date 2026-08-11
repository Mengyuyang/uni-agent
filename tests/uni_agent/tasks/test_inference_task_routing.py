from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from uni_agent.tasks import TaskConfigResolver, get_task, normalize_task_tools_kwargs


def test_resolver_loads_every_named_entry(tmp_path):
    config_path = tmp_path / "tasks.yaml"
    config_path.write_text(
        """
- name: task_a
  sandbox:
    provider: local
  agent:
    name: react
    model:
      temperature: 0.2
- name: task_b
  sandbox:
    provider: modal
  agent:
    name: claude_code
""".strip()
    )
    resolver = TaskConfigResolver.from_file(str(config_path))

    assert set(resolver.defaults_by_name) == {"task_a", "task_b"}
    assert resolver.defaults_by_name["task_a"]["agent"]["model"]["temperature"] == 0.2


def test_resolver_routes_by_name_and_applies_sample_and_runtime_overrides():
    defaults = {
        "task_a": {
            "name": "task_a",
            "sandbox": {"provider": "local"},
            "agent": {
                "name": "react",
                "max_steps": 50,
                "model": {
                    "base_url": "http://model:8000/v1",
                    "model_name": "policy",
                    "api_key": "key",
                    "temperature": 0.8,
                },
            },
        },
        "task_b": {
            "name": "task_b",
            "sandbox": {"provider": "modal"},
            "agent": {
                "name": "react",
                "model": {
                    "base_url": "http://model:8000/v1",
                    "model_name": "policy",
                    "api_key": "key",
                },
            },
        },
    }
    sample = {
        "name": "task_a",
        "sandbox": {"image": "sample-image"},
        "agent": {"max_steps": 200, "model": {"temperature": 0.3}},
    }

    resolver = TaskConfigResolver(defaults)
    resolved = resolver.resolve(
        sample,
        runtime_model={
            "base_url": "http://runtime:8000/v1",
            "model_name": "runtime-policy",
            "api_key": "runtime-key",
        },
    )

    assert resolved["name"] == "task_a"
    assert resolved["sandbox"] == {"provider": "local", "image": "sample-image"}
    assert resolved["agent"]["max_steps"] == 200
    assert resolved["agent"]["model"]["temperature"] == 0.3
    assert resolved["agent"]["model"]["base_url"] == "http://runtime:8000/v1"
    assert resolved["agent"]["model"]["model_name"] == "runtime-policy"
    assert resolved["agent"]["model"]["api_key"] == "runtime-key"


def test_resolver_rejects_missing_route():
    resolver = TaskConfigResolver({"other": {"name": "other"}})
    with pytest.raises(ValueError, match="no Task Config for sample task 'missing'"):
        resolver.resolve({"name": "missing"})


def test_openyuanrong_claude_code_example_is_a_valid_task_config():
    repo_root = Path(__file__).parents[3]
    config_path = repo_root / "examples/quickstart/inference/task_config_claude_code_openyuanrong.yaml"
    resolver = TaskConfigResolver.from_file(str(config_path))
    resolved = resolver.resolve(
        {
            "name": "swe_bench",
            "sandbox": {"image": "swebench/example:latest"},
            "prompt": [{"role": "user", "content": "fix the bug"}],
            "metadata": {"instance_id": "example"},
        },
        runtime_model={
            "base_url": "http://gateway.example:8000/sessions/1/v1",
            "model_name": "policy",
        },
    )

    task = get_task(resolved)

    assert task.config.sandbox.provider == "openyuanrong"
    assert task.config.sandbox.sandbox_kwargs["proxy_port"] == 38197
    assert task.config.sandbox.sandbox_kwargs["mounts"] == [
        {
            "target": "/opt/claude-code",
            "image_url": "7.227.53.47:8091/openyuanrong/claude-code-tool:latest",
        }
    ]
    assert task.config.agent.name == "claude_code"
    assert task.config.agent.executable == "/opt/claude-code/bin/claude"
    assert task.config.agent.workdir == "/testbed"
    assert task.config.agent.auto_install is False
    assert task.config.eval_timeout == 600


def test_openyuanrong_launcher_uses_its_checkout_and_non_destructive_ray_start():
    repo_root = Path(__file__).parents[3]
    launcher = (repo_root / "examples/quickstart/inference/run_infer_claude_code_openyuanrong.sh").read_text()

    assert 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"' in launcher
    assert 'REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"' in launcher
    assert "ensure_ray" in launcher
    assert "ray start --head" in launcher
    assert "ray stop" not in launcher
    assert "rm -rf /tmp/ray" not in launcher


def test_swe_bench_task_forwards_configured_eval_timeout(monkeypatch):
    from uni_agent.tasks.swe_bench.task import SWEBenchTask, SWEBenchTaskConfig

    class FakeSandbox:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

    class FakeAgent:
        async def run(self, *, sandbox, messages):
            return SimpleNamespace(finished=True)

    captured = {}

    async def fake_compute_reward(metadata, sandbox, eval_timeout):
        captured.update(metadata=metadata, sandbox=sandbox, eval_timeout=eval_timeout)
        return {"resolved": False}

    reward_module = ModuleType("uni_agent.tasks.swe_bench.reward")
    reward_module.compute_reward = fake_compute_reward
    monkeypatch.setitem(sys.modules, reward_module.__name__, reward_module)

    metadata = {"instance_id": "example"}
    task = SWEBenchTask(
        SWEBenchTaskConfig(
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


def test_normalize_task_tools_kwargs_preserves_task_shaped_rows():
    sample = {
        "extra_info": {
            "tools_kwargs": {
                "task": {"name": "swe_bench", "sandbox": {"image": "task-image"}},
            }
        }
    }

    assert normalize_task_tools_kwargs(sample) == sample["extra_info"]["tools_kwargs"]


def test_normalize_task_tools_kwargs_converts_legacy_recipe_rows():
    prompt = [{"role": "user", "content": "fix the bug"}]
    sample = {
        "prompt": prompt,
        "extra_info": {
            "tools_kwargs": {
                "env": {"deployment": {"image": "legacy-image"}},
                "reward": {
                    "name": "swe_bench",
                    "metadata": {"ground_truth": {"instance_id": "example"}},
                },
            }
        },
    }

    tools_kwargs = normalize_task_tools_kwargs(sample)

    assert tools_kwargs["task"] == {
        "name": "swe_bench",
        "sandbox": {"image": "legacy-image"},
        "prompt": prompt,
        "metadata": {"instance_id": "example"},
    }


def test_normalize_task_tools_kwargs_rejects_malformed_explicit_task():
    sample = {"extra_info": {"tools_kwargs": {"task": "not-a-mapping", "env": {}, "reward": {}}}}

    with pytest.raises(TypeError, match="tools_kwargs.task must be a mapping"):
        normalize_task_tools_kwargs(sample)
