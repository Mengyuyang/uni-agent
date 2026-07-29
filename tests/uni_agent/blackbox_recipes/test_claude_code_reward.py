from __future__ import annotations

import sys
import types

import pytest

from examples.blackbox_recipes.claude_code.reward import build_reward_context, evaluate_in_env


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_source", "module_name"),
    [
        ("swe_bench", "uni_agent.tasks.swe_bench.reward"),
        ("swebench", "uni_agent.tasks.swe_bench.reward"),
        ("princeton-nlp/SWE-bench_Verified", "uni_agent.tasks.swe_bench.reward"),
        ("swe_rebench", "uni_agent.tasks.swe_rebench.reward"),
        ("swerebench", "uni_agent.tasks.swe_rebench.reward"),
        ("nebius/SWE-rebench", "uni_agent.tasks.swe_rebench.reward"),
    ],
)
async def test_evaluate_in_env_routes_supported_data_sources(monkeypatch, data_source, module_name):
    calls = []
    fake_reward = types.ModuleType(module_name)

    async def compute_reward(metadata, env, eval_timeout):
        calls.append((metadata, env, eval_timeout))
        return {"resolved": True, "eval_completed": True}

    fake_reward.compute_reward = compute_reward
    monkeypatch.setitem(sys.modules, module_name, fake_reward)

    env = object()
    score, result = await evaluate_in_env(
        env,
        {
            "data_source": data_source,
            "reward_model": {"ground_truth": {"instance_id": "case-1"}},
        },
        eval_timeout=123,
    )

    assert score == 1.0
    assert result == {"resolved": True, "eval_completed": True}
    assert calls == [({"instance_id": "case-1"}, env, 123)]


@pytest.mark.asyncio
async def test_evaluate_in_env_rejects_unknown_data_source():
    with pytest.raises(ValueError, match="Unsupported reward data source: terminal_bench"):
        await evaluate_in_env(
            object(),
            {
                "data_source": "terminal_bench",
                "reward_model": {"ground_truth": {}},
            },
        )


def test_build_reward_context_accepts_task_config_shape(monkeypatch):
    monkeypatch.setenv("SWE_AGENT_EVAL_TIMEOUT", "321")

    metadata, eval_timeout = build_reward_context(
        {
            "task": {
                "name": "swe_rebench",
                "metadata": {"instance_id": "case-2"},
            }
        }
    )

    assert metadata == {
        "data_source": "swe_rebench",
        "reward_model": {"instance_id": "case-2"},
    }
    assert eval_timeout == 321


def test_build_reward_context_preserves_legacy_reward_precedence():
    metadata, _ = build_reward_context(
        {
            "reward": {
                "name": "swe_bench",
                "metadata": {"instance_id": "legacy-case"},
            },
            "task": {
                "name": "swe_rebench",
                "metadata": {"instance_id": "task-case"},
            },
        }
    )

    assert metadata == {
        "data_source": "swe_bench",
        "reward_model": {"instance_id": "legacy-case"},
    }
