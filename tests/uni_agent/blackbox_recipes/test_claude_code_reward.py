from __future__ import annotations

import sys
import types

import pytest

from examples.blackbox_recipes.claude_code.reward import build_reward_context, evaluate_in_env


@pytest.mark.asyncio
async def test_evaluate_in_env_routes_swe_rebench(monkeypatch):
    calls = []
    fake_reward = types.ModuleType("uni_agent.tasks.swe_rebench.reward")

    async def compute_reward(metadata, env, eval_timeout):
        calls.append((metadata, env, eval_timeout))
        return {"resolved": True, "eval_completed": True}

    fake_reward.compute_reward = compute_reward
    monkeypatch.setitem(sys.modules, "uni_agent.tasks.swe_rebench.reward", fake_reward)

    env = object()
    score, result = await evaluate_in_env(
        env,
        {
            "data_source": "swe_rebench",
            "reward_model": {"ground_truth": {"instance_id": "case-1"}},
        },
        eval_timeout=123,
    )

    assert score == 1.0
    assert result == {"resolved": True, "eval_completed": True}
    assert calls == [({"instance_id": "case-1"}, env, 123)]


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
