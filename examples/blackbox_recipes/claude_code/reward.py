"""Reward utilities for the blackbox claude-code recipe.

Self-contained; mirrors the mini-swe-agent reward module so claude_code/ does
not depend on mini_swe_agent/.

Contains:
- build_reward_context: extract reward metadata + eval_timeout from tools_kwargs
- compute_score: thin reward function that reads reward_score from extra_info
- evaluate_in_env: run reward evaluation in the sandbox env
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

_SWE_BENCH_ALIASES = {
    "swe_bench",
    "swebench",
    "princeton-nlp/swe-bench_verified",
}
_SWE_REBENCH_ALIASES = {
    "swe_rebench",
    "swerebench",
    "nebius/swe-rebench",
}


def _normalize_data_source(data_source: str) -> str:
    normalized = data_source.strip().lower()
    if normalized in _SWE_BENCH_ALIASES:
        return "swe_bench"
    elif normalized in _SWE_REBENCH_ALIASES:
        return "swe_rebench"
    else:
        return normalized


def build_reward_context(tools_kwargs: dict) -> tuple[dict[str, Any], int]:
    """Extract reward metadata and eval_timeout from per-sample tools_kwargs."""
    reward_config = tools_kwargs.get("reward", {})
    task_config = tools_kwargs.get("task", {})
    task_config = task_config if isinstance(task_config, Mapping) else {}
    task_metadata = task_config.get("metadata", {}) if isinstance(task_config.get("metadata", {}), Mapping) else {}
    metadata = {
        "data_source": reward_config.get("name") or task_config.get("name") or "unknown",
        "reward_model": reward_config.get("metadata") or task_metadata,
    }
    eval_timeout = int(os.environ.get("SWE_AGENT_EVAL_TIMEOUT", "600"))
    return metadata, eval_timeout


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> dict:
    """Read reward_score from extra_info, injected by the agent runner."""
    score = 0.0
    if extra_info and "reward_score" in extra_info:
        score = float(extra_info["reward_score"])
    return {"score": score}


async def evaluate_in_env(
    env,
    metadata: dict[str, Any],
    eval_timeout: int = 600,
) -> tuple[float, dict]:
    """Run reward evaluation in the sandbox env.

    Returns (score, eval_result) where score is 1.0/0.0 and
    eval_result contains details (eval_completed, resolved, etc.).
    """
    data_source = metadata.get("data_source", "unknown")
    reward_model = metadata.get("reward_model", {}) or {}
    if not isinstance(reward_model, Mapping):
        reward_model = {}

    normalized_data_source = _normalize_data_source(str(data_source))
    if normalized_data_source == "swe_bench":
        from uni_agent.tasks.swe_bench.reward import compute_reward
    elif normalized_data_source == "swe_rebench":
        from uni_agent.tasks.swe_rebench.reward import compute_reward
    else:
        raise ValueError(
            f"Unsupported reward data source: {data_source}. Supported: swe_bench, swe_rebench"
        )

    spec_metadata = reward_model.get("ground_truth", reward_model)
    result = await compute_reward(spec_metadata, env, eval_timeout=eval_timeout)
    score = 1.0 if result.get("resolved", False) else 0.0
    return score, result
