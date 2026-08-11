"""Task Config composition shared by standalone and framework-managed execution."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge ``overrides`` onto ``base`` without mutating either mapping.

    Nested dictionaries merge recursively. Lists and scalar values are replaced.
    """
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_task_tools_kwargs(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return task-shaped ``tools_kwargs`` for current or legacy SWE rows.

    Current rows already carry ``tools_kwargs.task``.  The Claude Code recipe's
    older parquet format instead carries ``tools_kwargs.env`` and
    ``tools_kwargs.reward``; convert only that complete legacy shape so the
    framework runner still receives one canonical Task Config.
    """
    extra_info = sample.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise TypeError("sample.extra_info must be a mapping")

    raw_tools_kwargs = extra_info.get("tools_kwargs")
    if not isinstance(raw_tools_kwargs, Mapping):
        raise TypeError("sample.extra_info.tools_kwargs must be a mapping")
    tools_kwargs = dict(raw_tools_kwargs)

    if "task" in tools_kwargs:
        task = tools_kwargs["task"]
        if not isinstance(task, Mapping):
            raise TypeError("sample.extra_info.tools_kwargs.task must be a mapping")
        tools_kwargs["task"] = dict(task)
        return tools_kwargs

    env_config = tools_kwargs.get("env")
    reward_config = tools_kwargs.get("reward")
    if not isinstance(env_config, Mapping) or not isinstance(reward_config, Mapping):
        raise ValueError("sample.extra_info.tools_kwargs requires either task or legacy env and reward mappings")

    image = env_config.get("image")
    if not image:
        deployment = env_config.get("deployment")
        if isinstance(deployment, Mapping):
            image = deployment.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError("legacy sample.extra_info.tools_kwargs.env requires an image")

    task_name = reward_config.get("name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError("legacy sample.extra_info.tools_kwargs.reward requires a name")

    metadata = reward_config.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("legacy sample.extra_info.tools_kwargs.reward.metadata must be a mapping")
    ground_truth = metadata.get("ground_truth")
    if ground_truth is not None:
        if not isinstance(ground_truth, Mapping):
            raise TypeError("legacy sample.extra_info.tools_kwargs.reward.metadata.ground_truth must be a mapping")
        metadata = ground_truth

    prompt = sample.get("prompt")
    if prompt is None:
        raise ValueError("legacy sample requires prompt")

    tools_kwargs["task"] = {
        "name": task_name,
        "sandbox": {"image": image},
        "prompt": prompt,
        "metadata": dict(metadata),
    }
    return tools_kwargs


@functools.lru_cache(maxsize=8)
def _load_task_config_file(path: str) -> dict[str, dict[str, Any]]:
    """Load a Task Config YAML file into a ``{name: config}`` index."""
    import yaml

    raw = yaml.safe_load(Path(path).expanduser().read_text())
    entries = raw if isinstance(raw, list) else [raw]
    index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(f"task_config_path {path!r}: each entry must be a mapping with a 'name' (got {entry!r})")
        name = str(entry["name"])
        if name in index:
            raise ValueError(f"task_config_path {path!r} contains duplicate task name {name!r}")
        index[name] = entry
    return index


@dataclass(frozen=True)
class TaskConfigResolver:
    """Route and compose Task Config defaults, sample values, and runtime bindings."""

    defaults_by_name: Mapping[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> TaskConfigResolver:
        """Build a resolver from a YAML mapping or list keyed by Task ``name``."""
        return cls(defaults_by_name=_load_task_config_file(path))

    def resolve(
        self,
        sample_config: Mapping[str, Any],
        *,
        runtime_model: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve one sample using Task Config → Sample Config → runtime model."""
        task_name = sample_config.get("name")
        if not task_name:
            raise ValueError("sample Task Config requires a 'name'")

        if self.defaults_by_name and task_name not in self.defaults_by_name:
            raise ValueError(
                f"no Task Config for sample task {task_name!r}; available configs: {sorted(self.defaults_by_name)}"
            )

        file_defaults = self.defaults_by_name.get(str(task_name), {})
        resolved = _deep_merge(dict(file_defaults), dict(sample_config))

        model_binding = {key: value for key, value in (runtime_model or {}).items() if value is not None}
        if model_binding:
            resolved = _deep_merge(resolved, {"agent": {"model": model_binding}})
        return resolved
