"""SWE-bench-like dataset that injects verl-standard reward fields.

Self-contained for the claude-code recipe; mirrors the mini-swe-agent dataset
so claude_code/ does not depend on mini_swe_agent/.
"""

from collections.abc import Mapping

from verl.utils.dataset.rl_dataset import RLHFDataset


def extract_image(env_config: dict) -> str:
    """Extract Docker image from env config, supporting both flat and nested formats.

    Flat:   env_config["image"]
    Nested: env_config["deployment"]["image"]
    """
    image = env_config.get("image")
    if image:
        return image
    deployment = env_config.get("deployment")
    if isinstance(deployment, dict):
        image = deployment.get("image")
        if image:
            return image
    return ""


class SWEBenchDataset(RLHFDataset):
    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        extra_info = row_dict.get("extra_info", {})
        tools_kwargs = extra_info.get("tools_kwargs", {})
        reward_config = tools_kwargs.get("reward", {})
        task_config = tools_kwargs.get("task", {})
        task_config = task_config if isinstance(task_config, Mapping) else {}
        task_metadata = task_config.get("metadata", {}) if isinstance(task_config.get("metadata", {}), Mapping) else {}

        data_source = row_dict.get("data_source") or reward_config.get("name") or task_config.get("name") or "unknown"
        row_dict["data_source"] = data_source
        row_dict.setdefault("reward_model", {"ground_truth": reward_config.get("metadata") or task_metadata})

        return row_dict
