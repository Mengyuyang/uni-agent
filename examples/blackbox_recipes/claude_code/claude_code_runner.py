"""Claude Code runner for the blackbox SWE-agent recipe.

Claude Code runs inside a remote sandbox via a sidecar tool image mounted at
``/opt/claude-code``. The runner creates the sandbox, invokes the ``claude``
binary pointed at the gateway, evaluates the reward in the same sandbox, and
posts ``reward_info`` to the per-session endpoint (same contract as the
mini-swe-agent runner).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
import time
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import httpx

from examples.blackbox_recipes.claude_code.dataset import extract_image
from examples.blackbox_recipes.claude_code.reward import build_reward_context, evaluate_in_env
from uni_agent.gateway.session import SessionHandle
from uni_agent.sandbox import Sandbox, SandboxConfig, build_sandbox

logger = logging.getLogger(__name__)

DEFAULT_TOOL_IMAGE = "swr.cn-east-3.myhuaweicloud.com/openyuanrong/claude-code-tool:latest"
TOOL_TARGET = "/opt/claude-code"
DEFAULT_GATEWAY_PROXY_PORT = 38197
_DIAGNOSTIC_LOG_LIMIT = 4000
_SWE_REBENCH_GIT_CLEAN_HISTORY = " && ".join(
    [
        "git tag -d $(git tag -l) || true",
        "git reflog expire --expire=now --all || true",
        "git gc --prune=now || true",
    ]
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _diagnostic_tail(value: str | None) -> str:
    return (value or "")[-_DIAGNOSTIC_LOG_LIMIT:]


def _create_dump_dir(session_id: str, sample_index: int) -> Path | None:
    dump_root = os.environ.get("CLAUDE_CODE_DUMP_DIR", "").strip()
    if not dump_root:
        return None

    safe_session_id = "".join(char if char.isalnum() or char in "._-" else "_" for char in session_id)[:80]
    session_hash = sha256(session_id.encode()).hexdigest()[:10]
    run_dir = Path(dump_root).expanduser() / f"sample-{sample_index}-{safe_session_id or 'session'}-{session_hash}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("failed to create Claude Code dump directory %s", run_dir, exc_info=True)
        return None
    logger.info("[sample %d] dumping Claude Code artifacts to %s", sample_index, run_dir)
    return run_dir


def _dump_text(run_dir: Path | None, name: str, content: str | None) -> None:
    if run_dir is None:
        return
    try:
        (run_dir / name).write_text(content or "", encoding="utf-8")
    except OSError:
        logger.warning("failed to write Claude Code dump artifact %s", run_dir / name, exc_info=True)


def _dump_json(run_dir: Path | None, name: str, payload: Mapping) -> None:
    _dump_text(run_dir, name, json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def extract_upstream(gateway_url: str) -> str:
    """Extract host:port from a gateway URL for upstream tunnel config."""
    parsed = urlparse(gateway_url)
    return f"{parsed.hostname}:{parsed.port}"


def rewrite_gateway_url(
    gateway_url: str,
    proxy_port: int = DEFAULT_GATEWAY_PROXY_PORT,
    *,
    strip_v1: bool = False,
) -> str:
    """Rewrite gateway URL to the sandbox-internal tunnel (127.0.0.1:<proxy_port>)."""
    parsed = urlparse(gateway_url)
    path = parsed.path.removesuffix("/v1") if strip_v1 else parsed.path
    return f"http://127.0.0.1:{proxy_port}{path}"


class SandboxEnvForReward:
    """Adapts :class:`Sandbox` to the async env interface used by reward
    evaluation (``communicate``, ``write_file``, ``read_file``, ``exec_shell``).
    """

    def __init__(self, sandbox):
        self._sandbox = sandbox

    async def communicate(self, input: str, timeout=600, check="ignore", error_msg="Command failed") -> str:
        result = await self._sandbox.exec_shell(input, timeout=int(timeout))
        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(
                f"{error_msg} (exit_code={result.exit_code}) stdout={result.stdout[:200]} stderr={result.stderr[:200]}"
            )
        return result.stdout

    async def write_file(self, path: str | Path, content: str) -> None:
        encoded = base64.b64encode(content.encode()).decode()
        await self.communicate(f"echo {encoded} | base64 -d > {path}", check="raise", error_msg=f"write {path}")

    async def read_file(self, path: str | Path, **_) -> str:
        return await self.communicate(f"cat {path}")

    async def exec_shell(self, command: str, *, workdir=None, timeout=600):
        return await self._sandbox.exec_shell(command, workdir=workdir, timeout=int(timeout))


def extract_task(raw_prompt) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt
    return next(
        (m["content"] for m in raw_prompt if isinstance(m, dict) and m.get("role") == "user"),
        str(raw_prompt),
    )


def _extract_issue_text(task: str) -> str:
    start = task.find("<issue_description>")
    end = task.find("</issue_description>")
    if start >= 0 and end > start:
        return task[start + len("<issue_description>") : end].strip()
    marker = "\nFollow these steps to resolve the issue:"
    if marker in task:
        return task.split(marker, 1)[0].strip()
    return task.strip()


def build_claude_task(raw_prompt, tools_kwargs: dict | None = None) -> str:
    if tools_kwargs is None:
        tools_kwargs = {}
    elif not isinstance(tools_kwargs, Mapping):
        raise TypeError(f"tools_kwargs must be a mapping, got {type(tools_kwargs).__name__}")

    if "task" in tools_kwargs:
        task_config = tools_kwargs["task"]
        if not isinstance(task_config, Mapping):
            raise TypeError(f"tools_kwargs.task must be a mapping, got {type(task_config).__name__}")
        metadata = task_config.get("metadata", {})
        metadata_path = "tools_kwargs.task.metadata"
    else:
        reward_config = tools_kwargs.get("reward", {})
        if not isinstance(reward_config, Mapping):
            raise TypeError(f"tools_kwargs.reward must be a mapping, got {type(reward_config).__name__}")
        metadata = reward_config.get("metadata", {})
        metadata_path = "tools_kwargs.reward.metadata"

    if not isinstance(metadata, Mapping):
        raise TypeError(f"{metadata_path} must be a mapping, got {type(metadata).__name__}")

    task = extract_task(raw_prompt)
    issue = metadata.get("problem_statement") or _extract_issue_text(task)

    return (
        "You are fixing a SWE-bench task in /testbed.\n\n"
        "Issue:\n"
        f"{issue}\n\n"
        "Rules:\n"
        "- Inspect the repository and locate the implementation relevant to the issue.\n"
        "- Make the minimal required changes to non-test files. Do not modify tests.\n"
        "- The development environment is already installed; do not install packages unless a test command proves it "
        "is necessary.\n"
        "- There is no submit tool in this environment. Do not try to submit.\n"
        "- Do not commit.\n"
        "- Run focused tests that are available in the checkout and/or directly reproduce the behavior described by "
        "the issue.\n"
        "- Do not treat one passing pre-existing test as sufficient proof that the requested behavior is fixed.\n"
        "- Once the implementation and focused validation are complete, print a short summary and exit.\n"
    )


def build_claude_command(
    *,
    task: str,
    base_url: str,
    max_turns: int,
    model: str = "default",
    permission_mode: str = "bypassPermissions",
    conda_env: str | None = "testbed",
    disable_web_tools: bool = True,
    disable_slash_commands: bool = True,
) -> str:
    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "not-needed",
        "ANTHROPIC_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_FORK_SUBAGENT": "0",
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "DISABLE_AUTOUPDATER": "1",
        "IS_SANDBOX": "1",
    }
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in env.items()]
    if conda_env:
        conda_prefix = f"/opt/miniconda3/envs/{conda_env}"
        env_assignments.extend(
            [
                f"CONDA_DEFAULT_ENV={shlex.quote(conda_env)}",
                f"CONDA_PREFIX={shlex.quote(conda_prefix)}",
                f"PATH={shlex.quote(conda_prefix + '/bin')}:/opt/miniconda3/bin:$PATH",
            ]
        )
    env_prefix = " ".join(env_assignments)
    argv = [
        "/opt/claude-code/bin/claude",
        "-p",
        task,
        "--model",
        model,
        "--max-turns",
        str(max_turns),
        "--permission-mode",
        permission_mode,
    ]
    if disable_slash_commands:
        argv.append("--disable-slash-commands")
    if disable_web_tools:
        argv.extend(["--disallowedTools", "Agent", "Task", "WebFetch", "WebSearch"])
    return (
        "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy; "
        "cd /testbed; "
        f"{env_prefix} " + shlex.join(argv)
    )


async def _create_claude_sandbox(
    *,
    image: str,
    sidecar_image: str,
    gateway_url: str,
    proxy_port: int,
) -> Sandbox:
    upstream = extract_upstream(gateway_url) if gateway_url else None
    config = SandboxConfig(
        provider=os.getenv("SANDBOX_PROVIDER", "openyuanrong"),
        image=image,
        sandbox_kwargs={
            "mounts": [{"target": TOOL_TARGET, "image_url": sidecar_image}],
            "upstream": upstream,
            "proxy_port": proxy_port,
        },
    )
    sandbox = build_sandbox(config)
    await sandbox.__aenter__(retry=10)
    return sandbox


async def claude_code_runner(
    *,
    raw_prompt,
    session: SessionHandle,
    sample_index: int,
    tools_kwargs: dict | None = None,
    tool_image: str = DEFAULT_TOOL_IMAGE,
    run_timeout: int = 7200,
    conda_env: str = "testbed",
    proxy_port: int = DEFAULT_GATEWAY_PROXY_PORT,
    **kwargs,
) -> None:
    """Run Claude Code inside a sandbox with sidecar tool mount.

    Flow:
        1. Create remote sandbox with the claude-code sidecar
        2. Run the claude binary against the gateway tunnel
        3. Evaluate reward in the same sandbox
        4. Post reward_info for the framework reward path
    """
    if tools_kwargs is None:
        tools_kwargs = {}
    elif not isinstance(tools_kwargs, Mapping):
        raise TypeError(f"tools_kwargs must be a mapping, got {type(tools_kwargs).__name__}")
    logger.info("claude_code_runner called, sample_index=%d", sample_index)

    task = build_claude_task(raw_prompt, tools_kwargs)
    metadata, eval_timeout = build_reward_context(tools_kwargs)
    evaluator = metadata["evaluator"]
    dump_dir = _create_dump_dir(session.session_id, sample_index)
    _dump_text(dump_dir, "prompt.txt", task)

    if "task" in tools_kwargs:
        task_config = tools_kwargs["task"]
        if not isinstance(task_config, Mapping):
            raise TypeError(f"tools_kwargs.task must be a mapping, got {type(task_config).__name__}")
        env_config = task_config.get("sandbox", {})
        env_path = "tools_kwargs.task.sandbox"
    else:
        env_config = tools_kwargs.get("env", {})
        env_path = "tools_kwargs.env"

    if not isinstance(env_config, Mapping):
        raise TypeError(f"{env_path} must be a mapping, got {type(env_config).__name__}")

    image = extract_image(env_config)
    if not image:
        raise ValueError(f"No Docker image found in {env_path} for sample {sample_index}")

    gateway_url = session.base_url
    if not gateway_url:
        raise ValueError(f"gateway_url is empty for sample {sample_index}")

    sandbox = await _create_claude_sandbox(
        image=image,
        sidecar_image=tool_image,
        gateway_url=gateway_url,
        proxy_port=proxy_port,
    )

    try:
        if evaluator == "swe_rebench":
            # Match SWEREBenchTask.run(): remove future history before the agent reads the repo.
            await sandbox.exec_shell(_SWE_REBENCH_GIT_CLEAN_HISTORY, workdir="/testbed")

        claude_base_url = rewrite_gateway_url(gateway_url, proxy_port=proxy_port, strip_v1=True)
        max_turns = int(os.environ.get("AGENT_MAX_TURNS", "100"))
        agent_cmd = build_claude_command(
            task=task,
            base_url=claude_base_url,
            max_turns=max_turns,
            conda_env=conda_env,
        )

        started_at = time.perf_counter()
        result = await sandbox.exec_shell(agent_cmd, timeout=int(run_timeout))
        elapsed = time.perf_counter() - started_at
        logger.info("[sample %d] claude-code finished rc=%s elapsed=%.1fs", sample_index, result.exit_code, elapsed)
        _dump_text(dump_dir, "response.txt", result.stdout)
        _dump_text(dump_dir, "stderr.txt", result.stderr)
        _dump_json(
            dump_dir,
            "agent.json",
            {"exit_code": result.exit_code, "elapsed_seconds": elapsed},
        )
        if result.exit_code != 0:
            logger.warning(
                "[sample %d] claude-code failed stdout_tail=%r stderr_tail=%r",
                sample_index,
                _diagnostic_tail(result.stdout),
                _diagnostic_tail(result.stderr),
            )
        elif _env_flag("CLAUDE_CODE_DIAGNOSTICS"):
            logger.info(
                "[sample %d] claude-code succeeded stdout_tail=%r stderr_tail=%r",
                sample_index,
                _diagnostic_tail(result.stdout),
                _diagnostic_tail(result.stderr),
            )

        if _env_flag("CLAUDE_CODE_DIAGNOSTICS"):
            try:
                repo_state = await sandbox.exec_shell(
                    "printf '%s\\n' '--- git status --short ---'; "
                    "git status --short; "
                    "printf '%s\\n' '--- git diff --stat ---'; "
                    "git diff --stat",
                    workdir="/testbed",
                    timeout=60,
                )
                logger.info(
                    "[sample %d] repository state before reward rc=%s stdout=%r stderr=%r",
                    sample_index,
                    repo_state.exit_code,
                    _diagnostic_tail(repo_state.stdout),
                    _diagnostic_tail(repo_state.stderr),
                )
            except Exception:
                logger.warning("[sample %d] failed to collect repository diagnostics", sample_index, exc_info=True)

        score, eval_result = await evaluate_in_env(SandboxEnvForReward(sandbox), metadata, eval_timeout)
        logger.info("[sample %d] reward done score=%s resolved=%s", sample_index, score, eval_result.get("resolved"))

        reward_info = {
            "reward_score": score,
            "claude_code_exit_code": result.exit_code,
            **eval_result,
        }
        _dump_json(dump_dir, "reward.json", reward_info)
        if not session.reward_info_url:
            raise ValueError(f"reward_info_url is empty for session {session.session_id}")
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(session.reward_info_url, json={"reward_info": reward_info})
            response.raise_for_status()
    finally:
        await sandbox.stop()
