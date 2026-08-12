from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from examples.blackbox_recipes.claude_code import claude_code_runner as runner_mod
from uni_agent.gateway.session import SessionHandle


@pytest.mark.parametrize("config_key", ["reward", "task"])
def test_build_claude_task_does_not_expose_evaluator_test_oracles(config_key):
    metadata = {
        "problem_statement": "Implement the requested behavior.",
        "FAIL_TO_PASS": ["hidden_fail_to_pass_sentinel"],
        "PASS_TO_PASS": json.dumps(["hidden_pass_to_pass_sentinel"]),
    }
    tools_kwargs = {config_key: {"name": "swe_rebench", "metadata": metadata}}

    task = runner_mod.build_claude_task("fallback issue", tools_kwargs)

    assert "Implement the requested behavior." in task
    assert "hidden_fail_to_pass_sentinel" not in task
    assert "hidden_pass_to_pass_sentinel" not in task
    assert "Relevant tests to run" not in task
    assert "listed relevant pytest" not in task
    assert "exit immediately" not in task
    assert "one passing pre-existing test" in task


def test_build_claude_task_falls_back_to_issue_description():
    raw_prompt = (
        "<issue_description>Fix the parser fallback.</issue_description>\nFollow these steps to resolve the issue:"
    )

    task = runner_mod.build_claude_task(raw_prompt, {"reward": {"name": "swe_bench", "metadata": {}}})

    assert "Fix the parser fallback." in task
    assert "<issue_description>" not in task


@pytest.mark.parametrize(
    ("tools_kwargs", "expected_path"),
    [
        ({"reward": [], "env": {"image": "image"}}, "tools_kwargs.reward"),
        ({"task": [], "env": {"image": "image"}}, "tools_kwargs.task"),
        ({"reward": {"metadata": []}, "env": {"image": "image"}}, "tools_kwargs.reward.metadata"),
        ({"task": {"metadata": []}, "env": {"image": "image"}}, "tools_kwargs.task.metadata"),
    ],
)
def test_build_claude_task_rejects_malformed_contracts(tools_kwargs, expected_path):
    with pytest.raises(TypeError, match=expected_path):
        runner_mod.build_claude_task("issue", tools_kwargs)


class _FakeSandbox:
    def __init__(self):
        self.calls: list[dict] = []
        self.stopped = False

    async def exec_shell(self, command, *, workdir=None, timeout=None):
        self.calls.append({"command": command, "workdir": workdir, "timeout": timeout})
        if command == "agent-command":
            return SimpleNamespace(exit_code=0, stdout="agent response", stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    async def stop(self):
        self.stopped = True


class _FakeResponse:
    def raise_for_status(self):
        return None


class _FakeAsyncClient:
    posts: list[tuple[str, dict]] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def post(self, url, json):
        self.posts.append((url, json))
        return _FakeResponse()


@pytest.mark.parametrize("contract", ["legacy", "task"])
def test_runner_uses_canonical_evaluator_for_rebench_cleanup(monkeypatch, contract):
    sandbox = _FakeSandbox()
    seen = {}

    async def fake_create_sandbox(**kwargs):
        return sandbox

    async def fake_evaluate(env, metadata, eval_timeout):
        seen["metadata"] = metadata
        seen["eval_timeout"] = eval_timeout
        return 1.0, {"resolved": True}

    if contract == "legacy":
        tools_kwargs = {
            "reward": {"name": "swe_rebench", "metadata": {"problem_statement": "fix it"}},
            "env": {"image": "sandbox-image", "post_setup_cmd": "must-not-run"},
        }
    else:
        tools_kwargs = {
            "task": {
                "name": "swe_rebench",
                "metadata": {"problem_statement": "fix it"},
                "sandbox": {"image": "sandbox-image", "post_setup_cmd": "must-not-run"},
            }
        }

    monkeypatch.delenv("CLAUDE_CODE_DIAGNOSTICS", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_DUMP_DIR", raising=False)
    monkeypatch.setenv("SWE_AGENT_EVAL_TIMEOUT", "321")
    monkeypatch.setattr(runner_mod, "_create_claude_sandbox", fake_create_sandbox)
    monkeypatch.setattr(runner_mod, "build_claude_command", lambda **kwargs: "agent-command")
    monkeypatch.setattr(runner_mod, "evaluate_in_env", fake_evaluate)
    monkeypatch.setattr(runner_mod.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.posts.clear()

    asyncio.run(
        runner_mod.claude_code_runner(
            raw_prompt="fix it",
            session=SessionHandle(
                session_id=f"session-{contract}",
                base_url="http://gateway:8000/v1",
                reward_info_url="http://gateway:8000/reward_info",
            ),
            sample_index=3,
            tools_kwargs=tools_kwargs,
        )
    )

    assert sandbox.calls[0] == {
        "command": runner_mod._SWE_REBENCH_GIT_CLEAN_HISTORY,
        "workdir": "/testbed",
        "timeout": None,
    }
    assert sandbox.calls[1]["command"] == "agent-command"
    assert all(call["command"] != "must-not-run" for call in sandbox.calls)
    assert seen["metadata"]["evaluator"] == "swe_rebench"
    assert seen["eval_timeout"] == 321
    assert sandbox.stopped is True
    assert _FakeAsyncClient.posts[0][1]["reward_info"]["reward_score"] == 1.0


def test_runner_dumps_prompt_response_and_reward(monkeypatch, tmp_path):
    sandbox = _FakeSandbox()

    async def fake_create_sandbox(**kwargs):
        return sandbox

    async def fake_evaluate(env, metadata, eval_timeout):
        return 0.0, {"resolved": False, "report": "failed hidden test"}

    monkeypatch.setenv("CLAUDE_CODE_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("CLAUDE_CODE_DIAGNOSTICS", raising=False)
    monkeypatch.setattr(runner_mod, "_create_claude_sandbox", fake_create_sandbox)
    monkeypatch.setattr(runner_mod, "build_claude_command", lambda **kwargs: "agent-command")
    monkeypatch.setattr(runner_mod, "evaluate_in_env", fake_evaluate)
    monkeypatch.setattr(runner_mod.httpx, "AsyncClient", _FakeAsyncClient)

    asyncio.run(
        runner_mod.claude_code_runner(
            raw_prompt="dump this issue",
            session=SessionHandle(
                session_id="session/unsafe",
                base_url="http://gateway:8000/v1",
                reward_info_url="http://gateway:8000/reward_info",
            ),
            sample_index=7,
            tools_kwargs={
                "reward": {"name": "swe_bench", "metadata": {"problem_statement": "dump this issue"}},
                "env": {"image": "sandbox-image"},
            },
        )
    )

    session_id = "session/unsafe"
    run_dir = tmp_path / f"sample-7-session_unsafe-{sha256(session_id.encode()).hexdigest()[:10]}"
    assert sandbox.calls[0]["command"] == "agent-command"
    assert all(call["command"] != runner_mod._SWE_REBENCH_GIT_CLEAN_HISTORY for call in sandbox.calls)
    assert "dump this issue" in (run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert (run_dir / "response.txt").read_text(encoding="utf-8") == "agent response"
    assert json.loads((run_dir / "agent.json").read_text(encoding="utf-8"))["exit_code"] == 0
    assert json.loads((run_dir / "reward.json").read_text(encoding="utf-8"))["resolved"] is False
