from __future__ import annotations

import pytest

from examples.blackbox_recipes.claude_code.claude_code_runner import (
    DEFAULT_GATEWAY_PROXY_PORT,
    _create_claude_sandbox,
    read_gateway_proxy_port,
    rewrite_gateway_url,
)


def test_rewrite_gateway_url_uses_configured_proxy_port():
    url = "http://10.0.0.8:40889/sessions/session-1/v1"

    assert rewrite_gateway_url(url, proxy_port=39001, strip_v1=True) == (
        "http://127.0.0.1:39001/sessions/session-1"
    )


def test_read_gateway_proxy_port_defaults_and_validates(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_PROXY_PORT", raising=False)
    assert read_gateway_proxy_port() == DEFAULT_GATEWAY_PROXY_PORT

    monkeypatch.setenv("CLAUDE_CODE_PROXY_PORT", "39001")
    assert read_gateway_proxy_port() == 39001

    monkeypatch.setenv("CLAUDE_CODE_PROXY_PORT", "not-an-int")
    with pytest.raises(ValueError, match="CLAUDE_CODE_PROXY_PORT must be an integer"):
        read_gateway_proxy_port()


@pytest.mark.asyncio
async def test_create_claude_sandbox_passes_matching_proxy_port(monkeypatch):
    captured = {}

    class FakeSandbox:
        async def __aenter__(self, retry=1):
            captured["retry"] = retry
            return self

    def fake_build_sandbox(config):
        captured["config"] = config
        return FakeSandbox()

    monkeypatch.setattr(
        "examples.blackbox_recipes.claude_code.claude_code_runner.build_sandbox",
        fake_build_sandbox,
    )

    await _create_claude_sandbox(
        image="swe-image",
        sidecar_image="claude-tool-image",
        gateway_url="http://10.0.0.8:40889/sessions/session-1/v1",
        proxy_port=39001,
    )

    sandbox_kwargs = captured["config"].sandbox_kwargs
    assert captured["retry"] == 10
    assert sandbox_kwargs["upstream"] == "10.0.0.8:40889"
    assert sandbox_kwargs["proxy_port"] == 39001
    assert sandbox_kwargs["mounts"] == [{"target": "/opt/claude-code", "image_url": "claude-tool-image"}]
