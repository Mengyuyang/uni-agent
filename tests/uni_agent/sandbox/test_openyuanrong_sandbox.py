from __future__ import annotations

import pytest

from uni_agent.sandbox import SandboxConfig, bind_gateway_endpoint
from uni_agent.sandbox.openyuanrong import DEFAULT_GATEWAY_PROXY_PORT, OpenyuanrongSandbox


def test_gateway_binding_adds_dynamic_upstream_and_default_proxy_port():
    config = SandboxConfig(
        provider="openyuanrong",
        image="swebench/example:latest",
        sandbox_kwargs={"mounts": [{"target": "/opt/tool", "image_url": "registry/tool:v1"}]},
    )

    bound, sandbox_url = bind_gateway_endpoint(
        config,
        "http://10.0.0.8:42317/sessions/session-1/v1",
    )

    assert config.sandbox_kwargs == {
        "mounts": [{"target": "/opt/tool", "image_url": "registry/tool:v1"}],
    }
    assert bound is not config
    assert bound.sandbox_kwargs == {
        "mounts": [{"target": "/opt/tool", "image_url": "registry/tool:v1"}],
        "upstream": "10.0.0.8:42317",
        "proxy_port": DEFAULT_GATEWAY_PROXY_PORT,
    }
    assert sandbox_url == "http://127.0.0.1:38197/sessions/session-1/v1"

    sandbox = OpenyuanrongSandbox.from_config(bound)
    assert sandbox.upstream == "10.0.0.8:42317"
    assert sandbox.proxy_port == DEFAULT_GATEWAY_PROXY_PORT
    assert sandbox.mounts == [{"target": "/opt/tool", "image_url": "registry/tool:v1"}]


def test_gateway_binding_runtime_endpoint_overrides_stale_upstream():
    config = SandboxConfig(
        provider="openyuanrong",
        sandbox_kwargs={
            "upstream": "stale.example:1111",
            "proxy_port": "39123",
        },
    )

    bound, sandbox_url = OpenyuanrongSandbox.bind_gateway_endpoint(
        config,
        "https://[2001:db8::10]:4443/sessions/session-2/v1",
    )

    assert bound.sandbox_kwargs["upstream"] == "[2001:db8::10]:4443"
    assert bound.sandbox_kwargs["proxy_port"] == 39123
    assert sandbox_url == "http://127.0.0.1:39123/sessions/session-2/v1"


@pytest.mark.parametrize(
    "base_url, message",
    [
        ("gateway.example:8000/sessions/1/v1", "invalid Gateway base_url"),
        ("http://gateway.example/sessions/1/v1", "explicit port"),
        ("http://gateway.example:0/sessions/1/v1", "Gateway port must be"),
        ("http://gateway.example:not-a-port/sessions/1/v1", "invalid Gateway port"),
        ("http://user:pass@gateway.example:8000/sessions/1/v1", "must not contain user info"),
    ],
)
def test_gateway_binding_rejects_invalid_gateway_urls(base_url, message):
    config = SandboxConfig(provider="openyuanrong")

    with pytest.raises(ValueError, match=message):
        OpenyuanrongSandbox.bind_gateway_endpoint(config, base_url)


@pytest.mark.parametrize("proxy_port", [True, 38197.5, 0, 65536, "not-a-port"])
def test_gateway_binding_rejects_invalid_proxy_ports(proxy_port):
    config = SandboxConfig(
        provider="openyuanrong",
        sandbox_kwargs={"proxy_port": proxy_port},
    )

    with pytest.raises(ValueError, match="proxy_port"):
        OpenyuanrongSandbox.bind_gateway_endpoint(
            config,
            "http://gateway.example:8000/sessions/1/v1",
        )


def test_other_sandbox_providers_keep_gateway_url_unchanged():
    config = SandboxConfig(provider="local")
    base_url = "http://gateway.example:8000/sessions/1/v1"

    bound, sandbox_url = bind_gateway_endpoint(config, base_url)

    assert bound is config
    assert sandbox_url == base_url
