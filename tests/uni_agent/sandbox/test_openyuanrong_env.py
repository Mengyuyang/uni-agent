from __future__ import annotations

import os
import sys
import types

from uni_agent.sandbox import openyuanrong


def test_load_sandbox_module_accepts_akernel_env(monkeypatch):
    fake_sdk = types.ModuleType("openyuanrong_sandbox_sdk")
    monkeypatch.setitem(sys.modules, "openyuanrong_sandbox_sdk", fake_sdk)
    monkeypatch.setattr(openyuanrong, "_sdk_initialized", False)
    monkeypatch.setenv("USE_OPENYUANRONG_SDK", "1")
    monkeypatch.delenv("OPENYUANRONG_SERVER_ADDRESS", raising=False)
    monkeypatch.delenv("OPENYUANRONG_TOKEN", raising=False)
    monkeypatch.delenv("OPENYUANRONG_TUNNEL_SSL_VERIFY", raising=False)
    monkeypatch.setenv("AKERNEL_SERVER_ADDRESS", "akernel.example:8888")
    monkeypatch.setenv("AKERNEL_TOKEN", "token")
    monkeypatch.setenv("AKERNEL_TUNNEL_SSL_VERIFY", "1")

    assert openyuanrong._load_sandbox_module() is fake_sdk
    assert os.environ["OPENYUANRONG_SERVER_ADDRESS"] == "akernel.example:8888"
    assert os.environ["OPENYUANRONG_TOKEN"] == "token"
    assert os.environ["OPENYUANRONG_TUNNEL_SSL_VERIFY"] == "1"
    assert os.environ["AKERNEL_TUNNEL_SSL_VERIFY"] == "1"
    assert os.environ["TUNNEL_SSL_VERIFY"] == "1"
