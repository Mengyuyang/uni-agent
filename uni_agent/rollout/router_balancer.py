"""Expose uni-agent decode selectors through verl's router protocol."""

from __future__ import annotations

import importlib
from typing import Any

from uni_agent.rollout.decode_selectors import DecodeCandidate


def _load_class(fqn: str) -> type:
    module_name, _, class_name = fqn.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"selector_class must be a fully-qualified class name, got {fqn!r}")
    return getattr(importlib.import_module(module_name), class_name)


class SelectorRouter:
    """Adapt a ``select``/``release`` selector to verl's server-balancer API."""

    def __init__(self, servers: dict[str, Any], config: dict[str, Any]) -> None:
        selector_class = config.get("selector_class")
        if not selector_class:
            raise ValueError("SelectorRouter requires selector_class in its YAML config")
        selector_kwargs = {key: value for key, value in config.items() if key not in {"router_class", "selector_class"}}
        self._selector = _load_class(str(selector_class))(**selector_kwargs)
        self._servers = dict(servers)
        self._order = list(self._servers)

    def require_acquire_fields(self) -> list[str]:
        return ["prompt_ids"]

    def require_release_fields(self) -> list[str]:
        return ["request_id"]

    def acquire_server(
        self,
        request_id: str,
        prompt_ids: list[int] | None = None,
    ) -> tuple[str, Any]:
        candidates = [DecodeCandidate(key=key, handle=self._servers[key]) for key in self._order]
        selected = self._selector.select(
            {"routing_key": request_id, "prompt_ids": prompt_ids},
            candidates,
        )
        server_id = str(getattr(selected, "key", selected))
        if server_id not in self._servers:
            raise RuntimeError(f"selector returned unknown server: {server_id!r}")
        return server_id, self._servers[server_id]

    def release_server(self, server_id: str, request_id: str | None = None) -> None:
        release = getattr(self._selector, "release", None)
        if callable(release) and server_id in self._order:
            release(self._order.index(server_id))

    def add_servers(self, servers: dict[str, Any]) -> None:
        for server_id, handle in servers.items():
            if server_id not in self._servers:
                self._servers[server_id] = handle
                self._order.append(server_id)

    def remove_servers(self, server_ids: list[str]) -> None:
        for server_id in server_ids:
            self._servers.pop(server_id, None)
            if server_id in self._order:
                self._order.remove(server_id)

    def get_all_servers(self) -> list[str]:
        return list(self._order)

    def get_status(self) -> dict[str, Any]:
        return {
            "selector": type(self._selector).__name__,
            "servers": list(self._order),
            "total_inflight": self.get_total_inflight(),
            "active_servers": len(self._order),
        }

    def clear_sticky_cache(self) -> dict[str, Any]:
        reset = getattr(self._selector, "reset", None)
        if callable(reset):
            reset()
        return {"cleared_entries": 0, "server_loads": {}}

    def get_total_inflight(self) -> int:
        pending_tokens = getattr(self._selector, "_pending_tokens", None)
        if isinstance(pending_tokens, list):
            return sum(int(value) for value in pending_tokens)
        pending_requests = getattr(self._selector, "_pending_requests", None)
        if isinstance(pending_requests, list):
            return sum(int(value) for value in pending_requests)
        return 0
