"""CPU tests for the uni-agent-to-verl router protocol adapter."""

import unittest
from importlib.resources import files

from omegaconf import OmegaConf

from uni_agent.rollout.router_balancer import SelectorRouter


class _NoPressure:
    def observe(self, keys):
        pass

    def snapshot(self, key):
        return None


class TestSelectorRouter(unittest.TestCase):
    def setUp(self):
        self.servers = {"http://decode-a:1": object(), "http://decode-b:1": object()}
        self.router = SelectorRouter(
            self.servers,
            {
                "router_class": "uni_agent.rollout.router_balancer.SelectorRouter",
                "selector_class": "uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector",
                "pressure_cache": _NoPressure(),
            },
        )

    def test_session_sticky_acquire_and_release(self):
        first_id, first_handle = self.router.acquire_server(
            "session-a",
            prompt_ids=list(range(32)),
        )
        second_id, second_handle = self.router.acquire_server(
            "session-a",
            prompt_ids=list(range(32)),
        )
        self.assertEqual(first_id, second_id)
        self.assertIs(first_handle, second_handle)
        self.assertEqual(self.router.get_total_inflight(), 64)
        self.router.release_server(first_id, "session-a")
        self.assertEqual(self.router.get_total_inflight(), 32)

    def test_server_pool_can_be_updated(self):
        new_handle = object()
        self.router.add_servers({"http://decode-c:1": new_handle})
        self.assertIn("http://decode-c:1", self.router.get_all_servers())
        self.router.remove_servers(["http://decode-a:1"])
        self.assertNotIn("http://decode-a:1", self.router.get_all_servers())

    def test_session_sticky_yaml_is_upstream_router_compatible(self):
        config = OmegaConf.to_container(
            OmegaConf.create(files("uni_agent.rollout.configs").joinpath("session_sticky.yaml").read_text()),
            resolve=True,
        )
        config["pressure_cache"] = _NoPressure()
        router = SelectorRouter(self.servers, config)
        server_id, _ = router.acquire_server("yaml-session", prompt_ids=[1, 2, 3])
        self.assertIn(server_id, self.servers)

    def test_declares_upstream_protocol_fields(self):
        self.assertEqual(self.router.require_acquire_fields(), ["prompt_ids"])
        self.assertEqual(self.router.require_release_fields(), ["request_id"])

    def test_all_packaged_policies_load(self):
        for policy in (
            "random",
            "least_load",
            "power_of_two",
            "consistent_hash",
            "rendezvous_hash",
            "cache_aware",
            "session_sticky",
        ):
            with self.subTest(policy=policy):
                config = OmegaConf.to_container(
                    OmegaConf.create(files("uni_agent.rollout.configs").joinpath(f"{policy}.yaml").read_text()),
                    resolve=True,
                )
                if policy == "session_sticky":
                    config["pressure_cache"] = _NoPressure()
                router = SelectorRouter(self.servers, config)
                server_id, handle = router.acquire_server("test-session", [1, 2, 3])
                self.assertIs(handle, self.servers[server_id])
                router.release_server(server_id, "test-session")

    def test_reset_preserves_pending_tokens(self):
        self.router.acquire_server("a", [1, 2, 3])
        self.router.clear_sticky_cache()
        self.assertEqual(self.router.get_total_inflight(), 3)

    def test_missing_selector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selector_class"):
            SelectorRouter(self.servers, {})

    def test_legacy_release_remains_fifo_not_request_exact(self):
        # Characterize the known source limitation; this is not a correctness claim.
        self.router = SelectorRouter(
            {"d": object()},
            {
                "selector_class": "uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector",
                "pressure_cache": _NoPressure(),
            },
        )
        self.router.acquire_server("long", [0] * 10000)
        self.router.acquire_server("short", [0] * 1000)
        self.router.release_server("d", "short")
        self.assertEqual(self.router.get_total_inflight(), 1000)  # True outstanding weight would be 10000.


if __name__ == "__main__":
    unittest.main()
