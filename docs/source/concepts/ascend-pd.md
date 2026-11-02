# Ascend PD routing: final-local snapshot

Updated: 2026-10-31. This fork PR follows `uni-agent1013-dr-final-local.rar`
(SHA256 `8AF98361FE5C19978106076F1B637687DC179F3A72ED656AD617FDA54124F9D5`).
It supersedes the earlier PD-specific adapter/configuration design.
The October 31 fix adds request-exact PD accounting beyond that source and
merges the existing fork main at `c78d6c4`; it does not update main itself.

## Responsibilities and configuration

Uni-Agent owns the pure selector classes, `DecodeCandidate`, and the new
`uni_agent.rollout.router_balancer.SelectorRouter`. The adapter implements
verl's standard router protocol. The companion verl change calls its existing
`get_router_handle` factory; no verl-local `pd_decode_selector.py` is needed.

```yaml
actor_rollout_ref:
  rollout:
    disaggregation:
      enabled: true
      transfer_backend: mooncake
    custom:
      pd_router_config_path: /absolute/path/to/session_sticky.yaml
```

Example policy YAML:

```yaml
router_class: uni_agent.rollout.router_balancer.SelectorRouter
selector_class: uni_agent.rollout.decode_selectors.SessionStickyDecodeSelector
inflight_token_budget: 2000000
```

This is a fragment, not a complete training recipe. The router path must be
readable in the Ray actor's environment. The upstream loader expects a normal
filesystem path, not the previous `pkg://...` resource URL. YAML files remain
packaged in the Uni-Agent wheel; `importlib.resources.files` can locate them.

The former `disaggregation.decode_policy.router_config_path` and role-specific
memory/batch schema are removed from the companion verl PR. An unset
`custom.pd_router_config_path` now retains PD round-robin, not the previous
default sticky/least-inflight router. Custom PD routing is Ascend-only; GPU PD
keeps its upstream selection and connector behavior.

## Policies

| YAML | Decision |
| --- | --- |
| random | Random candidate |
| least_load / power_of_two | Same class: sample k candidates, default k=2, minimize local in-flight count |
| consistent_hash | Virtual-node hash ring keyed by session |
| rendezvous_hash | Highest session/server hash score |
| cache_aware | Estimated token-prefix ownership with load-imbalance fallback |
| session_sticky | Session hash affinity with KV/queue/token-pressure fallback |

SWE tasks share system/tool prefixes, so prefix ownership alone can concentrate
unrelated sessions on one D. Session identity helps spread those sessions while
retaining cross-turn affinity. This does not guarantee actual engine KV hits.

The mounted Claude Code executable check is retained from the previous source.
It optimizes sandbox startup and is not part of KV transfer.

## Adapter lifecycle and known limits

The adapter forwards `prompt_ids`, maps candidates to server IDs/handles, and
exposes acquire/release, server-pool updates, status and cache reset.

**The updated PD path separates affinity from reservation identity.**
`request_id` at the router remains the session-affinity key; the companion verl
dispatch supplies a fresh `reservation_id` on every call, including concurrent
turns/retries with identical session/request IDs. It is an optional declared
acquire/release field, not a new rollout configuration setting.

The adapter records reservation-to-server ownership. Session-sticky records
each reservation's prompt weight and exposes `release_request(index, id)`.
Finishing the 1000-token request before the 10000-token request now leaves
10000 outstanding. Duplicate/unknown releases are no-ops, wrong-server releases
are rejected without discarding ownership, and duplicate active acquires fail
before incrementing load. Cache reset preserves active reservations.

Upgrade both companion PRs together. Callers without `reservation_id` fall back
to `request_id` as identity and must keep it unique across active calls; a
duplicate active ID is rejected (the PD wrapper may then fall back to RR).
Direct selector callers without identity still have the legacy FIFO API; the
fixed PD adapter never uses it for session-sticky reservations. Reservation IDs
must not be reused for retries, since a late release can otherwise match a new call.

Pool membership changes are rejected while reservations are active. Drain
before adding/removing nodes to prevent index-based policies from resetting
another request's accounting. This is an intentional lifecycle restriction,
not transparent hot-resizing support.

`get_total_inflight` returns token weight for session-sticky, request count for
least-load/cache-aware, and zero for stateless strategies. These are not uniform
request-count semantics; in particular zero token weight is not proof that no
reservations exist.

All-hot routing still dispatches and is not admission control. Metrics can be
stale/missing. Source-comment performance numbers were not independently
reproduced. Final-local contains no dependency tree; matched vLLM-Ascend/Mooncake
with the required connector/RPC remains an external deployment prerequisite.
The local ledger is not a distributed transaction: actor failure or cancellation
while an acquire RPC result is being delivered still needs lifecycle recovery.

## Validation and provenance

```bash
python -m pytest -q tests/test_decode_selectors.py tests/test_router_balancer.py tests/uni_agent/agents/test_claude_code_agent.py --noconftest
```

Local Python 3.12 result: **66 passed, 7 subtests passed**. Tests use actual
policies/adapter and a no-network pressure source. They do not validate Ray
actor startup, NPU, actual sandbox installation, or KV transport.
Changed-file pre-commit checks (Ruff, formatting, mypy and compileall) passed.
Paired Python-method checks also ran actual verl dispatch with this adapter:
short-first completion and short-request cancellation both produced
`11000 -> 10000 -> 0` with empty ledgers afterwards. These are not Ray/NPU tests.

Porting additionally preserves package-data inclusion, avoids unused runtime
Ray/Torch imports, and extends adapter regression coverage. The private launchers
and archived runtime credentials are not included. In the archive,
`run_train_baseline.sh` and the launcher tests still reference removed schema
and missing dependencies; do not treat them as validated final recipes.

See [companion verl PR](https://github.com/Mengyuyang/verl/pull/2).
AI assistance was used; human line review and target-hardware validation remain
required before merge. This is a fork-only draft, not an upstream submission.
