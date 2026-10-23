# Ascend PD routing: final-local snapshot

Updated: 2026-10-23. This fork PR now follows `uni-agent1013-dr-final-local.rar`
(SHA256 `8AF98361FE5C19978106076F1B637687DC179F3A72ED656AD617FDA54124F9D5`).
It supersedes the earlier PD-specific adapter/configuration design.

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

**Receiving request_id is not request-exact release.** The final source still
ignores that argument inside release and calls `selector.release(index)`.
Session-sticky pops prompt weights FIFO: selecting 10000 then 1000 tokens and
finishing the short request first records 1000 instead of 10000 outstanding.
A characterization test preserves this known limitation; it does not certify
correct accounting. A follow-up needs unique per-call reservations separate
from the session-affinity key.

`get_total_inflight` returns token weight for session-sticky, request count for
least-load/cache-aware, and zero for stateless strategies. These are not uniform
request-count semantics. Pool changes reset index-based estimates; use them
with drain-aware lifecycle management.

All-hot routing still dispatches and is not admission control. Metrics can be
stale/missing. Source-comment performance numbers were not independently
reproduced. Final-local contains no dependency tree; matched vLLM-Ascend/Mooncake
with the required connector/RPC remains an external deployment prerequisite.

## Validation and provenance

```bash
python -m pytest -q tests/test_decode_selectors.py tests/test_router_balancer.py tests/uni_agent/agents/test_claude_code_agent.py --noconftest
```

Local Python 3.12 result: **59 passed, 7 subtests passed**. Tests use actual
policies/adapter and a no-network pressure source. They do not validate Ray
actor startup, NPU, actual sandbox installation, or KV transport.
Changed-file pre-commit checks (Ruff, formatting, mypy and compileall) passed.

Porting additionally preserves package-data inclusion, avoids unused runtime
Ray/Torch imports, and extends adapter regression coverage. The private launchers
and archived runtime credentials are not included. In the archive,
`run_train_baseline.sh` and the launcher tests still reference removed schema
and missing dependencies; do not treat them as validated final recipes.

See [companion verl PR](https://github.com/Mengyuyang/verl/pull/2).
AI assistance was used; human line review and target-hardware validation remain
required before merge. This is a fork-only draft, not an upstream submission.
