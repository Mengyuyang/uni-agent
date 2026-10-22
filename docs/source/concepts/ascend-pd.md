# Ascend PD routing: fork review guide

This change ports the routing portion of the corrected `uni-agent1013-dr.rar`
snapshot to current main. It is for review in the owner's fork, not an upstream
submission or a claim of hardware readiness.

## Responsibilities

- Uni-Agent selectors choose a decode server; they do not copy KV buffers.
- The companion verl change wraps selectors in its request-load-balancer
  protocol, runs P then D, and owns P/D engine lifecycle and metrics discovery.
- A matched vLLM-Ascend `MooncakeConnectorV1` and Mooncake build own KV transfer.
  Their dependency patches are not included in these two repository changes.

The current training architecture is retained. Retired blackbox runners are not
restored. Existing fork commits remain in the synchronized main's Git ancestry.

## Selecting a policy

In the companion verl configuration:

```yaml
actor_rollout_ref:
  rollout:
    disaggregation:
      enabled: true
      transfer_backend: mooncake
      decode_policy:
        router_config_path: pkg://uni_agent.rollout.configs/session_sticky.yaml
```

This is a configuration fragment, not a complete training recipe. P/D topology,
model, resource allocation and runtime dependencies must also be configured.
Set the path to null to use verl's built-in sticky/least-inflight router without
these optional selectors. The seven YAML resources are included in wheels.

| Policy | Decision |
| --- | --- |
| random | Random candidate; comparison baseline |
| least_load / power_of_two | Sample candidates and minimize local in-flight requests |
| consistent_hash | Virtual-node hash ring keyed by the session |
| rendezvous_hash | Highest hash score for session/server pair |
| cache_aware | Estimated token-prefix ownership, with load-imbalance fallback |
| session_sticky | Session hash affinity, with KV/queue/token-pressure fallback |

Cache-aware ownership is an estimate based on routed prompts, not proof that
the engine still holds those KV blocks. Cache invalidation must reset that state.
Session-sticky uses the session identity to avoid concentrating unrelated SWE
tasks that share the same system/tool prompt on a single decode replica.

## Source and porting differences

The original routing implementation and seven policy configs are retained.
The port adds package-data inclusion and moves the verl candidate type import
under `TYPE_CHECKING`, so importing the pure policies does not start importing
the entire verl/Torch runtime. Unused Ray import and formatting are cleaned up.
The mounted Claude Code executable check is a sandbox startup optimization, not
a KV-transfer feature; it prefers `/opt/claude-code/bin/claude` before installers.

## Known limitations and merge gates

The source's session-sticky policy releases prompt-token weights FIFO per
server. If requests finish out of order, pending-token estimates are incorrect:
after selecting 10000 then 1000 tokens, finishing the short request first leaves
1000 recorded rather than 10000. Request-exact reservation/release needs follow-up
before relying on this optional policy for accurate pressure decisions.

All-hot routing still dispatches a request; this is not admission control.
Metrics snapshots can be stale or absent. The source's performance numbers in
comments are colleague observations, not independently reproduced benchmarks.
Changing the candidate pool resets local bookkeeping and needs drain-aware use.

Run from this repository root:

```bash
python -m pytest -q tests/test_decode_selectors.py tests/uni_agent/agents/test_claude_code_agent.py --noconftest
```

Local result: **51 passed** on Python 3.12; changed-file Ruff lint/format and
Python compilation passed. These tests do not run NPU training, actual sandbox
installation, Ray routing, or Mooncake transfer. The companion verl application
tests were blocked locally by a Windows PyTorch DLL load failure.

AI assistance was used in analysis, porting, tests and documentation. Human
line-by-line review and target-hardware validation remain required before merge.
