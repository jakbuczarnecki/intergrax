# `harness.policy_smoke`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Policy bundle smoke** (W-OPS.8): low-risk tools under harness policy fragment `harness.policy_smoke`. Demonstrates `policy_fragment_id` on manifests for governance trace and capability graph edges.

## How it works

1. Tools: `rag.retrieve`, `websearch.query`.
2. `policy_fragment_id="harness.policy_smoke"` linked in capability graph.
3. Auto-merge into `RuntimePolicyBundle` planned (SK-BRIDGE.2).

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_POLICY_SMOKE

AgentContract(id="policy_lab", skills=[HARNESS_POLICY_SMOKE], ...)
```

## What you get

Reference skill for policy fragment declaration and W-OPS policy wiring tests.

## Tools unlocked

`rag.retrieve`, `websearch.query`

## Policy fragment

`harness.policy_smoke`
