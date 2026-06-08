# `harness.integration_bridge_smoke`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Integration bridge smoke** (T-EXPAND): provider-agnostic `storage.get` and `knowledge.search` paths. Validates `ToolWiringContext` binds integration slugs to catalog tools without agent importing vendors.

## How it works

1. `storage.get` → `ObjectStorage` integration.
2. `knowledge.search` → `WikiKnowledge` integration.
3. Gate test `test_harness_integration_bridge_smoke_merges_bridge_tools` validates resolver output.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_INTEGRATION_BRIDGE_SMOKE

AgentContract(id="bridge_lab", skills=[HARNESS_INTEGRATION_BRIDGE_SMOKE], ...)
```

Wire object storage + wiki slugs on lab integration profile.

## What you get

Proof that integration → tool bridge works for storage and knowledge categories.

## Tools unlocked

`storage.get`, `knowledge.search`
