# `harness.stack_demo`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**`requires_skills` demonstration** (W-OPS.9): shows transitive skill composition - parent `harness.tool_smoke` tools merge before this pack's `websearch.read_url` is added.

## How it works

1. `requires_skills=("harness.tool_smoke",)` - resolver visits dependency first.
2. Expanded `skill_ids` order: `harness.tool_smoke` → `harness.stack_demo`.
3. Merged tools: `rag.retrieve`, `websearch.query`, `websearch.read_url`.
4. Gate: `test_harness_requires_skills_demo.py`.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_STACK_DEMO

AgentContract(id="stack_lab", skills=[HARNESS_STACK_DEMO], ...)
```

Parent skill must be registered in same `SkillRegistry` (same harness bundle).

## What you get

Canonical example for skill stacking without duplicating parent `tool_ids` on the contract.

## Tools unlocked

| Source | `tool_id` |
|--------|-----------|
| Parent `harness.tool_smoke` | `rag.retrieve`, `websearch.query` |
| This pack | `websearch.read_url` |

## Related skills

- `harness.tool_smoke` - required dependency
- `legal.clause_compare` - production `requires_skills` pattern
