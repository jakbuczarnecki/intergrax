# `harness.tool_smoke`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Lab smoke test** for core catalog tool wiring: RAG retrieve + web search. Used by `EchoAgent`, `SignoffProbeAgent`, and gate tests to prove Tier-3 tool/skill pipeline works end-to-end.

## How it works

Resolves `rag.retrieve` and `websearch.query`. Minimal two-tool pack - parent of `harness.stack_demo` via `requires_skills`. Emits `SKILL_RESOLVED` on agent register when event bus wired.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_TOOL_SMOKE
from intergrax.applications._shared.skill_wiring import harness_platform_skill_profile

AgentContract(id="echo", skills=[HARNESS_TOOL_SMOKE], ...)
```

## What you get

| Benefit | Detail |
|---------|--------|
| **Fast harness validation** | Smallest real skill pack |
| **Gate test anchor** | `test_harness_reference_agent_skills` |
| **Dependency demo root** | Required by `harness.stack_demo` |

## Tools unlocked

`rag.retrieve`, `websearch.query`
