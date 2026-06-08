# `harness.context_demo`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Context assembly exercises** for `ContextBudgetPolicy` and `ContextManager` — retrieval-only pack without web search noise. Use in lab tests for R-Context trimming and `CONTEXT_ASSEMBLED` trace events.

## How it works

Single tool `rag.retrieve`. Prompt ref `harness.context_demo.system` targets context engineering demos. No policy fragment.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_CONTEXT_DEMO

AgentContract(id="context_lab", skills=[HARNESS_CONTEXT_DEMO], ...)
```

## What you get

Isolated retrieval allow-list for context budget gate tests without extra tools.

## Tools unlocked

`rag.retrieve`
