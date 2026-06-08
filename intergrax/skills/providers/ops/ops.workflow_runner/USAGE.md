# `ops.workflow_runner`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Orchestrated batch jobs**: trigger eval runs, RAG refresh pipelines, or Prefect/Airflow workflows and monitor completion. Use when agents coordinate offline work — not for synchronous user chat.

## How it works

1. `workflow.trigger` starts a run on `workflow_orchestrator` integration.
2. `workflow.poll` checks run status until terminal state.
3. `workflow.fetch_logs` retrieves tail logs for debugging failed runs.
4. Mirrors `harness.reliability_smoke` workflow slice as a dedicated product skill.

## How to use

```python
from intergrax.skills.providers.ops.manifests import OPS_WORKFLOW_RUNNER

AgentContract(id="eval_runner", skills=[OPS_WORKFLOW_RUNNER], ...)
```

Wire `workflow_orchestrator` slug (`prefect`, `airflow`, `temporal`, etc.).

## What you get

| Benefit | Detail |
|---------|--------|
| **Batch eval automation** | Standard trigger/poll/logs trio |
| **RAG refresh ops** | Re-index corpora on schedule via agent |
| **Orchestrator-agnostic** | Swap Prefect/Airflow at Tier-3 |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workflow.trigger` | Start workflow run |
| `workflow.poll` | Poll run status |
| `workflow.fetch_logs` | Fetch run logs |

## Related skills

- `rag.document_ingest` — often chained after refresh workflow
- `ops.trace_debug` — debug failed workflow runs
