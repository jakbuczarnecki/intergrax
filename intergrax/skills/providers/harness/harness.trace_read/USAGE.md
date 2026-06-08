# `harness.trace_read`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Harness run trace inspection**: read persisted SQLite harness runs, filter events, and query external observability backends. Use in lab debugging before escalating to `ops.trace_debug`.

## How it works

Unions harness-local tools (`harness.get_run`, `harness.get_run_events`) with `observability.query_traces` for vendor traces.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_TRACE_READ

AgentContract(id="trace_lab", skills=[HARNESS_TRACE_READ], ...)
```

Requires trace DB path on harness host bootstrap.

## What you get

Local + vendor trace read in one smoke-oriented pack.

## Tools unlocked

`harness.get_run`, `harness.get_run_events`, `observability.query_traces`
