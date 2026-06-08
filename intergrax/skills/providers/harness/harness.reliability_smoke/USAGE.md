# `harness.reliability_smoke`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

**Reliability / W-OPS smoke** (W-OPS.8): observability query, RAG read, security scan, workflow trigger. Exercises idempotent-friendly and side-effect tools when P6 integration stack is wired on lab host.

## How it works

Four-tool union — broader than `harness.tool_smoke`. Security and workflow tools resolve only when integrations present; skill declaration is unconditional.

## How to use

```python
from intergrax.skills.providers.harness.manifests import HARNESS_RELIABILITY_SMOKE

AgentContract(id="reliability_lab", skills=[HARNESS_RELIABILITY_SMOKE], ...)
```

## What you get

Single pack for W-OPS reliability gate coverage.

## Tools unlocked

`observability.query_traces`, `rag.retrieve`, `security.scan`, `workflow.trigger`

## Related skills

- `ops.security_audit` — production security audit variant
- `ops.workflow_runner` — full workflow poll/logs
