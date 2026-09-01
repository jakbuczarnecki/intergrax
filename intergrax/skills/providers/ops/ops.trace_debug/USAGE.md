# `ops.trace_debug`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

**Agent run debugging** for platform engineers: query LLM/agent traces, search logs, and capture errors. Use in lab harness investigations, post-mortems, and W-OPS reliability exercises - without granting full observability admin tools ad hoc.

## How it works

Resolves observability catalog tools bound to `ObservabilityBackend` on `ToolWiringContext` (`langfuse`, `otel`, `elasticsearch`, etc.). Each call goes through `ToolRuntime` with trace emission. Skill is read-oriented (query/search/capture), not mutating production state.

## How to use

```python
from intergrax.skills.providers.ops.manifests import OPS_TRACE_DEBUG
from intergrax.applications._shared.skill_wiring import ops_skill_profile

env.skill_profile = ops_skill_profile()
AgentContract(id="debug_agent", skills=[OPS_TRACE_DEBUG], ...)
```

Wire `observability_backend` integration slug on host profile.

## What you get

| Benefit | Detail |
|---------|--------|
| **Focused debug surface** | Three tools instead of full observability bundle |
| **Harness-aligned** | Complements `harness.trace_read` with vendor backends |
| **Error correlation** | `errors.capture` links to Sentry-style backends |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `observability.query_traces` | Query agent/LLM trace spans |
| `logs.search` | Search centralized log index |
| `errors.capture` | Report or fetch error events |

## Related skills

- `harness.trace_read` - SQLite harness run traces (local lab)
- `ops.incident_dispatch` - escalate after debug
