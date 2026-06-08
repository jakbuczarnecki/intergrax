# `health.integration_probe`

**Bundle:** `health` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Integration health probes for operator dashboards.

## How it works

health.check_* tools against wired integrations.

## How to use

health_skill_profile(); run from harness operator agents.

## What you get

Pre-flight backend readiness without custom scripts.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `health.check_integration` | Probe integration slug |
| `health.check_profile` | Validate environment profile |
| `health.check_relational_store` | Relational store probe |

## Related skills

- `ops.trace_debug`
- `vector_store.admin`
