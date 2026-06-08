# `hitl.decision_auditor`

**Bundle:** `hitl` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Audit HITL decisions with trace correlation.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `hitl` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `hitl.get_decision`, `hitl.list_for_task`, `observability.query_traces`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `hitl.get_decision` | Catalog tool |
| `hitl.list_for_task` | Catalog tool |
| `observability.query_traces` | Catalog tool |

## Related skills

- Other `hitl` bundle skills — see bundle [USAGE.md](../USAGE.md)
