# `ops.oncall_runbook`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

On-call runbook: logs, traces, and stakeholder notification.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `logs.search`, `observability.query_traces`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `logs.search` | Catalog tool |
| `observability.query_traces` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `ops` bundle skills — see bundle [USAGE.md](../USAGE.md)
