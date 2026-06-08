# `hitl.escalation_router`

**Bundle:** `hitl` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Escalate HITL queue depth to PagerDuty and notify.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `hitl` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `hitl.summarize_queue`, `pagerduty.trigger_incident`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `hitl.summarize_queue` | Catalog tool |
| `pagerduty.trigger_incident` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `hitl` bundle skills — see bundle [USAGE.md](../USAGE.md)
