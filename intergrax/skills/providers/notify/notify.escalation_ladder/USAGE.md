# `notify.escalation_ladder`

**Bundle:** `notify` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Escalation ladder: schedule, send, and PagerDuty trigger.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `notify` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `notify.schedule`, `notify.send`, `pagerduty.trigger_incident`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `notify.schedule` | Catalog tool |
| `notify.send` | Catalog tool |
| `pagerduty.trigger_incident` | Catalog tool |

## Related skills

- Other `notify` bundle skills - see bundle [USAGE.md](../USAGE.md)
