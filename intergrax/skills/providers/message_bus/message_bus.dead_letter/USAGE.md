# `message_bus.dead_letter`

**Bundle:** `message_bus` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Dead-letter hygiene: list, purge completed, and log search.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `message_bus` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `message_bus.list_tasks`, `message_bus.purge_completed`, `logs.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `message_bus.list_tasks` | Catalog tool |
| `message_bus.purge_completed` | Catalog tool |
| `logs.search` | Catalog tool |

## Related skills

- Other `message_bus` bundle skills — see bundle [USAGE.md](../USAGE.md)
