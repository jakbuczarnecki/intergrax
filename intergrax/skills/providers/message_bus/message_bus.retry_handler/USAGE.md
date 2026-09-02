# `message_bus.retry_handler`

**Bundle:** `message_bus` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Retry failed async tasks via re-enqueue and status poll.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `message_bus` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `message_bus.get_status`, `message_bus.enqueue`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `message_bus.get_status` | Catalog tool |
| `message_bus.enqueue` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `message_bus` bundle skills - see bundle [USAGE.md](../USAGE.md)
