# `message_bus.task_admin`

**Bundle:** `message_bus` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Message bus task queue administration.

## How it works

message_bus.list_tasks + cancel + purge_completed.

## How to use

message_bus_skill_profile(); operator cleanup agents.

## What you get

Queue hygiene complement to async_runner.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `message_bus.list_tasks` | List queued tasks |
| `message_bus.cancel` | Cancel task |
| `message_bus.purge_completed` | Purge completed tasks |

## Related skills

- `message_bus.async_runner`
- `ops.workflow_admin`
