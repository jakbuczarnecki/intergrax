# `message_bus.async_runner`

**Bundle:** `message_bus` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Async background tasks via message bus queue.

## How it works

message_bus.* via TaskQueue/MessageBus integration.

## How to use

message_bus_skill_profile(); wire message_bus slug.

## What you get

Long-running work without blocking sync Nexus loop.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `message_bus.enqueue` | Enqueue task |
| `message_bus.get_status` | Poll status |
| `message_bus.get_result` | Fetch result |

## Related skills

- `ops.workflow_runner`
- `eval.score_logger`
