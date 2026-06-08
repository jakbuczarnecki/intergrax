# `notify.scheduled_alerts`

**Bundle:** `notify` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Deferred notification scheduling with cancel and immediate send.

## How it works

notify.schedule/list/cancel + notify.send.

## How to use

notify_skill_profile(); wire notification_channel.

## What you get

Time-shifted alerts for long agent workflows.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `notify.schedule` | Schedule delivery |
| `notify.list_scheduled` | List pending |
| `notify.cancel_scheduled` | Cancel schedule |
| `notify.send` | Immediate send |

## Related skills

- `ops.incident_dispatch`
- `collaboration.outreach`
