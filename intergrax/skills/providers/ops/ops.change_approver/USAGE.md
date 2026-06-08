# `ops.change_approver`

**Bundle:** `ops` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Change approval loop: HITL pending, notify, workflow poll.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `ops` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `hitl.list_pending`, `notify.send`, `workflow.poll`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `hitl.list_pending` | Catalog tool |
| `notify.send` | Catalog tool |
| `workflow.poll` | Catalog tool |

## Related skills

- Other `ops` bundle skills — see bundle [USAGE.md](../USAGE.md)
