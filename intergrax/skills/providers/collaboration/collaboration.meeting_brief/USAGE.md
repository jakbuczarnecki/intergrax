# `collaboration.meeting_brief`

**Bundle:** `collaboration` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Meeting brief from calendar, user profile, and workspace draft.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `collaboration` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `collaboration.list_calendar`, `collaboration.get_user`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `collaboration.list_calendar` | Catalog tool |
| `collaboration.get_user` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `collaboration` bundle skills - see bundle [USAGE.md](../USAGE.md)
