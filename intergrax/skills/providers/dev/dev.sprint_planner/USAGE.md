# `dev.sprint_planner`

**Bundle:** `dev` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Sprint planning with issues, calendar, and scratchpad memory.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `dev` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `issues.search`, `collaboration.list_calendar`, `memory.write`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `issues.search` | Catalog tool |
| `collaboration.list_calendar` | Catalog tool |
| `memory.write` | Catalog tool |

## Related skills

- Other `dev` bundle skills — see bundle [USAGE.md](../USAGE.md)
