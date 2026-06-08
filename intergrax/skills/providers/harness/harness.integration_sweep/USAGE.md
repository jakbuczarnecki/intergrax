# `harness.integration_sweep`

**Bundle:** `harness` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Integration sweep with catalog introspection and skill resolve.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `harness` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `health.check_integration`, `catalog.list_tools`, `skill.resolve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `health.check_integration` | Catalog tool |
| `catalog.list_tools` | Catalog tool |
| `skill.resolve` | Catalog tool |

## Related skills

- Other `harness` bundle skills — see bundle [USAGE.md](../USAGE.md)
