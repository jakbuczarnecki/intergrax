# `data.pipeline_probe`

**Bundle:** `data` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Data pipeline health: SQL probe, records query, store check.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `data` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `database.query`, `records.query`, `health.check_relational_store`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `database.query` | Catalog tool |
| `records.query` | Catalog tool |
| `health.check_relational_store` | Catalog tool |

## Related skills

- Other `data` bundle skills — see bundle [USAGE.md](../USAGE.md)
