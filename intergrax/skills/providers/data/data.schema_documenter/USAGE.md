# `data.schema_documenter`

**Bundle:** `data` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Schema documentation for SQL and records stores.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `data` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `database.describe_schema`, `records.describe_collection`, `workspace.write_file`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `database.describe_schema` | Catalog tool |
| `records.describe_collection` | Catalog tool |
| `workspace.write_file` | Catalog tool |

## Related skills

- Other `data` bundle skills — see bundle [USAGE.md](../USAGE.md)
