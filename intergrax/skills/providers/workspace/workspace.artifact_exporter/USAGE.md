# `workspace.artifact_exporter`

**Bundle:** `workspace` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Export workspace artifacts to durable object storage.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `workspace` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `workspace.export_artifact`, `storage.put`, `workspace.list_files`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `workspace.export_artifact` | Catalog tool |
| `storage.put` | Catalog tool |
| `workspace.list_files` | Catalog tool |

## Related skills

- Other `workspace` bundle skills - see bundle [USAGE.md](../USAGE.md)
