# `storage.backup_sync`

**Bundle:** `storage` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Backup sync between object storage and workspace snapshot.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `storage` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `storage.get`, `storage.put`, `workspace.snapshot`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `storage.get` | Catalog tool |
| `storage.put` | Catalog tool |
| `workspace.snapshot` | Catalog tool |

## Related skills

- Other `storage` bundle skills — see bundle [USAGE.md](../USAGE.md)
