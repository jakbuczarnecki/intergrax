# `filesystem.stat_auditor`

**Bundle:** `filesystem` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Filesystem audit: stat, list, and read for operator hosts.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `filesystem` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `filesystem.stat`, `filesystem.list`, `filesystem.read_text`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `filesystem.stat` | Catalog tool |
| `filesystem.list` | Catalog tool |
| `filesystem.read_text` | Catalog tool |

## Related skills

- Other `filesystem` bundle skills - see bundle [USAGE.md](../USAGE.md)
