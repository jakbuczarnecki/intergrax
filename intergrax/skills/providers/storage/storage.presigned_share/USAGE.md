# `storage.presigned_share`

**Bundle:** `storage` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Presigned URL sharing with existence check and notify.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `storage` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `storage.presigned_url`, `storage.exists`, `notify.send`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `storage.presigned_url` | Catalog tool |
| `storage.exists` | Catalog tool |
| `notify.send` | Catalog tool |

## Related skills

- Other `storage` bundle skills - see bundle [USAGE.md](../USAGE.md)
