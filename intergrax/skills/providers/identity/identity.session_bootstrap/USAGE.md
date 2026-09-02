# `identity.session_bootstrap`

**Bundle:** `identity` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Bootstrap session from verified identity and memory seed.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `identity` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `identity.verify_token`, `identity.get_user`, `memory.write`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `identity.verify_token` | Catalog tool |
| `identity.get_user` | Catalog tool |
| `memory.write` | Catalog tool |

## Related skills

- Other `identity` bundle skills - see bundle [USAGE.md](../USAGE.md)
