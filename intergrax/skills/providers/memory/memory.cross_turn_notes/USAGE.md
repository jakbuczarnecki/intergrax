# `memory.cross_turn_notes`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Cross-turn note taking with list/read/write task memory.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `memory` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `memory.write`, `memory.list_keys`, `memory.read`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `memory.write` | Catalog tool |
| `memory.list_keys` | Catalog tool |
| `memory.read` | Catalog tool |

## Related skills

- Other `memory` bundle skills - see bundle [USAGE.md](../USAGE.md)
