# `cache.warm_prefetch`

**Bundle:** `cache` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Warm session cache from retrieval results.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `cache` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `cache.set`, `cache.get`, `rag.retrieve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `cache.set` | Catalog tool |
| `cache.get` | Catalog tool |
| `rag.retrieve` | Catalog tool |

## Related skills

- Other `cache` bundle skills — see bundle [USAGE.md](../USAGE.md)
