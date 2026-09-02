# `memory.fact_extractor`

**Bundle:** `memory` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Extract durable facts into LTM with context summarization.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `memory` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `ltm.write_fact`, `memory.read`, `context.summarize`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `ltm.write_fact` | Catalog tool |
| `memory.read` | Catalog tool |
| `context.summarize` | Catalog tool |

## Related skills

- Other `memory` bundle skills - see bundle [USAGE.md](../USAGE.md)
