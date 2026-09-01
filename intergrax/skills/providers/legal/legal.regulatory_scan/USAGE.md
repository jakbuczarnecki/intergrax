# `legal.regulatory_scan`

**Bundle:** `legal` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Regulatory lookup across web, wiki, and indexed corpus.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `legal` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `websearch.query`, `knowledge.search`, `rag.retrieve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `websearch.query` | Catalog tool |
| `knowledge.search` | Catalog tool |
| `rag.retrieve` | Catalog tool |

## Related skills

- Other `legal` bundle skills - see bundle [USAGE.md](../USAGE.md)
