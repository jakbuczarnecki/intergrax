# `rag.semantic_qa`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Semantic Q&A with memory search and document fetch.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `rag` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `rag.retrieve`, `rag.get_document`, `memory.search`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Catalog tool |
| `rag.get_document` | Catalog tool |
| `memory.search` | Catalog tool |

## Related skills

- Other `rag` bundle skills - see bundle [USAGE.md](../USAGE.md)
