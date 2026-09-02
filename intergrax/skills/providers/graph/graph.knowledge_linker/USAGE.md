# `graph.knowledge_linker`

**Bundle:** `graph` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Link graph entities to RAG grounding and LTM facts.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `graph` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `graph.run_query`, `rag.retrieve`, `ltm.write_fact`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `graph.run_query` | Catalog tool |
| `rag.retrieve` | Catalog tool |
| `ltm.write_fact` | Catalog tool |

## Related skills

- Other `graph` bundle skills - see bundle [USAGE.md](../USAGE.md)
