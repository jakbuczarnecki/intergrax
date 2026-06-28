# `local.workspace.search`

**Bundle:** `local` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Semantic search and tenant-scoped evidence retrieval over locally indexed documents in the LKW pipeline.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy. Invokes `rag.retrieve` with workspace filters for tenant-scoped evidence.

## How to use

Enable bundle `local` on `SkillProfile` or attach this manifest to `AgentContract.skills` for `LocalSearchAgent`.

## What you get

Governed access to: `rag.retrieve`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.retrieve` | Semantic retrieval over tenant-scoped indexed documents |

## Related skills

- `local.workspace.index` — populate the index before search
- `local.workspace.synthesize` — consume search evidence in drafts
- Other `local` bundle skills — see bundle [USAGE.md](../USAGE.md)
