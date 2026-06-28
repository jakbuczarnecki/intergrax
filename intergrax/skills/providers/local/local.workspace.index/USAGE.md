# `local.workspace.index`

**Bundle:** `local` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Index user-local source paths into a tenant-scoped RAG vector store for the Local Knowledge Workspace (LKW) pipeline.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy. Invokes `rag.ingest_document` with metadata source paths from the workspace environment profile.

## How to use

Enable bundle `local` on `SkillProfile` or attach this manifest to `AgentContract.skills` for `LocalIndexerAgent`.

## What you get

Governed access to: `rag.ingest_document`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.ingest_document` | Ingest documents into tenant-scoped RAG index |

## Related skills

- `local.workspace.search` — retrieve indexed evidence
- `local.workspace.synthesize` — draft from retrieved evidence
- Other `local` bundle skills — see bundle [USAGE.md](../USAGE.md)
