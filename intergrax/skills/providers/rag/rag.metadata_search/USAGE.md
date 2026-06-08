# `rag.metadata_search`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Metadata-filtered document discovery without destructive ops.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `rag` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `rag.search_by_metadata`, `rag.list_documents`, `rag.describe_collection`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.search_by_metadata` | Catalog tool |
| `rag.list_documents` | Catalog tool |
| `rag.describe_collection` | Catalog tool |

## Related skills

- Other `rag` bundle skills — see bundle [USAGE.md](../USAGE.md)
