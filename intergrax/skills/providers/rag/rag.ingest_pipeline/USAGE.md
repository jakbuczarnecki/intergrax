# `rag.ingest_pipeline`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

End-to-end ingest: parse, ingest, and index readiness check.

## How it works

Resolves registered `tool_id`s at agent bind time via `SkillResolver`; tools execute through `ToolRuntime` under host policy.

## How to use

Enable bundle `rag` on `SkillProfile` or attach this manifest to `AgentContract.skills`.

## What you get

Governed access to: `document.parse`, `rag.ingest_document`, `rag.check_index_status`.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `document.parse` | Catalog tool |
| `rag.ingest_document` | Catalog tool |
| `rag.check_index_status` | Catalog tool |

## Related skills

- Other `rag` bundle skills - see bundle [USAGE.md](../USAGE.md)
