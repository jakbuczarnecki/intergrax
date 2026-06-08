# `rag.collection_lifecycle`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `high`

## Purpose

Controlled index lifecycle: metadata search, delete, and purge.

## How it works

HIGH risk tier; destructive tools gated by ToolProfile and policy.

## How to use

Admin-only hosts; pair with rag.index_admin before purge.

## What you get

Grouped destructive ops under one governed skill.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.search_by_metadata` | Metadata filter scan |
| `rag.delete_documents` | Delete by document id |
| `rag.purge_collection` | Controlled collection purge |

## Related skills

- `rag.index_admin`
- `rag.document_ingest`
