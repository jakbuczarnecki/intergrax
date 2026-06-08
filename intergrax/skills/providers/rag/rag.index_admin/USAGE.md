# `rag.index_admin`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `low`

## Purpose

Vector index introspection for operators and indexer agents.

## How it works

Read-only RAG admin tools resolved at registration via SkillResolver.

## How to use

SkillProfile(enabled_bundles=['rag']); skills=[RAG_INDEX_ADMIN] on AgentContract.

## What you get

Standard admin surface without ad-hoc tool lists on indexer agents.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.list_collections` | List index collections |
| `rag.describe_collection` | Collection stats |
| `rag.check_index_status` | Readiness probe |
| `rag.list_documents` | Paginated document ids |

## Related skills

- `rag.document_ingest`
- `rag.collection_lifecycle`
