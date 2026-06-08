# `rag.retrieval_tuner`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

Retrieval tuning with preview and rerank before production queries.

## How it works

rag.preview_retrieval + rag.rerank + rag.retrieve.

## How to use

rag_skill_profile(); indexer and QA tuning agents.

## What you get

Tuning loop without exposing all RAG admin tools.

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `rag.preview_retrieval` | Preview retrieval candidates |
| `rag.rerank` | Rerank result set |
| `rag.retrieve` | Execute retrieval |

## Related skills

- `rag.hybrid_qa`
- `rag.index_admin`
