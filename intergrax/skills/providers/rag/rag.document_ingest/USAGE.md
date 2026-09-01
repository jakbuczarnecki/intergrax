# `rag.document_ingest`

**Bundle:** `rag` · **Version:** 1.0.0 · **Risk:** `medium`

## Purpose

End-to-end **document ingestion into the vector index**: parse source files, chunk + embed + index, and verify collection status. Use for indexer agents (LKW `local_indexer`), legal corpus onboarding, or research corpora - instead of declaring parse/ingest tools manually.

## How it works

1. Skill resolves to `document.parse`, `rag.ingest_document`, `rag.describe_collection`.
2. `document.parse` uses the configured `document_parser` integration from `ToolWiringContext`.
3. `rag.ingest_document` runs the harness ingest pipeline (chunk → embed → index).
4. `rag.describe_collection` confirms document count and collection health after ingest.
5. Prompt ref `rag.document_ingest.system` guides agent behaviour; bind in UAEP steps or Prompt Registry.

## How to use

### Tier-3 profile

```python
from intergrax.applications._shared.skill_wiring import lkw_skill_profile

env.skill_profile = lkw_skill_profile()  # includes rag bundle
```

Ensure `tool_profile` enables `document` bundle tools (auto-added by `extend_tool_profile_for_skills`) and RAG managers are wired in `build_application_tool_wiring`.

### Agent contract

```python
from intergrax.skills.providers.rag.manifests import RAG_DOCUMENT_INGEST

AgentContract(id="local_indexer", skills=[RAG_DOCUMENT_INGEST], ...)
```

### Pair with Q&A

```python
skills=[RAG_DOCUMENT_INGEST, RAG_HYBRID_QA]  # ingest then query
```

## What you get

| Benefit | Detail |
|---------|--------|
| **Standard ingest surface** | One manifest for all indexer-style agents |
| **Policy-governed parse** | No direct vendor SDK in Tier-2 code |
| **Post-ingest verification** | `describe_collection` built into the pack |
| **Tier-3 integration swap** | Change parser backend via `IntegrationProfile`, not agent code |

## Tools unlocked

| `tool_id` | Role |
|-----------|------|
| `document.parse` | Parse PDF/DOCX/etc. via catalog document parser |
| `rag.ingest_document` | Chunk, embed, and index into vector store |
| `rag.describe_collection` | Collection stats after ingest |

## Integrations required

`document_parser` slug on integration profile; `vector_store` + `embedding_provider` for ingest pipeline.

## Related skills

- `rag.hybrid_qa` - query after ingest
- `workspace.authoring` - draft artifacts alongside indexed sources
