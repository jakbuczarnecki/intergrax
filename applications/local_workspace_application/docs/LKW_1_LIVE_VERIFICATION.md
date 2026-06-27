# LKW.1 live verification status — 2026-06-27

## Current status

```text
LKW.1.11 — runtime tool registry parity: PASSED
LKW.1.12 — decision_emitted event phase mismatch: PASSED
LKW.1.13 — local_indexer RAG ingest live path: PASSED
LKW.1.14 — final live product smoke: PARTIAL (search retrieve_failed)
LKW.1.15 — tenant-scoped rag.retrieve + local_search allowlist: PASSED
LKW-H1 — trace/evidence and observability follow-ups after product smoke
```

## LKW.1.15 — tenant-scoped retrieve for live search

Status:

```text
PASSED
```

Root cause:

```text
1. RAG: wired retriever_manager targeted the default vectorstore while tenant-scoped
   resolve_tenant_scoped_vectorstore selected the lkw-smoke collection — filter mismatch
   surfaced as retriever_failed.
2. LKW: local_search contract had empty extra_tools/allowed_tools, so rag.retrieve was
   denied at the UAEP tool gateway (local_indexer already declared rag.ingest_document).
```

Fix chosen:

```text
- use_wired_retrieval_managers(): skip wired retriever when store tenant differs
- perform_rag_retrieve(): build retriever on scoped vectorstore when wired managers mismatch
- local_search contract: extra_tools=[rag_retrieve_contract()]
- search_job: preserve raw_tool_reason on retrieve_failed
```

Changed files:

```text
intergrax/tools/providers/rag/scope.py
intergrax/tools/providers/rag/service.py
agents/local_search/contract.py
agents/local_search/steps/search_job.py
agents/local_search/tests/test_contract.py
agents/local_search/tests/test_search_job.py
tests/unit/tools/providers/rag/test_rag_scope.py
```

Focused tests:

```text
tests/unit/tools/providers/rag/test_rag_scope.py -> 13 passed
tests/unit/integrations/providers/vector_store -> 29 passed
agents/local_search/tests -> 7 passed
```

Live smoke (fixture `lkw-final-smoke-115.txt`, marker `LKW_FINAL_SMOKE_20260627C`, collection `lkw-final-20260627103000`, tenant `lkw-smoke`):

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
search=results=1, query=LKW_FINAL_SMOKE_20260627C
synthesize=shadow artifact lkw-chain-summary.md when evidence supplied in metadata
source immutability=original fixture unchanged
logs=no RuntimeEventSchemaError, unknown_capability_tool, tool_gateway_not_available, ingest_failed, retriever_failed
qdrant=local_workspace__tenant__lkw-smoke, workspace_id=lkw-final-20260627103000, tenant_id=lkw-smoke
```

Follow-up (non-blocking for LKW.1.15):

```text
Standalone synthesize HTTP call with message-only (no evidence/draft) returns content_missing —
orchestration must pass search evidence into synthesize metadata until LKW pipeline capability exists.
total_tool_calls=0 remains an observability accounting gap (LKW-H1).
```

## Closeout rule

LKW.1 product path verified live:

```text
index -> search (tenant-scoped evidence) -> synthesize (with evidence) -> shadow artifact only
```

Next queue item:

```text
LKW-H1 — live trace/evidence inspection and observability follow-ups
```
