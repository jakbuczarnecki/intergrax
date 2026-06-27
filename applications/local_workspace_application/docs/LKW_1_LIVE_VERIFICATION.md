# LKW.1 live verification status — 2026-06-27

## Current status

```text
LKW.1 — CLOSED IN SCOPE / PRODUCT PROOF PASSED
LKW.1.11 — runtime tool registry parity: PASSED
LKW.1.12 — decision_emitted event phase mismatch: PASSED
LKW.1.13 — local_indexer RAG ingest live path: PASSED
LKW.1.14 — final live product smoke attempt: PARTIAL (tenant-scoped search retrieve_failed)
LKW.1.15 — tenant-scoped rag.retrieve + local_search allowlist + final product closeout: PASSED
LKW-H1 — NEXT: trace/evidence inspection and observability/tool-call accounting
```

## Verified LKW.1 product path

LKW.1 product path verified live:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Latest passing smoke:

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
search=results=1, marker evidence returned for tenant/workspace
synthesize=shadow artifact written when evidence supplied
source immutability=original fixture unchanged
logs=no RuntimeEventSchemaError, unknown_capability_tool, tool_gateway_not_available, ingest_failed, retriever_failed
qdrant=local_workspace__tenant__lkw-smoke, tenant_id=lkw-smoke, workspace_id=lkw-final-20260627103000
```

## LKW.1.15 — tenant-scoped retrieve for live search

Status:

```text
PASSED
```

Commits reported by operator:

```text
58740470 — fix(rag): restore tenant-scoped retrieve for LKW search
1af2fd26 — docs(lkw): record final live product smoke
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

Tenant-scoped retrieve verification:

```text
ingest tenant/workspace: lkw-smoke / lkw-final-20260627103000 -> ingested=1, chunks=1
retrieve same tenant/workspace: used=true, results=1, marker LKW_FINAL_SMOKE_20260627C
retrieve wrong tenant: regression test preserves isolation
retrieve wrong workspace: regression test preserves isolation
```

## LKW.1.14 — partial smoke that exposed the retrieve blocker

Status:

```text
PARTIAL / superseded by LKW.1.15
```

Result:

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
search=local_search: search failed — retrieve_failed
synthesize=shadow_workspace_required / no shadow write because evidence was missing
source immutability=OK
logs=no RuntimeEventSchemaError, unknown_capability_tool, tool_gateway_not_available, ingest_failed
qdrant=point with marker existed under tenant lkw-smoke and workspace_id lkw-final-20260627072645
```

Interpretation:

```text
Index was not the blocker. Tenant-scoped retrieve and local_search tool allowlist were the blockers.
Those blockers were fixed in LKW.1.15.
```

## Earlier LKW.1 live blockers

| ID | Result |
|----|--------|
| LKW.1.9 | Qdrant point-id compatibility fixed. |
| LKW.1.10 | Tenant scope consistency fixed. |
| LKW.1.11 | Runtime tool registry parity fixed. |
| LKW.1.12 | `decision_emitted` phase mismatch fixed. |
| LKW.1.13 | UAEP/ACP catalog invocation bridge fixed; live index ingests into Qdrant. |

## Known follow-ups after LKW.1

```text
total_tool_calls=0 remains an observability/accounting gap.
Standalone synthesize with message-only input can return content_missing.
```

Classification:

```text
total_tool_calls=0 -> LKW-H1 / observability and tool-call accounting
message-only synthesize content_missing -> LKW.2 / pipeline-orchestration input contract
```

## Closeout rule

LKW.1 is closed in scope for the verified live product path:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Next queue item:

```text
LKW-H1 — live trace/evidence inspection and observability/tool-call accounting
```
