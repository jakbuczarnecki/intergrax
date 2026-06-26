# 2026-06-26 — LKW.1.12/LKW.1.13 live ingest unblockers

## Summary

LKW.1 live execution moved from event/schema and tool-invocation blockers to a successful live RAG ingest.

Current LKW.1 status after this journal entry:

```text
LKW.1.11 — runtime tool registry parity: PASSED
LKW.1.12 — decision_emitted phase mismatch: PASSED
LKW.1.13 — local_indexer RAG ingest live path: PASSED
LKW.1.14 — next: final live product smoke index -> search -> synthesize
```

## LKW.1.12 — decision_emitted phase mismatch

Status:

```text
PASSED
```

Commit reported by operator:

```text
47e2ce15
```

Root cause:

```text
NexusPlanningRunner emitted RuntimeEventType.DECISION_EMITTED with phase=PLANNING,
while the event catalog requires DECISION_EMITTED to use phase=STEP_EXECUTION.
```

Fix:

```text
Planning no longer emits DECISION_EMITTED with a planning phase.
The planning DecisionRecord is carried in the PLAN_CREATED payload instead.
The canonical UAEP step-level DECISION_EMITTED event remains STEP_EXECUTION.
```

Focused verification:

```text
uv run pytest tests/unit/runtime/events -q -> 96 passed
uv run pytest tests/unit/applications/test_application_tool_registry_runtime_parity.py -q -> 1 passed
uv run pytest tests/unit/tools/providers/rag/test_rag_scope.py -q -> 10 passed
```

Live verification after LKW.1.12:

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
logs=no RuntimeEventSchemaError / no decision_emitted phase mismatch
index still returned ingested=0, chunks=0, total_tool_calls=0
```

Conclusion:

```text
The event/schema blocker was fixed.
The next blocker was local_indexer not reaching successful rag.ingest_document.
```

Classification:

```text
Platform-reusable
```

Reason: runtime event catalog/schema correctness affects every Tier-3 product host using the same runtime event bus and validating store.

## LKW.1.13 — local_indexer RAG ingest execution

Status:

```text
PASSED
```

Commit:

```text
4bc407533e991d93636668b7d7cae78e41a5a3c6
```

Root cause:

```text
UAEP/ACP executed local_indexer with a stub RuntimeContext that did not carry the application tool registry.
The ACP path also missed uaep_exec_ctx and allowed_tools propagation.
The live path therefore failed before rag.ingest_document could succeed.
```

Fix:

```text
- apply_host_tool_invoker_to_runtime_context in UAEP
- attach_acp_catalog_exec_ctx in ACP
- allowed_tools propagation from request.metadata in acp_run
- declarative invoker injection outside ACP session flag
- better tool error propagation in run_index_job/runtime_helpers
```

Focused verification:

```text
agents/local_indexer/tests
tests/unit/applications/test_application_tool_registry_runtime_parity.py
tests/unit/tools/providers/rag/test_rag_scope.py
tests/unit/agents/persistence/test_tool_invoker_wiring.py
-> 22 passed
```

Live verification:

```text
health={"status":"ok"}
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
logs=no unknown_capability_tool, no RuntimeEventSchemaError, no ingest_failed
qdrant=tenant collection intergrax__tenant__lkw-smoke present
collection_id=lkw-ingestfix-20260626162943
```

Classification:

```text
Platform-reusable
```

Reason: the fix bridges host catalog tool invocation into UAEP/ACP cognitive agent execution. Future Tier-3 product hosts using authored cognitive agents and catalog tools can reuse the same execution path.

## Non-blocking follow-up

```text
total_tool_calls=0 still appears in the run summary.
```

Classification:

```text
LKW-H1 / observability follow-up
```

It is not a product execution blocker while live behavior is verified through tool effects and Qdrant evidence.

## Next step

```text
LKW.1.14 — final live product smoke
```

Scope:

```text
index fixture -> search marker/evidence -> synthesize shadow artifact -> verify source immutability
```

LKW.1 should not be closed until LKW.1.14 proves the full product path.
