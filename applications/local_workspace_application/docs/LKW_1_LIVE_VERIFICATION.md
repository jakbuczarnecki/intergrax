# LKW.1 live verification status — 2026-06-26

## Current status

```text
LKW.1.11 — runtime tool registry parity: PASSED
LKW.1.12 — decision_emitted event phase mismatch: PASSED
LKW.1.13 — local_indexer RAG ingest live path: PASSED
LKW.1.14 — next: final live product smoke index -> search -> synthesize
LKW-H1 — trace/evidence and observability follow-ups after product smoke
```

## Current product proof position

The live `local.workspace.index` path now reaches RAG ingest through the live Docker HTTP stack.

Latest confirmed live index result:

```text
accepted=1
rejected=0
ingested=1
chunks=1
```

Qdrant confirmation:

```text
tenant collection present: intergrax__tenant__lkw-smoke
ingest collection_id: lkw-ingestfix-20260626162943
```

The remaining product closeout is not another index-only smoke. The next proof must verify the whole product path:

```text
index -> search -> synthesize
```

## LKW.1.11 — runtime registry parity

Status:

```text
PASSED in implementation/unit scope; later live blockers were separate.
```

Implementation commit reported by operator:

```text
47b8667e48fb834829bcb321b37367789e62e896
```

Original issue:

```text
ApplicationToolWiring.registry was built in Tier-3,
but the runtime gateway/invoker path used a different registry.
```

Focused tests:

```text
uv run pytest tests/unit/applications/test_application_tool_registry_runtime_parity.py -q
-> 1 passed

uv run pytest tests/unit/tools/providers/rag/test_rag_scope.py -q
-> 10 passed
```

After LKW.1.11, live HTTP still did not ingest. The next blocker was the runtime event schema issue fixed in LKW.1.12.

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
while the runtime event catalog requires DECISION_EMITTED to use phase=STEP_EXECUTION.
The validating runtime event store rejected the event during persistence.
```

Fix chosen:

```text
Removed DECISION_EMITTED emission from the planning phase.
Moved the planning DecisionRecord into PLAN_CREATED payload as decision_record.
Kept DECISION_EMITTED as the canonical step-level UAEP decision event.
```

Focused tests:

```text
uv run pytest tests/unit/runtime/events -q
-> 96 passed

uv run pytest tests/unit/applications/test_application_tool_registry_runtime_parity.py -q
-> 1 passed

uv run pytest tests/unit/tools/providers/rag/test_rag_scope.py -q
-> 10 passed
```

Live result after LKW.1.12:

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
index=completed, accepted=1, ingested=0, chunks=0, total_tool_calls=0
logs=no RuntimeEventSchemaError / no decision_emitted phase mismatch
qdrant=no lkw-phasefix-* collection because ingested=0
```

Interpretation:

```text
The event phase blocker was fixed.
The next blocker was local_indexer not reaching successful rag.ingest_document.
```

## LKW.1.13 — local_indexer RAG ingest execution

Status:

```text
PASSED
```

Commit reported by operator:

```text
4bc407533e991d93636668b7d7cae78e41a5a3c6
```

Root cause:

```text
UAEP/ACP ran local_indexer with a stub RuntimeContext that did not carry the application tool registry.
The ACP path also missed uaep_exec_ctx and proper allowed_tools propagation.
As a result, the live indexer path produced unknown_capability_tool:rag.ingest_document before the fix.
```

Fix chosen:

```text
- apply_host_tool_invoker_to_runtime_context in UAEP
- attach_acp_catalog_exec_ctx in ACP
- propagate allowed_tools from request.metadata in acp_run
- inject declarative invoker outside ACP session flag
- improve tool error propagation in run_index_job/runtime_helpers
```

Changed files reported by operator:

```text
intergrax/agents/authoring/acp_uaep_shim.py
intergrax/agents/authoring/acp_run.py
intergrax/agents/uaep.py
intergrax/agents/persistence/tool_invoker_wiring.py
agents/lkw_shared/runtime_helpers.py
agents/local_indexer/steps/index_job.py
agents/local_indexer/tests/test_index_job.py
tests/unit/agents/persistence/test_tool_invoker_wiring.py
```

Focused tests:

```text
agents/local_indexer/tests
tests/unit/applications/test_application_tool_registry_runtime_parity.py
tests/unit/tools/providers/rag/test_rag_scope.py
tests/unit/agents/persistence/test_tool_invoker_wiring.py
-> 22 passed
```

Live result after LKW.1.13:

```text
health={"status":"ok"}
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
logs=no unknown_capability_tool, no RuntimeEventSchemaError, no ingest_failed
qdrant=tenant collection intergrax__tenant__lkw-smoke present
```

Platform propagation:

```text
Platform-reusable.
The fix bridges host catalog tool invocation into UAEP/ACP cognitive agent execution.
Future Tier-3 applications using authored cognitive agents and catalog tools benefit from the same path.
```

## Known non-blocking follow-up

```text
total_tool_calls=0 remains an observability/summary accounting bug.
It is not a product blocker while live index/search/synthesize behavior is verified through actual tool effects and Qdrant evidence.
```

Recommended follow-up classification:

```text
LKW-H1 / observability follow-up, not LKW.1 execution blocker.
```

## Next task

```text
LKW.1.14 — final live product smoke
```

Scope:

```text
Run the full live Docker HTTP path:
index fixture -> search marker/evidence -> synthesize shadow artifact -> verify original source immutability.
```

Acceptance:

```text
- health endpoint returns ok
- agents endpoint lists local_indexer/local_search/local_synthesizer
- index returns ingested=1 and chunks>0
- search retrieves marker or fixture sentence with evidence
- synthesize completes and writes only under shadow workspace
- original source file remains unchanged
- no RuntimeEventSchemaError
- no unknown_capability_tool
- no tool_gateway_not_available
```

## Closeout rule

Do not close LKW.1 until LKW.1.14 verifies the full product path:

```text
index -> search -> synthesize -> shadow artifact only
```
