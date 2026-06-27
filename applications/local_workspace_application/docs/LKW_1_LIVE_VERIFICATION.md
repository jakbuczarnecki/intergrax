# LKW.1 live verification status — 2026-06-27

## Current status

```text
LKW.1 — CLOSED IN SCOPE / PRODUCT PROOF PASSED
LKW.1.11 — runtime tool registry parity: PASSED
LKW.1.12 — decision_emitted event phase mismatch: PASSED
LKW.1.13 — local_indexer RAG ingest live path: PASSED
LKW.1.14 — final live product smoke attempt: PARTIAL (tenant-scoped search retrieve_failed)
LKW.1.15 — tenant-scoped rag.retrieve + local_search allowlist + final product closeout: PASSED
LKW-H1.1 — live index tool-call accounting: PASSED
LKW-H1.2 — trace/evidence contract and inspection surface: PASSED WITH PLATFORM FOLLOW-UPS
LKW-H1.3 — PASSED WITH PLATFORM FOLLOW-UPS
```

## Verified LKW.1 product path

LKW.1 product path verified live:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Latest passing product smoke:

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

## LKW-H1.1 — live index tool-call accounting

Status:

```text
PASSED
```

Commits reported by operator:

```text
62621bc1 — fix(runtime): account for catalog tool calls in LKW runs
a22222e0 — fix(runtime): propagate live catalog tool calls into app summary
```

Root cause:

```text
1. Catalog tool calls through RuntimeExecutionContext.invoke_tool() executed correctly,
   but the first H1.1 fix only proved the ACP/kernel harvest path.
2. Live LKW HTTP runs use the UAEP execute_uaep_step_via_kernel path.
3. build_uaep_step_context() did not include uaep_exec_ctx in AgentStepContext.metadata,
   so HarnessKernel could not drain pending ToolCallRecords from the RuntimeExecutionContext
   used by local_indexer.run_step().
4. Product behavior still worked (ingested=1), but application_run_summary.v1 reported
   total_tool_calls=0 because the live UAEP bridge path did not expose those tool calls
   to the kernel trace/app summary.
```

Fix chosen:

```text
- RuntimeExecutionContext.invoke_tool() records pending ToolCallRecord entries.
- HarnessKernel._build_step_record() drains pending tool calls from step_ctx.metadata["uaep_exec_ctx"].
- build_uaep_step_context() now forwards uaep_exec_ctx on the UAEP bridge path used by live Nexus runs.
- Regression coverage proves UAEP bridge tool calls reach trace_summary and application_run_summary.v1.
```

Changed files:

```text
intergrax/contracts/runtime_execution_context.py
intergrax/runtime/kernel/step_kernel.py
intergrax/agents/authoring/uaep_step_bridge.py
tests/unit/agents/persistence/test_tool_invoker_wiring.py
tests/unit/runtime/kernel/test_step_kernel.py
tests/unit/agents/authoring/test_uaep_step_bridge.py
```

Focused tests:

```text
uv run pytest tests/unit/runtime/kernel/test_step_kernel.py::test_kernel_harvests_uaep_catalog_tool_calls -q
uv run pytest tests/unit/agents/persistence/test_tool_invoker_wiring.py -q
uv run pytest tests/unit/agents/authoring/test_uaep_step_bridge.py::test_uaep_kernel_bridge_harvests_catalog_tool_calls_for_app_summary -q
uv run pytest applications/local_workspace_application/tests -q

Result: 11 passed, 4 warnings
```

Live index smoke after H1.1:

```text
local.workspace.index
accepted=1
rejected=0
ingested=1
chunks=1
application_run_summary.v1.agent_invocations[0].total_tool_calls=1
```

Interpretation:

```text
The known total_tool_calls=0 gap is fixed for the live LKW index path
(rag.ingest_document through UAEP/Nexus). Search and synthesize accounting
verification remains a follow-up under H1 because those live paths should also
show rag.retrieve and workspace.write_file respectively.
```

Known non-blocking warning:

```text
The focused application tests emitted async runtime plugin warnings about
coroutines not awaited in event_bus/task_trace plugin handlers. Those warnings
are not part of H1.1 tool-call accounting and should be tracked separately as
runtime-events/observability cleanup if they remain reproducible.
```

## LKW-H1.2 — trace/evidence contract and inspection surface

Status:

```text
PASSED WITH PLATFORM FOLLOW-UPS
```

Delivered:

```text
POST /v1/local_workspace/run attaches metadata["lkw_evidence.v1"] alongside application_run_summary.v1.
Curated read model from AgentRunTrace step diagnostics — no full_trace on HTTP response.
Typed diagnostics: lkw.index_summary.v1, lkw.search_summary.v1, lkw.synthesize_summary.v1.
Unsafe fields redacted (query_text, content, raw_chunks, documents, …).
Index smoke verifies total_tool_calls>0; synthesize smoke verifies shadow artifact path/ref.
```

Changed files:

```text
applications/local_workspace_application/serving/evidence_slice.py
applications/local_workspace_application/serving/run_metadata.py
applications/local_workspace_application/serving/fastapi_router.py
agents/local_indexer/diagnostics.py
agents/local_search/diagnostics.py
agents/local_synthesizer/diagnostics.py
applications/local_workspace_application/tests/test_evidence_slice.py
applications/local_workspace_application/tests/test_lkw_evidence_metadata.py
applications/local_workspace_application/tests/test_lkw_evidence_live_smoke.py
```

Focused tests:

```text
uv run pytest applications/local_workspace_application/tests/test_evidence_slice.py -q
uv run pytest applications/local_workspace_application/tests/test_lkw_evidence_metadata.py -q
uv run pytest applications/local_workspace_application/tests/test_lkw_evidence_live_smoke.py -q

Result: 9 passed, 12 warnings
```

Platform follow-ups deferred:

```text
RAG ingest-specific observability contract
search/synthesize per-tool accounting (rag.retrieve, workspace.write_file) in trace/summary -> LKW-H1.3
policy decisions and raw tool reason/error at RuntimeEvent layer
async runtime plugin coroutine warnings in event_bus/task_trace handlers
ACP shadow_workspace_id propagation into execution structured_data (PF2 follow-up)
```

**LKW-PF1:** immediate `TOOL_*` RuntimeEvents — PASSED WITH FOLLOW-UP (see §LKW-PF1).

**LKW-PF2:** RunArtifactBundle / WorkspaceArtifactRef — PASSED WITH FOLLOW-UP (see §LKW-PF2).

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
| LKW.1.15 | Tenant-scoped retrieve and local_search allowlist fixed; product proof closed. |
| LKW-H1.1 | UAEP live bridge tool-call accounting fixed for index/app summary. |

## LKW-PF1 — immediate tool RuntimeEvents

Status:

```text
PASSED WITH FOLLOW-UP
```

Scope:

```text
RuntimeExecutionContext.invoke_tool emits TOOL_REQUESTED before gateway invocation and
TOOL_COMPLETED / TOOL_DENIED / TOOL_FAILED after, using generic platform payload only
(tool_id, status, latency_ms, args_digest, error_code, agent_id, task_id, run_id, phase).
No raw tool args or LKW-specific schemas in core runtime.
```

Focused tests:

```text
uv run pytest tests/unit/contracts/test_invoke_tool_runtime_events.py tests/unit/agents/authoring/test_runtime_rag_call_recording.py -q

Result: 10 passed
```

Follow-up (visibility gap):

```text
LKW live HTTP smoke has no public runtime-event log surface; TOOL_* events are persisted
via runtime event bus but not yet assertable from POST /v1/local_workspace/run responses.
```

Queue:

```text
NEXT: LKW.2
```

## LKW-PF2 — RunArtifactBundle / WorkspaceArtifactRef for synthesize

Status:

```text
PASSED WITH FOLLOW-UP
```

Decision:

```text
Reuse existing platform contracts (no LKW-specific artifact layer):
- intergrax/contracts/task_artifacts.py — RunArtifactBundle, WorkspaceArtifactRef
- run_artifact_bundle.v1 metadata key (nexus task finisher rollup)
LKW application promotes bundle on HTTP metadata and correlates synthesize diagnostics
to workspace refs by artifact_path / artifact_ref.
```

Wiring:

```text
applications/local_workspace_application/serving/run_artifact_metadata.py
applications/local_workspace_application/serving/fastapi_router.py
```

Metadata exposed (safe):

```text
run_artifact_bundle.v1.workspace[].artifact_id
run_artifact_bundle.v1.workspace[].workspace_id
run_artifact_bundle.v1.workspace[].relative_path
run_artifact_bundle.v1.workspace[].uri
run_artifact_bundle.v1.workspace[].size_bytes / sha256
lkw.synthesize_summary.v1 (unchanged domain diagnostic)
```

Not exposed:

```text
raw synthesized content, raw evidence text, raw prompts, full file bodies
```

Focused tests:

```text
uv run pytest applications/local_workspace_application/tests/test_run_artifact_metadata.py applications/local_workspace_application/tests/test_lkw_evidence_metadata.py -q
```

Follow-up:

```text
ACP session path should propagate shadow_workspace_id into AgentExecutionResult.structured_data
so bundle workspace resolution does not rely solely on task-id fallback open_or_create.
```

Queue:

```text
NEXT: LKW.2
```

## Known follow-ups after LKW.1 / H1.1 / H1.2

```text
Search/synthesize per-tool accounting in trace/summary -> LKW-H1.3 (closed)
RAG ingest observability contract -> platform deferred
ACP shadow_workspace_id on execution structured_data -> PF2 follow-up
Standalone synthesize with message-only input can return content_missing -> LKW.2 / pipeline-orchestration input contract
```

Classification:

```text
index total_tool_calls=0 -> fixed in LKW-H1.1 for live UAEP path
curated lkw_evidence.v1 inspection surface -> fixed in LKW-H1.2 for index/search/synthesize diagnostics
search/synthesize per-tool visibility in trace/summary -> LKW-H1.3
message-only synthesize content_missing -> LKW.2 / pipeline-orchestration input contract
```

## Closeout rule

LKW.1 is closed in scope for the verified live product path:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Next queue item:

```text
LKW-H1.3 — smoke/assertion hardening for inspectable LKW run output
```
