# Trusted Ask Workspace discovery

**Task:** MVP-1  
**Status:** complete  
**Base commit:** `f290a6113703c65b824f4743ebdea5ee604eb51a`  
**Governing plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · [`PRODUCT_FIRST_MVP.md`](../../../docs/plan/PRODUCT_FIRST_MVP.md)  
**Architecture:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

---

## 1. Decision summary

```text
Reuse managed-workspace search evidence (local.workspace.search → search_summary.evidence).
Do not use local.workspace.synthesize as the Ask answer engine.
LKW owns Ask orchestration via AskAnswerAssembler, citation projection,
product run persistence and completed-run read.
```

Frozen grounded-answer mechanism:

```text
verified WorkspaceSearchHitV1 evidence
→ deterministic context projection
→ one bounded model invocation owned by LKW (AskAnswerAssembler)
→ typed grounded answer
→ citation projection from the same verified evidence
```

Shortest safe MVP-2 path:

```text
POST .../workspaces/{workspace_id}/ask
→ tenant/workspace authorization (same as search)
→ Task(local.workspace.search) via LocalWorkspaceTaskExecutor
→ verified WorkspaceSearchHitV1 evidence
→ sufficiency gate (empty → insufficient_evidence; model skipped; no invented answer)
→ LKW AskAnswerAssembler (indexed context → one bounded model call → typed answer)
→ surface-neutral citations projected from used_evidence_ids → verified hits
→ persist Ask run (question, evidence, answer, citations, status)
→ return typed result including run_id
→ GET .../asks/{run_id} (tenant-scoped) survives restart
```

Discovery does not implement this path.

---

## 2. Frozen user-visible workflow

```text
HTTP question
→ workspace-scoped retrieval
→ structured evidence
→ grounded answer or explicit insufficient-evidence
→ stable source citations
→ persisted question/evidence/answer run
→ completed-run read after restart
```

No Slack fields. HTTP is the canonical surface for MVP-2. MCP remains untouched.

---

## 3. Existing code path

### 3.1 HTTP and application entry

| Item | Existing |
|------|----------|
| Managed search route | `POST /v1/local_workspace/workspaces/{workspace_id}/search` — `search_workspace` in `serving/workspace_routes.py` |
| Request model | `WorkspaceSearchRequestV1` (`query`, `limit`) — `serving/workspace_schemas.py` |
| Response model | `WorkspaceSearchResponseV1` → `list[WorkspaceSearchHitV1]` |
| Tenant resolution | `resolve_tenant_id` — request context → `X-Tenant-Id` → body → `"default"` |
| Workspace auth | `ManagedWorkspaceService.get_workspace(tenant_id, workspace_id)`; missing → HTTP 404 |
| Generic run route | `POST /v1/local_workspace/run` — `LocalWorkspaceRunService.run_task` in `serving/fastapi_router.py` |
| Execution boundary | `LocalWorkspaceTaskExecutor.execute` → `UnifiedTaskRunner` → Nexus (`host/task_executor.py`) |
| Run ID creation | `new_run_id()` from `intergrax.runtime.task.task_run_bridge`; set as `Task.task_id` |

Managed search builds:

```text
Task(
  capability="local.workspace.search",
  message=body.query,
  metadata={tenant_id, workspace_id, collection_id=workspace_id, query, top_k, requested_by}
)
```

then `await task_executor.execute(task)` and maps `TaskResult.execution_result.structured_data["search_summary"]`.

**Sync vs async:** HTTP awaits execution (synchronous request/response). Sync enqueue is used for folder sync operations, not for search.

There is **no** Ask Workspace route today.

### 3.2 Workspace search

Trace `local.workspace.search`:

| Step | Location |
|------|----------|
| Capability | `agents/local_search/capabilities.py` |
| Agent | `LocalSearchAgent.act` → `run_search_job` (`agents/local_search/local_search_agent.py`) |
| Skill | `local.workspace.search` — `intergrax/skills/providers/local/manifests.py` |
| Tool | `rag.retrieve` (`RAG_RETRIEVE_TOOL_ID`) |
| Output | `search_summary` dict exported to `AgentRunResult.structured_data` in `LocalSearchAgent.on_run_end` |
| Vector/document boundary | Catalog tool `rag.retrieve`; LKW verifies hits against `ManagedWorkspaceRepository.get_document_ref` |

Reasons (`SearchSummaryReason` in `agents/local_search/diagnostics.py`):

- `query_missing`
- `tool_gateway_not_available`
- `retrieve_failed`
- `retrieve_complete`

Product HTTP mapping (`_map_search_hits` in `workspace_routes.py`) drops incomplete evidence, drops wrong-workspace items, verifies document refs, and raises `SearchEvidenceIncompleteError` → HTTP 502 `search_evidence_incomplete` when evidence cannot be verified. Empty verified hits return `results: []` with HTTP 200.

### 3.3 Typed evidence

**Canonical product-facing evidence for managed workspaces** is the verified hit:

```text
WorkspaceSearchHitV1
  document_id, source_id, workspace_id, source_path, file_name, score, snippet, metadata
```

(`serving/workspace_schemas.py`)

**Upstream agent evidence** is a structured dict list inside `search_summary.evidence`, built by `_format_evidence_item` / `_format_evidence` in `agents/local_search/steps/search_job.py`:

| Field | Present |
|-------|---------|
| `text` / `snippet` | yes |
| `source_path` | yes |
| `chunk_id` | yes (when available) |
| `score` | yes (when available) |
| `document_id` | yes (from chunk metadata) |
| `source_id` | yes |
| `workspace_id` | yes |
| `file_name` | yes |
| `metadata` | yes (provider/chunk meta) |
| page/line/offset | only if present inside `metadata` (not first-class) |
| tenant_id | on Task/metadata/diagnostics, not per evidence item |

Stability: **yes for synthesis/handoff consumption as dict evidence** — already consumed by synthesizer and pipeline handoff tests. Not a dedicated Pydantic domain class at agent layer; product HTTP promotes verified fields to `WorkspaceSearchHitV1`.

Verdict:

```text
typed domain evidence (dict + WorkspaceSearchHitV1 projection)
not merely opaque provider payloads at the product boundary
```

Provider-specific keys may still appear inside `metadata`; MVP-2 must not leak them into the Ask result.

### 3.4 Synthesis

Closest capability: `local.workspace.synthesize` → `LocalSynthesizerAgent` → `run_synthesize_job` (`agents/local_synthesizer/steps/synthesize_job.py`).

| Concern | Actual behavior |
|---------|-----------------|
| Expected evidence | `metadata.evidence` or `metadata.search_summary.evidence`, or prior-output handoff from `local_search` |
| Model-provider boundary | Job does **not** call an LLM; it builds markdown from evidence/message and writes a file |
| Output | `synthesize_summary` (`used`, `reason`, artifact path/ref, `num_evidence_items`) — not a Q&A answer with citations |
| Shadow requirement | Fails with `shadow_workspace_required` unless `shadow_workspace=True` |
| Empty evidence | If no evidence but `message` present, `_resolve_content` writes a draft from the message alone — **ungrounded content path** |
| Compatibility with search | Dict evidence from `search_summary` **can** be consumed (see acceptance/handoff tests) |

**Mismatch for Ask Workspace:** search produces verified retrieval evidence; synthesizer produces shadow **draft artifacts**, not a grounded answer with stable citations. Architecture prose in `ARCHITECTURE.md` §10.3 mentioning LLM synthesis is ahead of the implemented job.

**MVP-2 must not treat `local.workspace.synthesize` as the Ask answer engine.**

### 3.5 Citations

| Layer | Type | Role for Ask |
|-------|------|--------------|
| RAG engine | `intergrax.rag.retrieval.citation.Citation` | Retrieval provenance (`chunk_id`, `source_id`, `source_label`, `page`, `score`, `excerpt`, `metadata`) |
| Nexus response | `intergrax.runtime.nexus.responses.response_schema.Citation` | Generic runtime citation (`source_id`, `source_type`, `source_label`, …) |
| Search job | dict citations from `rag.retrieve`, folded into evidence | Intermediate |
| Product HTTP search | no citation type — hits are `WorkspaceSearchHitV1` | Closest product source reference |
| Ask Workspace | **absent** | Must be projected in MVP-2 |

No Slack-specific citation fields exist. MVP-2 should define a surface-neutral Ask citation schema projected from verified evidence (`document_id`, `source_id`, `workspace_id`, `source_path`/`file_name`, optional `chunk_id`/`score`, `snippet`/`excerpt`). Prefer extending product schemas over inventing a platform-wide citation rewrite.

### 3.6 Run lifecycle and persistence

| Kind | Existing | Suitable for Ask? |
|------|----------|-------------------|
| Trace persistence | Nexus/runtime diagnostics, evidence metadata on `TaskResult.metadata` | No — observability, not product Ask history |
| Task lifecycle | In-process `Task`/`TaskResult` via `UnifiedTaskRunner`; `run_id` == task id for sync execute | Ephemeral for LKW HTTP `/run` and managed search |
| Operation persistence | `WorkspaceOperation` via `ManagedWorkspaceRepository` + `GET /operations/{operation_id}` | Sync lifecycle only |
| Platform `RunService` | `intergrax/fastapi_core/runs/` | **Not mounted** by LKW sync execute path |
| Product Ask run persistence | **absent** | Required for MVP-2 |

Managed search creates a `run_id` but returns only `WorkspaceSearchResponseV1` (no persisted Ask run, no read-back).

### 3.7 Completed-run read

| Path | Exists? |
|------|---------|
| `GET` Ask / completed product run for question/answer/citations | **No** |
| `GET /v1/local_workspace/operations/{operation_id}` | Yes — sync operations only |
| `POST /v1/local_workspace/run` | Returns result once; no GET by `run_id` |

**Concrete gap:** no suitable completed Ask-run read path.

### 3.8 Insufficient-evidence behavior

| Case | Current behavior | Boundary |
|------|------------------|----------|
| No verified hits | HTTP 200, `results: []` | Managed search |
| Incomplete evidence fields | 502 `search_evidence_incomplete` | `_map_search_hits` |
| Unverified / provenance mismatch | dropped; may 502 if only unverified remain | `_map_search_hits` |
| Wrong workspace in evidence | dropped | `_map_search_hits` |
| Provider/`rag.retrieve` failure | `search_summary.used=False`, reason `retrieve_failed` | `run_search_job` |
| Empty query | agent `query_missing`; HTTP request forbids empty via `min_length=1` | schemas + agent |
| Synthesize with no evidence | may draft from raw message | **does not enforce** “no invented grounded answer” |

Intended product rule is **not** enforced end-to-end for Ask. MVP-2 must add an Ask-layer sufficiency gate before any grounded answer is produced.

---

## 4. Frozen Ask Workspace contract

### 4.1 Request

Surface-neutral HTTP body (proposed):

```text
WorkspaceAskRequestV1
  question: str (min_length=1)
  limit: int = 10 (ge=1, le=100)   # already justified by WorkspaceSearchRequestV1
```

Context from route/headers (not Slack):

```text
tenant_id ← resolve_tenant_id
workspace_id ← path
```

Forbidden: Slack channel/user/team fields, provider-specific retrieval knobs beyond existing `limit`.

### 4.2 Execution

```text
workspace-scoped question
→ authorization and scope validation (get_workspace)
→ structured retrieval (local.workspace.search via LocalWorkspaceTaskExecutor)
→ map/verify evidence (reuse _map_search_hits semantics)
→ evidence sufficiency decision (empty verified evidence → skip model)
→ LKW AskAnswerAssembler
   (question + verified WorkspaceSearchHitV1
    → deterministic indexed context
    → one bounded model invocation
    → AskAnswerAssemblyResult)
→ citation projection from used_evidence_ids → verified hits
→ typed final result
→ persist Ask run
```

Frozen component: **`AskAnswerAssembler`** in the LKW application layer.

It must not be introduced as a new Tier-2 agent, platform-wide Ask framework,
Nexus replacement, rewrite of `local.workspace.synthesize`, generic synthesis
abstraction, or Slack-specific component.

### 4.3 Evidence

Persist and return verified evidence compatible with `WorkspaceSearchHitV1` fields (or an Ask-specific alias that does not add Slack/provider leakage).

### 4.4 Answer

Answer engine for MVP-2 is **`AskAnswerAssembler`** (LKW application layer).

#### Input contract

`AskAnswerAssembler` receives only:

```text
question: str
evidence: list[WorkspaceSearchHitV1]
```

It may also receive the minimum existing model runtime dependency already available
to the LKW host.

It must not receive: raw `rag.retrieve` provider responses; unverified evidence;
Slack fields; arbitrary task metadata; filesystem access; write-artifact
instructions; prior free-form agent messages as evidence.

#### Deterministic context projection

Before invoking the model, the assembler creates a bounded context only from
verified fields:

```text
document_id
source_id
workspace_id
file_name
source_path
snippet
score
approved location metadata, when present
```

Provider-specific keys inside raw metadata must not be included unless explicitly
promoted into the frozen product schema.

The context preserves a stable evidence index:

```text
E1
E2
E3
...
```

Each evidence item used by the model must have one stable evidence index.

#### Model invocation

```text
one bounded model invocation through the existing LKW/runtime model boundary
```

Do not require a new provider or new agent. Exact prompt text is not frozen in
MVP-1; the behavioral contract is frozen. The model prompt must instruct the
model to:

1. answer only from supplied evidence;
2. not use external knowledge;
3. not invent missing facts;
4. return insufficient evidence when the evidence does not support an answer;
5. reference evidence indexes used for each material claim.

#### Typed assembler output

Logical contract for MVP-2 (not implemented in MVP-1):

```text
AskAnswerAssemblyResult
  status: completed | insufficient_evidence
  answer: str | null
  used_evidence_ids: list[str]
```

Where `used_evidence_ids` contains stable evidence indexes such as `E1`, `E3`.

#### Citation projection

```text
citations are produced only by mapping used_evidence_ids
back to the original verified WorkspaceSearchHitV1 objects
```

The model must not generate final citation objects. The model may only identify
evidence indexes. LKW code creates the final typed citations.

This guarantees that citations refer to verified evidence; model-generated paths
or source identifiers are impossible; provider metadata does not leak;
Slack-specific data does not appear.

#### Insufficient-evidence boundary

**Before model invocation** — if verified evidence is empty:

```text
status = insufficient_evidence
answer = null
citations = []
model invocation = skipped
```

**After model invocation** — if the model reports insufficient support or returns
no valid used evidence indexes:

```text
status = insufficient_evidence
answer = null
citations = []
```

**Completed answer** — a result may have `status = completed` only when:

* answer is non-empty;
* at least one valid evidence index is returned;
* every evidence index maps to verified evidence;
* citations are projected successfully.

If any returned evidence index is unknown: **assembly failure**. Do not silently
ignore invalid evidence references.

#### Explicitly forbidden fallback

```text
The raw user question must never be used as answer content or as a substitute for evidence.
```

`local.workspace.synthesize` message-only fallback must not be used by Ask Workspace.

#### Product result statuses

| Status | Meaning |
|--------|---------|
| `completed` | Grounded answer present with at least one projected citation |
| `insufficient_evidence` | No invented answer; `answer = null`; empty citations |
| `failed` | Search/assembly/persistence failure |

Answer text must be produced only from verified evidence when status is `completed`.

### 4.5 Citations

Surface-neutral list projected from verified evidence, for example:

```text
AskCitationV1
  document_id
  source_id
  workspace_id
  source_path
  file_name
  chunk_id? 
  score?
  excerpt  # from snippet
```

Same structure for HTTP now and Slack later. No Slack-only fields.

### 4.6 Persisted run

LKW DocumentStore-backed Ask run (same persistence style as managed workspaces/operations), tenant + workspace scoped, fields at minimum:

- `run_id`
- `tenant_id`
- `workspace_id`
- `question`
- `status`
- `evidence` (selected/verified)
- `answer` (or null on insufficient/failed)
- `citations`
- `error` (optional)
- `created_at` / `completed_at`

Restart: run readable after process restart from durable store.

### 4.7 Completed-run read

```text
GET /v1/local_workspace/asks/{run_id}
  Header: X-Tenant-Id (same resolve_tenant_id rules)
```

- unknown / other-tenant → 404  
- incomplete (if async ever introduced) → explicit non-final status  
MVP-2 may keep ask execution synchronous (like search) while still persisting for read-after-restart.

### 4.8 Failure states

| Case | Behavior |
|------|----------|
| Unknown workspace | 404 |
| Unauthorized tenant/workspace | 404 (no cross-tenant leak) |
| Empty question | 422 validation |
| No evidence | `insufficient_evidence`; persist; no invented answer |
| Insufficient / unverified evidence | `insufficient_evidence` or 502 if search evidence pipeline broken (align with search hardening) |
| Answer assembly failure | `failed` + error; persist |
| Persistence failure | 502; do not claim success |
| Missing run on GET | 404 |
| Incomplete run on GET | return non-final status if applicable |

---

## 5. Product/platform ownership

| Concern | Existing owner | Required owner | Reason |
|---------|----------------|----------------|--------|
| HTTP request parsing | LKW `serving/` | LKW | Product API schemas |
| Identity and tenant context | Intergrax request context + LKW `resolve_tenant_id` | Intergrax + LKW | Propagate identity; LKW resolves for product routes |
| Workspace authorization | LKW `ManagedWorkspaceService` | LKW | Workspace is product domain |
| Ask Workspace orchestration | absent | **LKW** | Product capability; surfaces must not own it |
| Search execution | `local_search` / `local.workspace.search` | agents + Intergrax tools | Reuse as-is |
| Evidence model | search_summary dict + `WorkspaceSearchHitV1` | LKW product projection + agent structured_data | Consume existing; minimal Ask schema if needed |
| Synthesis / answer for Ask | shadow `local.workspace.synthesize` (wrong shape) | **LKW Ask answer assembly** | Must produce grounded Q&A, not draft files |
| Citation creation | RAG/Nexus types unused by product Ask | **LKW projection from evidence** | Surface-neutral product citations |
| Final typed result | absent | LKW | Ask response contract |
| Run lifecycle (Ask) | absent | LKW | Product Ask run, not sync operation |
| Run persistence (Ask) | absent | LKW DocumentStore repository | Restart-safe product state |
| Completed-run read | absent | LKW HTTP | Product read API |
| HTTP rendering | LKW FastAPI | LKW | Canonical surface |
| Future Slack rendering | not in scope | Slack adapter only | Render typed Ask result; no orchestration |

Rules confirmed:

```text
LKW owns product capabilities and local execution.
Intergrax owns intake, identity/context, task execution, reusable delivery contracts.
HTTP/MCP/Slack are replaceable surfaces.
Slack must not own Ask orchestration, search, synthesis, citations or persistence.
```

---

## 6. Existing test coverage

| Boundary | Test file | Test name | Proves | Does not prove |
|----------|-----------|-----------|--------|----------------|
| Workspace search contract (API) | `tests/workspaces/test_managed_workspace_api.py` | `test_sync_search_idempotency_and_workspace_isolation` | Sync → search hits with provenance; empty other workspace | Ask endpoint; answer; citations product model |
| Typed evidence mapping | `tests/workspaces/test_managed_workspace_hardening.py` | `test_map_search_hits_maps_complete_evidence_without_file_read` | Complete evidence → `WorkspaceSearchHitV1` without filesystem snippet rebuild | Ask persistence |
| Incomplete evidence | same | `test_map_search_hits_*` incomplete/unverified cases | Incomplete evidence rejected | Ask insufficient-evidence UX |
| Workspace isolation (hits) | same | `test_map_search_hits_drops_cross_workspace_evidence` | Wrong `workspace_id` dropped | Ask orchestration isolation |
| Tenant isolation | `test_managed_workspace_api.py` | `test_tenant_isolation_workspace_404` | Foreign tenant 404 on workspace/ops | Ask run read isolation |
| Search-to-synthesis compatibility | `tests/test_lkw_acceptance_index_search_synthesize.py` | acceptance flow | Search evidence consumed by synthesize job as draft input | Grounded Q&A; citation projection; no message fallback |
| Prior-output handoff | `agents/local_search/tests/test_search_handoff.py` | `test_local_search_exports_search_summary_for_graph_handoff` | `search_summary` exported for graph | Product Ask |
| Synthesize consumes handoff | `agents/local_synthesizer/tests/test_synthesize_job.py` | `test_run_synthesize_job_consumes_prior_search_handoff` | Evidence → shadow draft | Insufficient-evidence gate for Ask |
| Citation serialization (product Ask) | — | — | — | **Missing** |
| Run persistence (Ask) | — | — | — | **Missing** |
| Completed-run read (Ask) | — | — | — | **Missing** |
| Restart persistence (vectors/ops) | LKW.5 / product sync proofs | various | Vectors/ops survive restart | Ask runs |
| Insufficient evidence (Ask) | — | — | Empty search results only | No invented grounded answer for Ask |

Live proofs are not substitutes for the focused Ask contract tests listed in §11.

---

## 7. Confirmed gaps

1. No Ask Workspace HTTP capability or typed product result.  
2. No product Ask run persistence (question/evidence/answer/citations/status).  
3. No completed Ask-run read path.  
4. `local.workspace.synthesize` is a shadow-draft writer, not grounded Ask synthesis.  
5. No product Ask citation schema (only RAG/Nexus/search-hit building blocks).  
6. Insufficient-evidence rule not enforced for a grounded Ask answer.  
7. Search→synthesize acceptance proves draft handoff, not Ask Q&A with citations.

---

## 8. Major blocker classification

```text
PRODUCT_BLOCKING
```

**Description:** There is no LKW Ask Workspace product orchestration that turns verified search evidence into a grounded answer with stable citations and a restart-durable, readable product run. Existing `local.workspace.synthesize` cannot serve that role (shadow draft + ungrounded message fallback). Platform `RunService` is not used by LKW’s sync execute path.

**Why it blocks MVP-2:** Without this product layer, the vertical slice cannot deliver “question → evidence → answer → citations → persisted run → read”.

**Smallest acceptable resolution (MVP-2):**

- LKW Ask service + HTTP ask/read routes
- Reuse managed search execution and `_map_search_hits` semantics
- LKW `AskAnswerAssembler`: verified hits → deterministic indexed context → one bounded model invocation → typed `AskAnswerAssemblyResult`
- Citation projection by LKW from `used_evidence_ids` → verified hits (model never creates citation objects)
- DocumentStore Ask-run repository (mirror managed-workspace persistence style)
- Sufficiency gate before any `completed` answer (empty evidence skips the model)
- Explicit forbid: raw user question must never be used as answer content

**Owner:** LKW application (`serving/`, `workspaces/` or sibling Ask module).

**Explicitly not generalized yet:**

- platform-wide Ask framework
- Nexus/RunService redesign as mandatory dependency
- Slack adapter work
- rewriting `local.workspace.synthesize` into a universal answer engine
- new Tier-2 agent or generic synthesis abstraction
- new vector/model providers

```text
NO second major platform gap opened during discovery.
```

---

## 9. Exact MVP-2 implementation scope

**One-sentence summary:** Implement surface-neutral HTTP Ask Workspace that reuses managed search evidence, applies an insufficient-evidence gate, produces a grounded answer via `AskAnswerAssembler` with projected citations, persists the run, and supports completed-run read after restart — validated by focused tests plus one controlled live proof.

**Allowed:**

- Ask request/response/citation schemas in LKW serving
- Ask orchestration service in LKW
- LKW `AskAnswerAssembler` (question + verified `WorkspaceSearchHitV1` → indexed context → one bounded model call → typed assembly result)
- DocumentStore-backed Ask run persistence + GET read
- Reuse `LocalWorkspaceTaskExecutor` + `local.workspace.search`
- Reuse search hit mapping / provenance checks
- Citation projection from `used_evidence_ids` → verified hits
- Focused unit/API/contract tests listed in §11
- One controlled end-to-end live proof (mandatory acceptance; not the main debugging loop)
- Update `IMPLEMENTATION_PLAN.md` / proof docs as required by that live proof

**Forbidden:**

- Slack / Teams
- Changing folder sync / ingest / queues unless a concrete Ask blocker
- Broad synthesizer redesign or artifact generation as Ask dependency
- Using `local.workspace.synthesize` or its message-only fallback as the Ask answer engine
- New Tier-2 agent / platform-wide Ask framework / generic synthesis abstraction
- New providers / observability / token optimization
- Platform RunService migration unless strictly necessary and product-pulled
- Using the raw user question as answer content or as a substitute for evidence

---

## 10. MVP-2 acceptance criteria

1. `POST /v1/local_workspace/workspaces/{workspace_id}/ask` accepts surface-neutral question.
2. Unknown/foreign workspace → 404.
3. Empty question → 422.
4. Retrieval uses workspace-scoped `local.workspace.search` with tenant context.
5. Verified evidence only; wrong-workspace/unverified evidence never grounds an answer.
6. Grounded answers are produced only by `AskAnswerAssembler`: verified hits → deterministic indexed context → one bounded model invocation → typed answer with `used_evidence_ids`.
7. Citations are projected by LKW from `used_evidence_ids` → verified hits; the model never creates final citation objects.
8. With real documents and a real question → `completed` answer + at least one projected citation.
9. With no/insufficient evidence → `insufficient_evidence` and **no** invented grounded answer; empty evidence skips the model.
10. Unknown evidence indexes cause assembly failure; no fabricated citations.
11. The raw user question is never used as answer content or as a substitute for evidence.
12. Response includes `run_id`, `workspace_id`, `status`, `question`, answer or insufficient result, citations, completion/failure info.
13. No Slack-specific fields; no provider payload leakage in the public Ask result.
14. Run persisted with question, evidence, answer, citations, status, timestamps.
15. `GET .../asks/{run_id}` returns completed run for owning tenant; foreign tenant → 404.
16. After process restart, completed run still readable.
17. HTTP/MCP search surfaces remain functional; Ask does not break them.
18. Focused tests in §11 pass in this order: contract → unit → boundary integration → application API.
19. A single controlled live proof must pass after all focused tests.

Mandatory live proof validates (no Slack; no design partner required):

```text
real workspace
→ real synchronized documents
→ POST Ask request
→ verified retrieval evidence
→ bounded grounded answer assembly
→ projected citations
→ persisted completed run
→ GET completed run after host restart
```

The live proof must not be used as the main debugging loop.

---

## 11. MVP-2 required tests

Required validation order:

```text
contract test
→ focused unit tests
→ boundary integration tests
→ application API tests
→ one controlled live proof
```

### Citation projection boundary test (mandatory design)

| Field | Value |
|-------|-------|
| File | `applications/local_workspace_application/tests/workspaces/test_ask_workspace_contract.py` (new) |
| Name | `test_ask_workspace_search_evidence_to_citations_without_provider_or_slack_leakage` |
| Setup | In-memory DocumentStore; workspace + document ref; stub `LocalWorkspaceTaskExecutor` returning typed `search_summary.evidence` including extra provider `metadata` keys and no Slack fields |
| Inputs | Ask question + workspace/tenant headers |
| Expected | Path `verified search hits → indexed model context → used_evidence_ids → typed citations projected by LKW`; answer grounded in snippet text; citations carry `document_id`, `source_path`/`file_name`, excerpt |
| Forbidden | Citations created directly from model text; any `slack_*` keys; raw provider-only keys on top-level Ask result/citations |
| Failure cases | empty evidence → `insufficient_evidence` and null answer; cross-workspace evidence → not cited |

### Assembler and grounding tests (exact names)

1. `test_ask_answer_assembler_uses_only_verified_evidence` — assembler input is verified `WorkspaceSearchHitV1`; deterministic context contains only approved fields; raw provider metadata absent; one model call; model receives indexed evidence; typed result contains answer and valid `used_evidence_ids`.
2. `test_ask_answer_assembler_skips_model_when_evidence_is_empty` — no model call; `insufficient_evidence`; answer null; citations empty.
3. `test_ask_answer_assembler_rejects_unknown_evidence_reference` — model returns unknown evidence ID; result is not `completed`; no fabricated citation; failure persisted or propagated per MVP-2 contract.
4. `test_completed_answer_requires_at_least_one_verified_citation` — `completed` cannot be returned without at least one valid projected citation.

### Minimum additional tests

1. Happy-path contract — real question → answer + citations + `run_id`.
2. Insufficient evidence — no invented answer.
3. Tenant isolation — foreign tenant cannot read Ask run / workspace.
4. Workspace isolation — evidence from other workspace never grounds answer.
5. Persisted completed-run read — GET returns stored payload.
6. Restart persistence — reload repository/store; GET still succeeds.
7. Answer assembly failure — `failed` persisted; GET shows error.
8. Provider-specific payload does not leak into Ask response.
9. No Slack-specific fields in schemas (`extra="forbid"` / explicit asserts).
10. One controlled end-to-end live proof (mandatory; after focused tests; not the main debugging loop).

Do not implement these tests in MVP-1.

---

## 12. Explicitly deferred work

- Slack Socket Mode / MVP-3–4  
- Microsoft Teams  
- Adapting `local.workspace.synthesize` into general LLM Q&A  
- Artifact generation / shadow drafts as Ask dependency  
- Full document reconciliation  
- Token optimization  
- Platform `RunService` adoption for all LKW tasks  
- Broad hallucination framework beyond Ask sufficiency gate  
- Second conversational adapter for portability proof  

---

## 13. Inspected files

Governing:

- `docs/plan/PRODUCT_FIRST_MVP.md`
- `applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md`
- `applications/local_workspace_application/docs/ARCHITECTURE.md`

LKW application:

- `applications/local_workspace_application/serving/workspace_routes.py`
- `applications/local_workspace_application/serving/workspace_schemas.py`
- `applications/local_workspace_application/serving/fastapi_router.py`
- `applications/local_workspace_application/serving/schemas.py`
- `applications/local_workspace_application/host/task_executor.py`
- `applications/local_workspace_application/workspaces/models.py`
- `applications/local_workspace_application/tests/workspaces/test_managed_workspace_api.py`
- `applications/local_workspace_application/tests/workspaces/test_managed_workspace_hardening.py`
- `applications/local_workspace_application/tests/test_lkw_acceptance_index_search_synthesize.py`

Agents / skills:

- `agents/local_search/local_search_agent.py`
- `agents/local_search/steps/search_job.py`
- `agents/local_search/diagnostics.py`
- `agents/local_search/tests/test_search_handoff.py`
- `agents/local_synthesizer/steps/synthesize_job.py`
- `agents/local_synthesizer/tests/test_synthesize_job.py`
- `intergrax/skills/providers/local/manifests.py`

Shared (directly referenced):

- `intergrax/rag/retrieval/citation.py`
- `intergrax/runtime/nexus/responses/response_schema.py` (`Citation`)
- `intergrax/fastapi_core/runs/` (presence; not mounted by LKW sync Ask path)
- `intergrax/runtime/task/task_run_bridge.py` (`new_run_id`)
