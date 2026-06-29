# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** · **LKW.1 Closed in scope** · **LKW-H1.2 Passed with platform follow-ups · LKW-H1.3 Passed with platform follow-ups** · **LKW-PF2A Closed** · **LKW.2 Closed — pipeline proof passed (LKW.2.4C + closeout smoke)**

Latest live proof snapshot: **2026-06-27 — LKW.1.15 PASSED / LKW.1 PRODUCT PROOF CLOSED IN SCOPE**. Tenant-scoped `rag.retrieve` works live for `tenant_id=lkw-smoke` with workspace filtering; `local.workspace.search` returns marker evidence; `local.workspace.synthesize` writes a shadow artifact when evidence/draft is supplied. Product closeout path verified live:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Latest observability snapshot: **2026-06-28 — LKW-PF1A CLOSED** · **LKW-DF1 CLOSED** (async runtime plugin coroutine warnings). Immediate catalog tool calls on the UAEP path emit generic `RuntimeEventType.TOOL_*` from `RuntimeExecutionContext.invoke_tool` (platform payload only; no raw args). LKW HTTP runs attach curated `lkw_evidence.v1` alongside `application_run_summary.v1` and redacted `runtime_event_summary.v1`. **Next platform phase:** **OBS-EXPORT-1** — external observability export boundary (LKW remains proof workload; vendor integrations out of scope).

Current LKW.2 execution status: §5 below. LKW.1/H1 historical live proof: [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md).  
Application-local history: [`journal/`](journal/).  
Platform proof loop: [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md).

Principle: **local backend daemon** · **thin frontends** · **Slack optional** · **shadow writes only** · **LKW proves the platform**.

---

## 0. Product boundary reminder

| | Backend (`lkw-host`) | Frontend (clients) |
|---|---------------------|-------------------|
| **Runs on** | localhost daemon | Tray / Cursor / Slack / curl |
| **Contains** | Nexus, agents, RAG, index, policy, trace | UI + HTTP calls only |
| **Must not** | — | RAG, LLM, direct file index, agent loops |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.

---

## 0a. LKW platform proof rule

LKW is not only a product proof. LKW is the first proof that Intergrax can repeatedly create, configure, run, package, deploy, observe, and evolve agent applications.

Every non-trivial LKW wave has two acceptance layers:

1. **Product acceptance** — the LKW capability works.
2. **Platform acceptance** — reusable lessons are propagated to platform code, scaffold templates, env/settings, packaging, Docker, CI/CD, or documentation when applicable.

Canonical loop: [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md).

### Required propagation checklist

Before closing any LKW wave, answer:

- [ ] Did every discovered bug, workaround, repeated pattern, missing diagnostic, scaffold gap, config mismatch, Docker/build issue, dependency issue, and CI/runbook gap receive a classification?
- [ ] Does this change belong only to LKW, or should it move to shared platform code?
- [ ] Should application scaffold generate this pattern for the next product host?
- [ ] Should agent scaffold generate this contract, test, or documentation pattern?
- [ ] Does `.env.example` match `host/settings.py` and production validation?
- [ ] Does `pyproject.toml` need a dependency split or optional dependency group?
- [ ] Does Docker still build and run with the correct files, env profile, port, and healthcheck?
- [ ] Does CI need a new application smoke test or Docker build check?
- [ ] Does the deploy/runbook still describe the real execution path?
- [ ] Does the implementation plan identify both the LKW work and the platform propagation work?

`NexusLoop` constructor width and `StepKernelContext` width remain deferred watchlist items unless LKW exposes concrete implementation or testing pain.

---

## 0b. Cursor token guardrails

Every LKW iteration must bound read scope, search scope, and test scope so Cursor does not burn tokens on repo-wide exploration.

### Rules

- Every implementation prompt must include: **Goal**, **Read scope**, **Code search scope**, **Stop condition**, **Do not touch**, **Test scope**, **Report format**.
- Cursor must **not** audit the whole repository unless the prompt explicitly requires it.
- Cursor must **not** run repo-wide glob/search across all Python or Markdown files.
- Cursor must **stop** after the first hit grep/search once the implementation point is located — then implement immediately.
- Cursor must read **only cited document sections**, not full architecture/plan hubs or domain packs.
- Cursor may expand scope only when a cited file imports a dependency that must change, a targeted test fails because of a cross-module contract, the implementation point does not exist in the given scope, or [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) reveals a real need to change scaffold, env, Docker, or CI.
- Expansion budget: at most **3 files** outside read scope per task. After 3 reads or 3 failed greps: stop and report.
- Pattern anchor for Tier-2 catalog-tool work: [`intergrax/agents/authoring/runtime_tool_helpers.py`](../../../intergrax/agents/authoring/runtime_tool_helpers.py).
- Default tests: new/changed test + one narrow smoke, not the full suite.
- Default report: terse — changed files, tests run, pass/fail, commit SHA, platform propagation yes/no.

### Prompt template

```text
Repo: `jakbuczarnecki/intergrax`, branch `development`.

Goal:
<one sentence — task ID + outcome>

Read scope:
- `<path>` — section `<id>` only
- `<path>` — `<function or line range if known>`
- `intergrax/agents/authoring/runtime_tool_helpers.py` — when task invokes catalog tools
- existing tests: `<path or glob under one module>`

Implementation point:
- `<agents/<agent>/steps/<job>.py>` — edit here; do not search runtime for invoke_tool pattern

Code search scope:
Search only:
- `<pattern>`
- `<pattern>`
Paths: `<tier/module glob>` — not repo-wide glob search

Stop condition:
Stop reading/searching once implementation point is located; implement immediately.

Do not touch:
- <explicit out-of-scope items>

Test scope:
- new/changed test: `<path>`
- narrow smoke: `<path>`

Report format:
Terse: changed files, tests run, pass/fail, commit SHA, platform propagation yes/no + brief reason.

Acceptance:
- <task-specific acceptance bullets>
```

---

## 0c. Operator workflow for this track

This track is executed one task at a time.

1. Select exactly one task from this plan.
2. Describe the goal, known status, implementation/diagnostic scope, acceptance criteria, and explicit out-of-scope items.
3. For complex tasks, prepare a scoped Cursor instruction first.
4. For simple tasks, implement only after explicit operator confirmation.
5. Do not change repository files, create commits, or update docs unless the operator explicitly asks for it.
6. If a diagnostic task finds a defect, classify it first; implementation is a separate follow-up unless the operator approves immediate repair.
7. Before closing a task, classify each discovered bug, workaround, repeated pattern, missing diagnostic, scaffold gap, config mismatch, Docker/build issue, dependency issue, or CI/runbook gap as `LKW-specific`, `Platform-reusable`, or `Platform-reusable deferred` according to [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) §3.

---

## 1. Wave queue

| ID | Title | Depends | Status | Priority |
|----|-------|---------|--------|----------|
| LKW.0 | Scaffold + architecture v2 | — | **Done** | — |
| LKW.3 | `filesystem.*` + allowlist | LKW.0 | **Done** | — |
| LKW-H0 | Minimal runtime hardening for product proof | LKW.0 | **Closed for LKW.1 entry / monitor** | Critical |
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW-H0 | **Closed in scope — product proof passed after LKW.1.15** | Critical |
| LKW-H1 | LKW live trace/evidence inspection and tool-call accounting | LKW.1 | **Completed for LKW.2 entry; deferred platform topics tracked separately** | High |
| LKW.2 | Graph pipeline + `local.workspace.*` skills | LKW.1, LKW-H1 | **Closed — pipeline proof passed** | High |
| LKW.4 | Background ingest queue (`message_bus`) | LKW.1 | Planned | Medium |
| LKW.5 | `LKW_DATA_HOME` + persistent vector storage | LKW.1 | Planned | High |
| LKW.6 | OS daemon + interaction intake router | LKW.1 | Planned | High |
| LKW.6b | Slack Socket Mode (optional) | LKW.6 | Planned | Medium |
| LKW.7 | File watcher + incremental index | LKW.4, LKW.5 | Planned | Medium |
| LKW.8 | Tray thin client | LKW.6 | Deferred | Low |
| LKW-H2 | Evidence/maturity wording cleanup | LKW.1 | Planned | Medium |
| LKW-H3 | Packaging/adoption simplification | LKW.1 or LKW.2 | Planned | Medium |
| LKW-W | Deferred architecture watchlist | LKW proof pain only | Deferred | Watch |

---

## 2. Closed support wave — LKW-H0: minimal runtime hardening for product proof

This is not a broad harness refactor wave. These tasks are allowed because they directly improve safety, bounded execution, and diagnosability for LKW.1.

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW-H0.1 | Strict/product runtime must not silently default-allow when policy wiring is missing | runtime policy / kernel wiring | Closed / monitor | Update shared config/scaffold guidance if unsafe defaults are generic |
| LKW-H0.2 | Add `max_steps` boundary regression test | runtime kernel or ACP session tests | Closed / monitor | Update generated guidance only if step-limit semantics are exposed to app/agent authors |
| LKW-H0.3 | Emit diagnostic/runtime event for post-finalization hook failure | Nexus lifecycle / runtime events | Closed / monitor | Propagate generic diagnostic/event pattern to runtime docs/templates if applicable |

Out of scope: `NexusLoop` constructor refactor, `StepKernelContext` decomposition, hosted observability product, full packaging split, and product features outside LKW safety/diagnosability.

---

## 3. Closed wave — LKW.1: Domain UAEP proof

### Goal

Deliver the first real LKW product and platform proof:

```text
POST /v1/local_workspace/run
  -> local.workspace.index using metadata.source_paths
  -> rag.ingest_document
  -> local.workspace.search
  -> rag.retrieve with tenant-scoped evidence
  -> local.workspace.synthesize with evidence/draft
  -> workspace.write_file under shadow root
```

### Result

Status: **Closed in scope after LKW.1.15**.

Product proof verified live:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Important boundary:

```text
Standalone synthesize with message-only input can still return content_missing.
That is not an LKW.1 closeout blocker; it belongs to LKW.2 pipeline/orchestration,
where search evidence should be passed into synthesize automatically.
```

### LKW.1 task map

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/` | **Closed** | Update agent scaffold/docs if canonical |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/` | **Closed** | Update evidence/result patterns if reusable |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/` | **Closed** | Update scaffold guidance for shadow-write outputs if generic |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | application tests | **Closed** | Add scaffold/test template if canonical |
| LKW.1.5 | Env/settings parity check | `.env.example`, `host/settings.py`, docs | **Closed / configured** | Inform scaffolded app settings pattern |
| LKW.1.6 | Docker/run parity | Dockerfile, compose, build/run docs | **Closed** | Docker build/run lessons propagated or recorded |
| LKW.1.7 | Live HTTP smoke baseline | Docker compose + `/health` + `/agents` + `/run` | **Partial / superseded** | Showed host/routing worked but RAG-backed flow was not yet proven |
| LKW.1.8 | Diagnose live RAG ingest failure | Docker logs/runtime output + RAG path | **Diagnosed** | Queued Qdrant id, tenant scope, gateway registry, and diagnostics follow-ups |
| LKW.1.9 | Fix Qdrant-compatible RAG ingest point ids | Qdrant provider | **Completed** | Platform-reusable point-id normalization |
| LKW.1.10 | Fix tenant scope consistency for live RAG ingest/retrieve | RAG scope + LKW shared helpers | **Completed** | Platform-reusable tenant/workspace/user source-of-truth handling |
| LKW.1.11 | Fix runtime tool gateway registry parity for catalog tools | runtime/app tool wiring | **Completed** | Platform-reusable catalog registry parity path |
| LKW.1.12 | Fix `decision_emitted` runtime event phase mismatch | runtime events/planning | **Completed** | Platform-reusable event catalog/schema correctness |
| LKW.1.13 | Restore local_indexer live RAG ingest execution | UAEP/ACP bridge + local indexer | **Completed** | Platform-reusable host catalog tool invocation bridge |
| LKW.1.14 | Final live product smoke attempt | Docker HTTP live smoke | **Partial** | Search failed due tenant-scoped retrieve/local_search allowlist blockers |
| LKW.1.15 | Fix tenant-scoped `rag.retrieve` + `local_search` tool allowlist; rerun product smoke | RAG scope/service + `agents/local_search/contract.py` | **Completed / closeout passed** | Platform-reusable wired-retriever rebinding; LKW search live proof |

### LKW.1.7–LKW.1.15 blocker history

Detailed history lives in:

- [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md)
- [`journal/2026-06-26-lkw-1-11-live-verification.md`](journal/2026-06-26-lkw-1-11-live-verification.md)
- [`journal/2026-06-26-lkw-1-12-1-13-live-ingest-unblocked.md`](journal/2026-06-26-lkw-1-12-1-13-live-ingest-unblocked.md)

Summary:

| ID | Result |
|----|--------|
| LKW.1.9 | Qdrant point-id compatibility fixed. |
| LKW.1.10 | Tenant scope consistency fixed. |
| LKW.1.11 | Runtime tool registry parity fixed. |
| LKW.1.12 | `decision_emitted` phase mismatch fixed. |
| LKW.1.13 | UAEP/ACP catalog invocation bridge fixed; live index ingests into Qdrant. |
| LKW.1.14 | Full smoke exposed tenant-scoped retrieve and local_search allowlist blockers. |
| LKW.1.15 | Tenant-scoped retrieve fixed; local_search allowlist fixed; live product path passed. |

Current live proof after LKW.1.15:

```text
health=ok
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
search=results=1, tenant-scoped marker evidence returned
synthesize=shadow artifact written when evidence is supplied
source immutability=original fixture unchanged
logs=no RuntimeEventSchemaError, unknown_capability_tool, tool_gateway_not_available, ingest_failed, retriever_failed
qdrant=local_workspace__tenant__lkw-smoke, tenant_id=lkw-smoke, workspace_id=lkw-final-20260627103000
```

### Product acceptance criteria

- [x] Docker stack can run the LKW application host.
- [x] `/health` responds successfully.
- [x] `/v1/local_workspace/agents` lists `local.workspace.index`, `local.workspace.search`, and `local.workspace.synthesize`.
- [x] `POST /v1/local_workspace/run` reaches the index agent.
- [x] `local.workspace.index` invokes `rag.ingest_document` and ingests at least one chunk in the live path.
- [x] Follow-up search returns tenant-scoped answer/evidence referencing ingested content.
- [x] Synthesize writes a shadow artifact based on retrieved/supplied evidence in the live path.
- [x] Original user files are not modified.
- [x] No Slack, tray, watcher, or OS service required.

### Platform acceptance criteria

- [x] Qdrant point-id compatibility fixed in LKW.1.9.
- [x] Tenant/workspace/user source-of-truth fixed in LKW.1.10.
- [x] Runtime gateway / application tool registry parity fixed in LKW.1.11.
- [x] Runtime event phase contract fixed in LKW.1.12.
- [x] UAEP/ACP host catalog tool invocation bridge fixed in LKW.1.13.
- [x] Tenant-scoped retrieve and local_search allowed-tool declaration fixed in LKW.1.15.
- [x] Live index tool-call accounting fixed in LKW-H1.1.
- [x] Broader trace/evidence inspection lessons are reflected in LKW-H1.2 (curated `lkw_evidence.v1` slice); H1.3 smoke assertions passed (see §4).
- [ ] Any remaining env/settings/scaffold/Docker/CI implications from LKW.1 are recorded in H1/H3 if they prove reusable.

### Known follow-ups after LKW.1 *(historical — superseded by LKW-H1 / PF closeout; see §4)*

| Follow-up | Classification | Target *(at time of LKW.1 closeout)* |
|----------|----------------|--------|
| Search/synthesize per-tool accounting (`rag.retrieve`, `workspace.write_file`) in trace/summary | Observability/accounting | **Closed** — LKW-H1.3 |
| RuntimeEvent `TOOL_*` at event layer | Observability/platform | **Closed** — LKW-PF1 / **LKW-PF1A** (`runtime_event_summary.v1`) |
| `RunArtifactBundle` / `WorkspaceArtifactRef` platform wiring | Observability/platform | **Closed** — LKW-PF2 / **LKW-PF2A** (`shadow_workspace_id` propagation) |
| Policy decisions, raw tool reason/error at RuntimeEvent layer | Observability/platform | Platform deferred |
| RAG ingest-specific observability contract | Observability/platform | Platform deferred (optional) |
| Async runtime plugin coroutine warnings | Observability/platform | **Closed** — LKW-DF1 (platform `RuntimeEventBus.record` async handler dispatch before LKW.2.4C) |
| Standalone synthesize with message-only input returns `content_missing` | Pipeline/orchestration input contract | LKW.2 |
| Developer first-run/adoption simplification | Packaging/adoption | LKW-H3 |

### Out of scope for LKW.1

- Tray UI.
- Slack.
- File watcher.
- OS service installer.
- Full `local.workspace.*` skill bundle, except minimal stubs explicitly needed for LKW.1 tests.
- Hosted observability dashboard.
- Grafana/Tempo/OpenTelemetry Collector as a blocker for live proof.
- Broad harness refactor unrelated to LKW acceptance or platform propagation.

---

## 4. LKW-H1: live trace/evidence inspection for LKW runs

### Goal

Make one real LKW run inspectable without reading internal runtime code, and ensure the trace/evidence/accounting pattern is reusable by future applications.

LKW-H1 is **not** the hosted observability stack. It is the minimum local inspection surface needed for a developer/operator to understand a run. Grafana, Tempo, and an OpenTelemetry Collector remain optional future operational infrastructure unless a later task explicitly scopes them.

### Known diagnosed input

LKW.1.15 passed product behavior. H1.1 fixed the first accounting gap for live index runs:

```text
local.workspace.index -> rag.ingest_document -> application_run_summary.v1 total_tool_calls=1
```

H1.1–H1.3 are closed for LKW.2 entry. Platform event-layer topics originally tracked here were closed in **LKW-PF1** / **LKW-PF1A** (RuntimeEvent `TOOL_*`), **LKW-PF2** / **LKW-PF2A** (`RunArtifactBundle` / `WorkspaceArtifactRef`, `shadow_workspace_id`). Remaining deferred platform topics: optional RAG ingest observability contract; policy/raw tool reason decisions at RuntimeEvent layer. Async runtime plugin coroutine warnings closed in **LKW-DF1**.

H1 must improve visibility. It must not replace or reopen product execution blockers that are already fixed in LKW.1.9–LKW.1.15.

### Required inspection fields

For every LKW proof run, the operator should be able to inspect:

- submitted task and capability;
- task id and run id;
- selected agent;
- step sequence;
- invoked tools and outcomes;
- raw tool status and reason/error;
- policy decisions;
- RAG ingest/retrieve evidence;
- shadow workspace artifact path;
- terminal outcome;
- diagnostics from non-fatal lifecycle/finalization failures.

### H1.1 result

Status:

```text
PASSED for live index tool-call accounting
```

Summary:

```text
62621bc1 fixed catalog tool-call recording and kernel harvest for RuntimeExecutionContext.invoke_tool().
a22222e0 fixed the live UAEP bridge path by forwarding uaep_exec_ctx from build_uaep_step_context().
Focused tests passed: 11 passed, 4 warnings.
Live index smoke passed: accepted=1, ingested=1, chunks=1, total_tool_calls=1.
```

Known non-blocking warnings *(historical — closed LKW-DF1)*:

```text
Application tests emitted async runtime plugin warnings about coroutines not awaited in event_bus/task_trace handlers.
Closed in LKW-DF1: RuntimeEventBus.record now runs or schedules async handler results on the sync dispatch path.
```

### Tasks

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW-H1.1 | Fix live run observability and tool-call accounting, including `total_tool_calls=0` | runtime/tool accounting + LKW host evidence | **Completed / index live accounting passed** | Reusable UAEP/ACP RuntimeExecutionContext tool-call accounting covered by platform tests |
| LKW-H1.2 | Ensure LKW run emits/records tool, policy, RAG, and shadow artifact evidence | runtime events + LKW host | **Completed / PASSED WITH PLATFORM FOLLOW-UPS** | Curated `lkw_evidence.v1` read model reusable by app hosts; platform event wiring deferred |
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | application tests | **Passed with platform follow-ups** | Update generated app test pattern if reusable |

Acceptance:

- [x] A reviewer can see what happened in an LKW run from task submission to terminal result (`application_run_summary.v1` + `lkw_evidence.v1`).
- [x] RAG evidence counts/source refs and shadow artifact path/ref are visible via typed `lkw.*_summary.v1` diagnostics (redacted).
- [x] `total_tool_calls=0` is fixed for the live index path (`rag.ingest_document`).
- [x] Search/synthesize per-tool accounting is verified for `rag.retrieve` and `workspace.write_file` in trace/summary.
- [ ] Raw tool status/reason/error and policy decisions at RuntimeEvent layer — platform deferred.
- [x] No hosted dashboard or external observability backend is required.
- [x] Platform proof checklist in §0a is completed for H1.2 closeout scope.

### H1.2 result

Status:

```text
PASSED WITH PLATFORM FOLLOW-UPS
```

Delivered in LKW host:

```text
- serving/evidence_slice.py — curated lkw_evidence.v1 from AgentRunTrace step diagnostics
- serving/run_metadata.py — attach_lkw_evidence_metadata() on TaskResult
- serving/fastapi_router.py — lkw_evidence.v1 on POST /v1/local_workspace/run response metadata
- typed diagnostics: lkw.index_summary.v1, lkw.search_summary.v1, lkw.synthesize_summary.v1
- unsafe field redaction (query_text, content, raw_chunks, …)
- full_trace / agent_run_trace not exposed on HTTP response (by design)
```

Focused tests:

```text
uv run pytest applications/local_workspace_application/tests/test_evidence_slice.py -q
uv run pytest applications/local_workspace_application/tests/test_lkw_evidence_metadata.py -q
uv run pytest applications/local_workspace_application/tests/test_lkw_evidence_live_smoke.py -q

Result: 9 passed, 12 warnings
```

Verified smoke assertions:

```text
index — lkw.index_summary.v1 fields + total_tool_calls>0 + no raw fixture text in evidence
search — lkw.search_summary.v1 num_results/evidence_count/source_refs + redaction
synthesize — lkw.synthesize_summary.v1 shadow_write/artifact_path|artifact_ref + redaction
```

Platform follow-ups deferred at H1.2 closeout *(historical — superseded)*:

```text
RunArtifactBundle / WorkspaceArtifactRef platform wiring -> closed LKW-PF2 / LKW-PF2A
RuntimeEvent TOOL_* HTTP visibility -> closed LKW-PF1 / LKW-PF1A
search/synthesize per-tool accounting + LKW-H1.3 smoke/assertion hardening -> closed LKW-H1.3
ACP shadow_workspace_id propagation -> closed LKW-PF2A
Still deferred: optional RAG ingest observability contract; policy/raw tool reason at RuntimeEvent layer
Async runtime plugin coroutine warnings -> closed LKW-DF1
```

**LKW-PF1 (2026-06-27):** PASSED WITH FOLLOW-UP — immediate `TOOL_*` RuntimeEvents wired in `RuntimeExecutionContext.invoke_tool`; unit coverage in `tests/unit/contracts/test_invoke_tool_runtime_events.py`. Follow-up closed in **LKW-PF1A** (`runtime_event_summary.v1` on HTTP run metadata).

**LKW-PF1A (2026-06-28):** CLOSED — `serving/runtime_event_metadata.py` aggregates platform `TOOL_*` events into `runtime_event_summary.v1` on POST `/v1/local_workspace/run` (counts by tool_id/type only; no raw args). Live smoke asserts index/search/synthesize tool visibility.

**LKW-CI1 (2026-06-28):** CLOSED — live smoke test avoids circular `intergrax.runtime.task` package import; `AgentEngine` exposes `shadow_manager`/`sandbox_manager` for graph executor isolation wiring.

**LKW-PF2 (2026-06-27):** PASSED WITH FOLLOW-UP — platform contract reused (`intergrax/contracts/task_artifacts.py`: `RunArtifactBundle`, `WorkspaceArtifactRef`; key `run_artifact_bundle.v1`). LKW wires synthesize shadow artifacts through existing nexus task-finisher bundle rollup and promotes bundle on HTTP metadata via `serving/run_artifact_metadata.py`. Domain diagnostic `lkw.synthesize_summary.v1` preserved; bundle exposes refs/paths only (no raw synthesized content). Follow-up closed in **LKW-PF2A**.

**LKW-PF2A (2026-06-28):** CLOSED — `shadow_workspace_id` propagates through ACP session (`acp_run.py` + `exec_ctx_isolation.py`) and UAEP path (`route.extra` → `runtime_answer_to_agent_result`) into `AgentExecutionResult.structured_data`; `run_artifact_bundle_builder.py` resolves workspace refs deterministically. Unit coverage: `tests/unit/agents/test_acp_shadow_workspace_propagation.py`.




### Platform follow-ups before LKW.2

| ID | Title | Narrow scope |
|----|-------|--------------|
| **LKW-PF1** | Immediate tool RuntimeEvent emission | **PASSED WITH FOLLOW-UP** — `invoke_tool` emits `TOOL_REQUESTED/COMPLETED/FAILED/DENIED` with generic platform payload (`tool_id`, `status`, `latency_ms`, `args_digest`, `error_code`, spine ids). HTTP visibility closed in **LKW-PF1A**. |
| **LKW-PF1A** | Safe runtime event HTTP read surface | **CLOSED** — `runtime_event_summary.v1` on POST `/v1/local_workspace/run` metadata; redacted TOOL_* counts by tool_id. |
| **LKW-CI1** | Live smoke standalone import cleanup | **CLOSED** — test avoids circular task package import; graph isolation wiring unblocked for standalone smoke. |
| **LKW-PF2** | RunArtifactBundle / WorkspaceArtifactRef for synthesize artifacts | **PASSED WITH FOLLOW-UP** — reuses platform `run_artifact_bundle.v1` / `WorkspaceArtifactRef`; LKW HTTP responses expose bundle via `run_artifact_metadata.py`; synthesize diagnostic correlated by `artifact_path` / `artifact_ref`. Follow-up closed in **LKW-PF2A**. |
| **LKW-PF2A** | ACP shadow_workspace_id propagation | **CLOSED** — `isolation_structured_data_from_exec_ctx` exports typed `shadow_workspace_id` into ACP/UAEP execution structured_data; bundle builder correlates synthesize workspace refs without LKW-only workaround. |
### H1.3 result

Status:

```text
PASSED WITH PLATFORM FOLLOW-UPS
```

Focused tests:

```text
uv run pytest applications/local_workspace_application/tests/test_evidence_slice.py applications/local_workspace_application/tests/test_lkw_evidence_metadata.py applications/local_workspace_application/tests/test_lkw_evidence_live_smoke.py tests/unit/agents/authoring/test_runtime_rag_call_recording.py tests/unit/runtime/kernel/test_step_kernel.py tests/unit/agents/authoring/test_uaep_step_bridge.py -q

Result: 38 passed, 12 warnings
```

Verified:

```text
index -> rag.ingest_document as ToolCallRecord, application_run_summary.v1 total_tool_calls>0
search -> rag.retrieve as ToolCallRecord + RagCallRecord, total_rag_calls propagated, hit_count/collection_id populated safely
synthesize -> workspace.write_file visible in safe metadata, raw content not exposed
runtime remains application-agnostic
typed diagnostics/evidence slice pattern is reusable across Tier-3 apps
developer ergonomics acceptable; helper/template/docs follow-up recommended
```

Platform follow-ups remain outside H1.3 *(current deferred queue)*:

```text
optional RAG ingest observability contract
policy/raw tool reason decisions at RuntimeEvent layer
developer ergonomics helper/template/docs
```

Async runtime plugin coroutine warnings in event_bus/task_trace handlers -> closed LKW-DF1.

Closed since H1.3 closeout: RuntimeEvent TOOL_* (LKW-PF1 / LKW-PF1A); RunArtifactBundle / WorkspaceArtifactRef + ACP shadow_workspace_id (LKW-PF2 / LKW-PF2A).

**LKW-PF1:** PASSED WITH FOLLOW-UP — see IMPLEMENTATION_PLAN §Platform follow-ups before LKW.2.
---

## 5. LKW.2: graph pipeline + local workspace skills

**Progress:** LKW.2.1–LKW.2.4C **done**; **LKW.2 closeout passed (2026-06-28)**. Live pipeline proof for `local.workspace.pipeline` **passed** (LKW.2.4C). Direct-capability regression smoke passed. **Next platform phase:** OBS-EXPORT-1.

| ID | Task | Module | Owner | Status | Platform propagation |
|----|------|--------|-------|--------|----------------------|
| LKW.2.1 | Add `intergrax/skills/providers/local/` bundle | Tier-0 skills | Tier-0 | **Done** | Update skill scaffold/catalog docs if pattern is reusable |
| LKW.2.2 | Add `skill_ids` to local agent contracts | `agents/local_*` contracts | Tier-2 | **Done** | Update agent scaffold to generate correct `skill_ids` pattern if needed |
| LKW.2.3 | Enable `skill_bundles=["harness", "local"]` | `host/environment_profile.py` | Tier-3 | **Done** | Update app scaffold/environment templates if bundle pattern is generic |
| LKW.2.4A | Register graph/pipeline capability `local.workspace.pipeline` | manifest / graph spec | Tier-1/3 | **Done** — graph spec registered | Update app scaffold or graph docs if this becomes canonical multi-agent pipeline pattern |
| LKW.2.4A1 | Allow graph trigger capabilities in package closure | applications packaging | Tier-3 | **Done** — graph trigger package closure | — |
| LKW.2.4A2 | Align pipeline graph nodes with agent roster | graph spec | Tier-1/3 | **Done** — graph node roster identity | — |
| LKW.2.4B | Pass search evidence into pipeline synthesize | pipeline graph | Tier-1/3 | **Done** — search evidence handoff into synthesize | — |
| LKW.2.4C | Live pipeline proof and metadata preservation | live verification | Tier-3 | **Done** — live pipeline proof passed | — |

Acceptance:

- [x] Single `POST /v1/local_workspace/run` with `capability=local.workspace.pipeline` can run index → search → synthesize without manual capability selection *(LKW.2.4C)*.
- [x] Tool access is resolved through `skill_ids`, not ad-hoc allowlists in agent code *(LKW.2.1–LKW.2.2)*.
- [x] Existing LKW.1 index/search/synthesize direct capabilities still pass *(LKW.2 closeout smoke — `test_lkw_evidence_live_smoke_index/search/synthesize`)*.
- [x] Pipeline passes search evidence/draft into synthesize so message-only `content_missing` is not exposed in the normal product pipeline *(LKW.2.4B)*.
- [x] Metadata preservation on pipeline run: `application_run_summary.v1`, redacted `lkw_evidence.v1`, `runtime_event_summary.v1`, `run_artifact_bundle.v1` *(LKW.2.4C)*.
- [x] Platform proof checklist in §0a is completed *(LKW.2 closeout — see below)*.

**Observability boundary (OBS-EXPORT):**

- LKW.2.4 pipeline proof is a prerequisite workload for future platform **OBS-EXPORT** work (see [`docs/plan/OBSERVABILITY.md`](../../../docs/plan/OBSERVABILITY.md) Phase OBS-EXPORT); LKW is the proof workload, not the integration owner.
- **INTEGRATIONS-1D / LKW observability platform wiring — Done (2026-06-28):** `build_local_workspace_observability_plugins` composes platform **`ObservabilityExportOperatorConfig`** → **`build_otlp_observability_export_runtime_plugin`**; disabled by default; registered only via explicit LKW factory/bootstrap opt-in; no LKW-specific exporter.
- LKW.2 pipeline remains unchanged — no graph/pipeline rework in INTEGRATIONS-1D.
- Vendor observability integrations (Langfuse, Arize/Phoenix, Elasticsearch) remain **out of scope** for LKW; OTLP uses platform integration path only.
- LKW must continue to use platform observability contracts only: `application_run_summary.v1`, `lkw_evidence.v1`, `runtime_event_summary.v1`, `run_artifact_bundle.v1`, RuntimeEvent `TOOL_*`, ToolCallRecord/RagCallRecord, WorkspaceArtifactRef.
- Do not add LKW-specific exporters, telemetry buses, vendor SDK calls, trace stores, or observability workarounds.
- **OBS-EXPORT-5** (Langfuse/Arize/Phoenix adapters) remains the next platform phase; LKW is not the integration owner.

### LKW.2 closeout result

Status:

```text
CLOSED — PIPELINE PROOF PASSED
```

Focused closeout smoke:

```text
uv run pytest applications/local_workspace_application/tests/test_lkw_evidence_live_smoke.py \
  applications/local_workspace_application/tests/host/test_local_workspace_environment_profile.py -q -W default

Result: 13 passed (2026-06-28); no coroutine-never-awaited warnings on focused path
```

Verified:

```text
direct — local.workspace.index / search / synthesize still pass after pipeline graph work
pipeline — local.workspace.pipeline runs local_indexer -> local_search -> local_synthesizer -> shadow artifact
metadata — application_run_summary.v1, lkw_evidence.v1, runtime_event_summary.v1, run_artifact_bundle.v1 preserved on pipeline run
```

### LKW.2 closeout — §0a platform proof checklist

- [x] Did every discovered bug, workaround, repeated pattern, missing diagnostic, scaffold gap, config mismatch, Docker/build issue, dependency issue, and CI/runbook gap receive a classification? — Yes; see deferred queue below.
- [x] Does this change belong only to LKW, or should it move to shared platform code? — Product pipeline/graph spec is LKW; local skill bundle + graph trigger closure patterns flagged for future scaffold propagation (§5 task table).
- [ ] Should application scaffold generate this pattern for the next product host? — Deferred to LKW-H3 (not an LKW.2 closeout blocker).
- [ ] Should agent scaffold generate this contract, test, or documentation pattern? — LKW.2.2 `skill_ids` pattern recorded; scaffold propagation deferred to LKW-H3.
- [x] Does `.env.example` match `host/settings.py` and production validation? — Unchanged by LKW.2; LKW.1 parity still holds.
- [x] Does `pyproject.toml` need a dependency split or optional dependency group? — No LKW.2 closeout change required.
- [x] Does Docker still build and run with the correct files, env profile, port, and healthcheck? — Unchanged by LKW.2; LKW.1 Docker path still valid.
- [x] Does CI need a new application smoke test or Docker build check? — Live smoke in `test_lkw_evidence_live_smoke.py` covers direct capabilities + pipeline; environment profile tests cover graph spec closure.
- [x] Does the deploy/runbook still describe the real execution path? — README and USER_JOURNEY unchanged; pipeline capability documented.
- [x] Does the implementation plan identify both the LKW work and the platform propagation work? — Yes (§5 task table + deferred queue).

Platform-reusable deferred at LKW.2 closeout *(not blockers)*:

| Follow-up | Classification | Notes |
|----------|----------------|-------|
| RuntimeEvent `TOOL_*` on ACP graph path | Platform-reusable deferred | Tool visibility via `application_run_summary.total_tool_calls` works; event-bus `TOOL_*` on graph path incomplete |
| RAG ingest-specific observability contract | Platform deferred (optional) | — |
| Policy/raw tool reason at RuntimeEvent layer | Platform deferred | — |
| Developer first-run/adoption simplification | LKW-H3 | Helper/template/docs ergonomics |

---

## 6. Post-LKW.1 hardening and adoption waves

### LKW-H2 — evidence/maturity wording cleanup

| ID | Task | Module | Acceptance |
|----|------|--------|------------|
| LKW-H2.1 | Clarify architecture maturity vs live product proof vs production claim | README / product-validation docs / LKW docs | Documentation does not imply deterministic evidence is full production certification |
| LKW-H2.2 | Add LKW proof status wording | LKW docs | LKW is described as product proof passed for LKW.1, with H1/H2/H3 still tracking maturity/adoption follow-ups |

### LKW-H3 — packaging/adoption simplification

| ID | Task | Module | Acceptance |
|----|------|--------|------------|
| LKW-H3.1 | Define minimal developer first-run path for LKW and scaffolded apps | README / BUILD_AND_DEPLOY / LKW docs / scaffold docs | **Done** — first-run section added to README.md, linked from docs/README.md |
| LKW-H3.2 | Decide optional dependency split | `pyproject.toml` / docs | Minimal install story is clear; heavy optional stacks are documented or split |
| LKW-H3.3 | Propagate adoption lessons to application scaffold | `intergrax/scaffold/` | Next generated product application inherits the improved env/build/deploy documentation pattern |

