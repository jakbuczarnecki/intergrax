# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** · **LKW.1 Closed in scope** · **Active queue: LKW-H1.2 → LKW-H1.3 → LKW.2**

Latest live proof snapshot: **2026-06-27 — LKW.1.15 PASSED / LKW.1 PRODUCT PROOF CLOSED IN SCOPE**. Tenant-scoped `rag.retrieve` works live for `tenant_id=lkw-smoke` with workspace filtering; `local.workspace.search` returns marker evidence; `local.workspace.synthesize` writes a shadow artifact when evidence/draft is supplied. Product closeout path verified live:

```text
index -> search with tenant-scoped evidence -> synthesize with evidence -> shadow artifact only
```

Latest observability snapshot: **2026-06-27 — LKW-H1.1 PASSED for live index tool-call accounting**. Live UAEP/Nexus index runs now report `total_tool_calls=1` for `rag.ingest_document` after forwarding `uaep_exec_ctx` through `build_uaep_step_context()`.

Current status source of truth: [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md).  
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
| LKW-H1 | LKW live trace/evidence inspection and tool-call accounting | LKW.1 | **In progress — H1.1 passed; H1.2 next** | High |
| LKW.2 | Graph pipeline + `local.workspace.*` skills | LKW.1, LKW-H1 | Planned | High |
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
- [ ] Broader trace/evidence inspection lessons are reflected in LKW-H1.2/H1.3.
- [ ] Any remaining env/settings/scaffold/Docker/CI implications from LKW.1 are recorded in H1/H3 if they prove reusable.

### Known follow-ups after LKW.1

| Follow-up | Classification | Target |
|----------|----------------|--------|
| Search/synthesize tool-call visibility needs live verification after index accounting fix | Observability/accounting | LKW-H1.2 / LKW-H1.3 |
| Need inspectable tool status, raw reason/error, RAG evidence, shadow artifact path | Observability/evidence | LKW-H1.2 |
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

The remaining H1 work is broader inspection: search/synthesize accounting verification, raw tool status/reason, RAG evidence, policy decisions, and shadow artifact paths.

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

Known non-blocking warnings:

```text
Application tests emitted async runtime plugin warnings about coroutines not awaited in event_bus/task_trace handlers.
Those warnings are tracked separately from H1.1 tool-call accounting unless they become reproducible operational failures.
```

### Tasks

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW-H1.1 | Fix live run observability and tool-call accounting, including `total_tool_calls=0` | runtime/tool accounting + LKW host evidence | **Completed / index live accounting passed** | Reusable UAEP/ACP RuntimeExecutionContext tool-call accounting covered by platform tests |
| LKW-H1.2 | Ensure LKW run emits/records tool, policy, RAG, and shadow artifact evidence | runtime events + LKW host | **Next** | Update event/trace scaffold or docs if reusable |
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | application tests | Planned | Update generated app test pattern if reusable |

Acceptance:

- [ ] A reviewer can see what happened in an LKW run from task submission to terminal result.
- [ ] Tool calls, policy decisions, RAG evidence, raw tool reason/error, and shadow artifact path are visible.
- [x] `total_tool_calls=0` is fixed for the live index path (`rag.ingest_document`).
- [ ] Search/synthesize tool-call visibility is verified for `rag.retrieve` and `workspace.write_file`.
- [x] No hosted dashboard or external observability backend is required.
- [ ] Platform proof checklist in §0a is completed.

---

## 5. LKW.2: graph pipeline + local workspace skills

| ID | Task | Module | Owner | Platform propagation |
|----|------|--------|-------|----------------------|
| LKW.2.1 | Add `intergrax/skills/providers/local/` bundle | Tier-0 skills | Tier-0 | Update skill scaffold/catalog docs if pattern is reusable |
| LKW.2.2 | Add `skill_ids` to local agent contracts | `agents/local_*` contracts | Tier-2 | Update agent scaffold to generate correct `skill_ids` pattern if needed |
| LKW.2.3 | Enable `skill_bundles=["harness", "local"]` | `host/environment_profile.py` | Tier-3 | Update app scaffold/environment templates if bundle pattern is generic |
| LKW.2.4 | Add graph/pipeline capability `local.workspace.pipeline` | manifest / graph spec | Tier-1/3 | Update app scaffold or graph docs if this becomes canonical multi-agent pipeline pattern |

Acceptance:

- [ ] Single `POST /v1/local_workspace/run` with `capability=local.workspace.pipeline` can run index → search → synthesize without manual capability selection.
- [ ] Tool access is resolved through `skill_ids`, not ad-hoc allowlists in agent code.
- [ ] Existing LKW.1 index/search/synthesize direct capabilities still pass.
- [ ] Pipeline passes search evidence/draft into synthesize so message-only `content_missing` is not exposed in the normal product pipeline.
- [ ] Platform proof checklist in §0a is completed.

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
| LKW-H3.1 | Define minimal developer first-run path for LKW and scaffolded apps | README / BUILD_AND_DEPLOY / LKW docs / scaffold docs | New developer can run host, index fixture, search, and synthesize from documented commands |
| LKW-H3.2 | Decide optional dependency split | `pyproject.toml` / docs | Minimal install story is clear; heavy optional stacks are documented or split |
| LKW-H3.3 | Propagate adoption lessons to application scaffold | `intergrax/scaffold/` | Next generated product application inherits the improved env/build/deploy documentation pattern |
