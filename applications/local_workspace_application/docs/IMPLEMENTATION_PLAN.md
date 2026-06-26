# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** · **LKW.1.1–LKW.1.13 Passed/Closed in scope** · **Active queue: LKW.1.14 → LKW-H1 → LKW.2**

Latest live proof snapshot: **2026-06-26 — LKW.1.13 PASSED**. The live Docker HTTP `local.workspace.index` path now reaches `rag.ingest_document` and Qdrant: `accepted=1`, `rejected=0`, `ingested=1`, `chunks=1`. The previous Qdrant point-id, tenant scope, runtime event phase, and live catalog tool invocation blockers are fixed. The remaining LKW.1 proof gap is the full product smoke: `index -> search -> synthesize -> shadow artifact only`.

Current status source of truth: [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md).  
Application-local history: [`journal/`](journal/).  
Platform register: [`docs/plan/PLATFORM_FOUNDATION.md` §6.3a LKW.*](../../docs/plan/PLATFORM_FOUNDATION.md#63a-business-backlog-register-consolidated).

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
- Pattern anchor for LKW Tier-2 catalog-tool work: [`agents/lkw_shared/PATTERN.md`](../../agents/lkw_shared/PATTERN.md).
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
- `agents/lkw_shared/PATTERN.md` — when task invokes catalog tools
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
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW-H0 | **Active — LKW.1.14 next** | Critical |
| LKW-H1 | LKW live trace/evidence inspection | LKW.1.14 | **Queued after final product smoke** | High |
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

## 3. Active wave — LKW.1: Domain UAEP proof

### Goal

Deliver the first real LKW product and platform proof:

```text
POST /v1/local_workspace/run
  -> local.workspace.index using metadata.source_paths
  -> rag.ingest_document
  -> local.workspace.search
  -> rag.retrieve with evidence
  -> local.workspace.synthesize
  -> workspace.write_file under shadow root
```

### Current LKW.1 task map

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/` | **Closed** | Update agent scaffold/docs if canonical |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/` | **Closed** | Update evidence/result patterns if reusable |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/` | **Closed** | Update scaffold guidance for shadow-write outputs if generic |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | application tests | **Closed** | Add scaffold/test template if canonical |
| LKW.1.5 | Env/settings parity check | `.env.example`, `host/settings.py`, docs | **Closed / configured** | Inform scaffolded app settings pattern |
| LKW.1.6 | Docker/run parity | Dockerfile, compose, build/run docs | **Closed** | Docker build/run lessons propagated or recorded |
| LKW.1.7 | Live HTTP smoke baseline | Docker compose + `/health` + `/agents` + `/run` | **Partial / superseded by LKW.1.14** | Showed host/routing worked but RAG-backed flow was not proven |
| LKW.1.8 | Diagnose live RAG ingest failure | Docker logs/runtime output + RAG path | **Diagnosed** | Queued Qdrant id, tenant scope, gateway registry, and diagnostics follow-ups |
| LKW.1.9 | Fix Qdrant-compatible RAG ingest point ids | Qdrant provider | **Completed** | Platform-reusable point-id normalization |
| LKW.1.10 | Fix tenant scope consistency for live RAG ingest/retrieve | RAG scope + LKW shared helpers | **Completed** | Platform-reusable tenant/workspace/user source-of-truth handling |
| LKW.1.11 | Fix runtime tool gateway registry parity for catalog tools | runtime/app tool wiring | **Completed** | Platform-reusable catalog registry parity path |
| LKW.1.12 | Fix `decision_emitted` runtime event phase mismatch | runtime events/planning | **Completed** | Platform-reusable event catalog/schema correctness |
| LKW.1.13 | Restore local_indexer live RAG ingest execution | UAEP/ACP bridge + local indexer | **Completed** | Platform-reusable host catalog tool invocation bridge |
| LKW.1.14 | Final live product smoke: index → search → synthesize | Docker HTTP live smoke | **Next** | Product closeout plus platform proof checklist before LKW-H1/LKW.2 |

### LKW.1.7–LKW.1.13 blocker history

Detailed history lives in:

- [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md)
- [`journal/2026-06-26-lkw-1-11-live-verification.md`](journal/2026-06-26-lkw-1-11-live-verification.md)
- [`journal/2026-06-26-lkw-1-12-1-13-live-ingest-unblocked.md`](journal/2026-06-26-lkw-1-12-1-13-live-ingest-unblocked.md)

Summary:

| ID | Result |
|----|--------|
| LKW.1.9 | Qdrant point-id compatibility fixed. |
| LKW.1.10 | Tenant scope consistency fixed. |
| LKW.1.11 | Runtime tool registry parity fixed in focused/unit scope. |
| LKW.1.12 | `decision_emitted` phase mismatch fixed; live index still did not ingest. |
| LKW.1.13 | UAEP/ACP catalog invocation bridge fixed; live index now ingests into Qdrant. |

Current live index proof after LKW.1.13:

```text
health={"status":"ok"}
agents=local_indexer, local_search, local_synthesizer
index=accepted=1, rejected=0, ingested=1, chunks=1
qdrant=tenant collection intergrax__tenant__lkw-smoke present
```

Known non-blocking follow-up:

```text
total_tool_calls=0 remains an observability/summary accounting issue.
```

Classification:

```text
LKW-H1 / observability follow-up, not LKW.1 execution blocker.
```

### LKW.1.14 goal

Run the final full live Docker HTTP product smoke:

```text
index fixture -> search marker/evidence -> synthesize shadow artifact -> verify source immutability
```

Acceptance:

- [ ] Docker stack healthy.
- [ ] `/health` succeeds.
- [ ] `/v1/local_workspace/agents` lists index/search/synthesize.
- [ ] `local.workspace.index` invokes `rag.ingest_document` through the live runtime gateway.
- [ ] `local.workspace.index` ingests fixture with `ingested>0` and `chunks>0`.
- [ ] `local.workspace.search` retrieves evidence referencing the fixture.
- [ ] `local.workspace.synthesize` writes only to shadow workspace.
- [ ] Original fixture remains unchanged.
- [ ] No `RuntimeEventSchemaError`.
- [ ] No `unknown_capability_tool`.
- [ ] No `tool_gateway_not_available`.
- [ ] Product and platform acceptance criteria are updated before LKW.1 closeout.

### Product acceptance criteria

- [x] Docker stack can run the LKW application host.
- [x] `/health` responds successfully.
- [x] `/v1/local_workspace/agents` lists `local.workspace.index`, `local.workspace.search`, and `local.workspace.synthesize`.
- [x] `POST /v1/local_workspace/run` reaches the index agent.
- [x] `local.workspace.index` invokes `rag.ingest_document` and ingests at least one chunk in the live path.
- [ ] Follow-up search returns answer/evidence referencing ingested content.
- [ ] Synthesize writes a shadow artifact based on retrieved evidence in the live path.
- [x] Original user files are not modified in previous smoke runs.
- [x] No Slack, tray, watcher, or OS service required.

### Platform acceptance criteria

- [ ] Platform proof checklist in §0a is completed for LKW.1 closeout.
- [x] Qdrant point-id compatibility fixed in LKW.1.9.
- [x] Tenant/workspace/user source-of-truth fixed in LKW.1.10.
- [x] Runtime gateway / application tool registry parity fixed in LKW.1.11.
- [x] Runtime event phase contract fixed in LKW.1.12.
- [x] UAEP/ACP host catalog tool invocation bridge fixed in LKW.1.13.
- [ ] Live diagnostic/inspection lessons are reflected in LKW-H1 or recorded as follow-up.
- [ ] Any env/settings/scaffold/Docker/CI implications from LKW.1.14 are recorded before closeout.

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

Make one real LKW run inspectable without reading internal runtime code, and ensure the trace/evidence pattern is reusable by future applications.

LKW-H1 is **not** the hosted observability stack. It is the minimum local inspection surface needed for a developer/operator to understand a run. Grafana, Tempo, and an OpenTelemetry Collector remain optional future operational infrastructure unless a later task explicitly scopes them.

### Known diagnosed input

LKW.1.8–LKW.1.13 showed that platform/tool execution issues can be hidden by summaries. H1 must improve visibility, but it must not replace execution blockers. The current known follow-up is `total_tool_calls=0` despite successful live ingest effects.

### Required inspection fields

For every LKW.1 proof run, the operator should be able to inspect:

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

### Tasks

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW-H1.1 | Define minimal LKW trace/evidence inspection contract, including raw tool status and reason/error | LKW host/debug docs or tests | Planned | Promote reusable inspection contract to platform docs if generic |
| LKW-H1.2 | Ensure LKW run emits/records tool, policy, RAG, and shadow artifact evidence | runtime events + LKW host | Planned | Update event/trace scaffold or docs if reusable |
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | application tests | Planned | Update generated app test pattern if reusable |

Acceptance:

- [ ] A reviewer can see what happened in an LKW run from task submission to terminal result.
- [ ] Tool calls, policy decisions, RAG evidence, raw tool reason/error, and shadow artifact path are visible.
- [ ] No hosted dashboard or external observability backend is required.
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
- [ ] Platform proof checklist in §0a is completed.

---

## 6. Post-LKW.1 hardening and adoption waves

### LKW-H2 — evidence/maturity wording cleanup

| ID | Task | Module | Acceptance |
|----|------|--------|------------|
| LKW-H2.1 | Clarify architecture maturity vs live product proof vs production claim | README / product-validation docs / LKW docs | Documentation does not imply deterministic evidence is full production certification |
| LKW-H2.2 | Add LKW proof status wording | LKW docs | LKW is described as product proof in progress until live acceptance is met |

### LKW-H3 — packaging/adoption simplification

| ID | Task | Module | Acceptance |
|----|------|--------|------------|
| LKW-H3.1 | Define minimal developer first-run path for LKW and scaffolded apps | README / BUILD_AND_DEPLOY / LKW docs / scaffold docs | New developer can run host, index fixture, search, and synthesize from documented commands |
| LKW-H3.2 | Decide optional dependency split | `pyproject.toml` / docs | Minimal install story is clear; heavy optional stacks are documented or split |
| LKW-H3.3 | Propagate adoption lessons to application scaffold | `intergrax/scaffold/` | Next generated product application inherits improved env/build/deploy documentation pattern |

Potential packaging direction:

```text
intergrax-core
intergrax-lab
intergrax-lkw
intergrax-rag
intergrax-all
```

Do not start a full packaging split before LKW.1 has a useful proof path, but capture dependency lessons during every wave.

---

## 7. Deferred architecture watchlist

These items are real architectural pressure points, but they must not block LKW.1.

| ID | Topic | Current decision | Trigger for action |
|----|-------|------------------|-------------------|
| LKW-W1 | `NexusLoop` constructor width | Accept as composition-root pressure | Refactor only if LKW requires repeated custom wiring, makes tests brittle, or forces duplicated bootstrap logic |
| LKW-W2 | `StepKernelContext` width | Accept as kernel execution-context pressure | Refactor only if unrelated concerns start changing together or test setup becomes excessive |

### LKW.1.6 follow-ups (non-blocking)

Recorded at Docker/run parity closeout; do not block remaining LKW.1 work.

| ID | Topic | Notes |
|----|-------|-------|
| LKW.1.6-F1 | Legacy application Dockerfiles | Pre-scaffold application Dockerfiles should receive build-time factory smoke. |
| LKW.1.6-F2 | `attestation_demo` agent COPY | `attestation_demo` `COPY agents/ ./agents/` must be documented as a demo exception or narrowed to the required agent subset. |
| LKW.1.6-F3 | `architecture_health_wiring` global catalog | Remains governance-only; must not become default product application startup wiring. |

Watchlist rule: do not refactor these components because they look wide. Refactor only when LKW exposes a measurable implementation, testing, or maintenance cost.

---

## 8. Remaining product waves (summary)

Full task breakdown: [`ARCHITECTURE.md`](ARCHITECTURE.md) §15.2.

| ID | Key deliverables |
|----|------------------|
| LKW.4 | `message_bus` background ingest queue |
| LKW.5 | `LKW_DATA_HOME` in settings, persistent vector store path under `data/` |
| LKW.6 | `scripts/lkw-host`, systemd/launchd/Windows unit, `wire_interaction_intake_service` |
| LKW.6b | Socket Mode → `/lkw` mapping; HITL notify |
| LKW.7 | `host/indexer_worker.py`, watcher + queue |
| LKW.8 | `clients/lkw-tray/` — HTTP-only client |

---

## 9. End-to-end validation scenarios

| ID | Scenario | Waves |
|----|----------|-------|
| E0 | Runtime hardening gate: strict policy, max_steps boundary, finalization diagnostics | LKW-H0 |
| E1 | Developer proof: fixture/local doc ingest → search cites source → synthesize writes shadow artifact | LKW.1, LKW-H1 |
| E1a | Live HTTP smoke: Docker host + `/health` + `/agents` + `/run` direct capabilities | LKW.1.7 |
| E1b | Live RAG diagnostic: accepted fixture path produces `used=true`, chunks, retrievable evidence, or exposes exact blocker reason | LKW.1.8 |
| E1c | Qdrant point-id compatibility: live ingest stores chunks with valid Qdrant point ids | LKW.1.9 |
| E1d | Tenant scope consistency: ingest/retrieve use compatible tenant/workspace/user scope | LKW.1.10 |
| E1e | Runtime gateway registry parity: configured application catalog tools are invokable by live runtime gateway | LKW.1.11 |
| E1f | Runtime event phase contract: planning does not emit schema-invalid `decision_emitted` | LKW.1.12 |
| E1g | Live index path: `local_indexer` invokes `rag.ingest_document` and writes chunks to Qdrant | LKW.1.13 |
| E1h | LKW.1 closeout: full live `index -> search -> synthesize -> shadow write` passes | LKW.1.14 |
| E2 | Search at desk via MCP | LKW.1 |
| E3 | Pipeline report | LKW.2 |
| E4 | Install → pick folders → persistent index | LKW.5, LKW.6, LKW.8 |
| E5 | Auto-index new file | LKW.7 |
| E6 | Slack search (optional) | LKW.6b |
| E7 | Generate a second product app from scaffold and verify it inherits improved env/build/deploy patterns | LKW-H3 / scaffold propagation |

---

## 10. Verification commands

```bash
# Runtime hardening checks touched by LKW-H0
uv run pytest -m gate -q

# Host + agents, when touching LKW code
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Dev run (backend only)
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020

# Docker proof path
docker compose -f applications/local_workspace_application/docker/docker-compose.yml up -d --build
curl -sS http://127.0.0.1:8020/health
curl -sS http://127.0.0.1:8020/v1/local_workspace/agents
```

Add narrower commands next to each implementation task once exact modules are known.

---

## 11. Per-agent plans

- [`agents/local_indexer/IMPLEMENTATION_PLAN.md`](../../agents/local_indexer/IMPLEMENTATION_PLAN.md)
- [`agents/local_search/IMPLEMENTATION_PLAN.md`](../../agents/local_search/IMPLEMENTATION_PLAN.md)
- [`agents/local_synthesizer/IMPLEMENTATION_PLAN.md`](../../agents/local_synthesizer/IMPLEMENTATION_PLAN.md)

---

## 12. Platform alignment

- Harness maintenance: platform **§6.1** only.
- LKW product: platform **`docs/plan/PLATFORM_FOUNDATION.md` §6.3a** — update when wave scope changes.
- LKW hardening decision record: [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md).
- LKW platform proof loop: [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md).
- One wave per PR unless operator batches explicitly.
- Every harness/platform/scaffold/build change in this track must cite the LKW acceptance criterion or platform propagation requirement that justifies it.
- Current LKW.1 findings: Qdrant point-id compatibility is fixed in LKW.1.9; tenant scope consistency is fixed in LKW.1.10; runtime gateway/application registry parity is fixed in LKW.1.11; event phase mismatch is fixed in LKW.1.12; live RAG ingest execution is fixed in LKW.1.13; full live product smoke remains LKW.1.14; live diagnostic visibility remains queued for LKW-H1.

---

## 13. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion or platform propagation requirement;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Do not stop platform propagation when LKW reveals a reusable pattern. In that case, update the platform/scaffold/deploy surface in the same iteration or record a blocking follow-up before moving to the next LKW wave.
