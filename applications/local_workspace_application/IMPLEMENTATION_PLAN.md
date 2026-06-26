# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in the architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** (T6) · **LKW.1.1–LKW.1.6 Closed** · **LKW.1.7 Partial/Open** · **LKW.1.8 Diagnosed** · **Active queue: LKW.1.9 → LKW.1.10 → LKW.1.11 → LKW-H1**

Latest live proof snapshot: **2026-06-26 — LKW.1.8 DIAGNOSED**. Docker, HTTP routing, agent listing, manual-evidence synthesize, shadow write, and source-file immutability work. Live RAG ingest/search is not proven yet. Root cause found: Qdrant rejects generated point ids such as `ingest-lkw-live-smoke-0` because point ids must be unsigned integers or UUIDs. Additional diagnosed follow-ups: raw tool reason/error is hidden from the HTTP run surface, and tenant scope consistency must be verified after the point-id fix.

Platform register: [`docs/intergrax_runtime_architecture.md` §6.3a LKW.*](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)

Principle: **local backend daemon** · **thin frontends** · **Slack optional** · **shadow writes only** · **LKW proves the platform**

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
- Cursor must **not** run repo-wide glob/search across all Python or Markdown files (for example `**/*.{py,md}`).
- Cursor must **stop** after the first hit grep/search once the implementation point is located — then implement immediately.
- Cursor must read **only cited document sections**, not full architecture/plan hubs or domain packs.
- Cursor may expand scope **only** when:
  1. a cited file imports a dependency that must change,
  2. a targeted test fails because of a cross-module contract,
  3. the implementation point does not exist in the given scope,
  4. [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) reveals a real need to change scaffold, env, Docker, or CI.
- **Expansion budget:** at most **3 files** outside **Read scope** per task (exception 1–4). After 3 reads or 3 failed greps → **STOP**, one short question, wait for operator OK. Do not read `uaep.py`, `boundary_demo`, or Nexus internals to discover `invoke_tool` when [`agents/lkw_shared/PATTERN.md`](../../agents/lkw_shared/PATTERN.md) or agent **Pattern anchor** is in scope.
- **Pattern anchor (LKW Tier-2):** canonical tool/allowlist pattern lives in [`agents/lkw_shared/PATTERN.md`](../../agents/lkw_shared/PATTERN.md). Each LKW agent implements in `agents/<agent>/steps/<job>.py` (see agent `ARCHITECTURE.md` § Pattern anchor). Prompts **must** cite the step file as **Implementation point**.
- **Default tests:** new or changed test file + one narrow smoke — not a full Nexus/runtime/agent suite unless the prompt requires it.
- **Default report:** changed files, tests run, pass/fail, commit SHA, platform propagation yes/no (+ brief reason).
- **Full report** only when the operator explicitly asks (`pełny raport`, `full report`, `iteration summary`).

See also: [`docs/features/plan/TOKEN_OPTIMIZATION.md`](../../docs/features/plan/TOKEN_OPTIMIZATION.md) for cross-repo token policy.

### Prompt template (LKW tasks)

Copy this skeleton for every LKW implementation prompt in Cursor. Fill placeholders; do not omit **Token guardrails**.

```text
Repo: `jakbuczarnecki/intergrax`, branch `development`.

Goal:
<one sentence — task ID + outcome>

Read scope:
- `<path>` — section `<id>` only
- `<path>` — `<function or line range if known>`
- `agents/lkw_shared/PATTERN.md` — when task invokes catalog tools (LKW Tier-2)
- existing tests: `<path or glob under one module>`

Implementation point:
- `<agents/<agent>/steps/<job>.py>` — edit here; do not search runtime for invoke_tool pattern

Code search scope:
Search only:
- `<pattern>`
- `<pattern>`
Paths: `<tier/module glob>` — not `**/*.{py,md}`

Stop condition:
Stop reading/searching once `<implementation point>` is located; implement immediately.

Do not touch:
- <explicit out-of-scope items>

Test scope:
- new/changed test: `<path>`
- narrow smoke: `<path>` — not full suite unless this prompt requires it

Report format:
Terse: changed files, tests run, pass/fail, commit SHA, platform propagation yes/no + brief reason.
Full report only if operator asks.

Token guardrails:
- No whole-repo audit unless this prompt explicitly requires it.
- No repo-wide Python/Markdown glob search.
- Stop after first grep hit that locates the **Implementation point** (usually `steps/<job>.py`).
- Read only cited document sections; do not load full architecture/plan hubs.
- Expand scope only for the four exceptions in IMPLEMENTATION_PLAN §0b; **max 3 files** outside Read scope.
- Do not read runtime/Nexus sources for tool invocation when `agents/lkw_shared/PATTERN.md` is cited.
- Default tests: new/changed test + one narrow smoke.
- Default report: terse (see Report format). Full report only on operator request.

Acceptance:
- <task-specific acceptance bullets>

Na końcu podaj:
- zmienione pliki;
- uruchomione testy;
- wynik testów;
- commit SHA;
- czy trzeba było aktualizować scaffold/env/Docker/CI, a jeśli nie — dlaczego.
```

Every task in §2 and later waves should use this template unless the operator supplies an equivalent scoped prompt.

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
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW-H0 | **Active — LKW.1.8 diagnosed, LKW.1.9 next** | Critical |
| LKW-H1 | LKW live trace/evidence inspection | LKW.1.9/LKW.1.10/LKW.1.11 | **Queued after live RAG blockers** | High |
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

### Tasks

| ID | Task | Module | Status | Platform propagation |
|----|------|--------|--------|----------------------|
| LKW-H0.1 | Strict/product runtime must not silently default-allow when policy wiring is missing | runtime policy / kernel wiring | Closed / monitor | Update shared config/scaffold guidance if unsafe defaults are generic |
| LKW-H0.2 | Add `max_steps` boundary regression test | runtime kernel or ACP session tests | Closed / monitor | Update generated guidance only if step-limit semantics are exposed to app/agent authors |
| LKW-H0.3 | Emit diagnostic/runtime event for post-finalization hook failure | Nexus lifecycle / runtime events | Closed / monitor | Propagate generic diagnostic/event pattern to runtime docs/templates if applicable |

### Out of scope (LKW-H0)

- `NexusLoop` constructor refactor.
- `StepKernelContext` decomposition.
- Hosted observability product.
- Full packaging split.
- New product features outside LKW safety and diagnosability.

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
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/` `on_next_step` / cognitive pattern hooks | **Closed** | Update agent scaffold/docs if this becomes the canonical tool-invocation pattern |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/` `on_next_step` / cognitive pattern hooks | **Closed** | Update evidence/result patterns if reusable by generated agents |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/` `on_next_step` / cognitive pattern hooks | **Closed** | Update scaffold guidance for shadow-write outputs if generic |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | `applications/.../tests/` or `tests/acceptance/` | **Closed** | Add scaffold/test template if this becomes the canonical app acceptance pattern |
| LKW.1.5 | Env/settings parity check | `.env.example`, `host/settings.py`, docs | **Closed / configured** | Ensure app settings pattern can inform scaffolded app settings |
| LKW.1.6 | Docker/run parity | Dockerfile, compose, build/run docs | **Closed** | Closed after Docker build/start parity, environment-scoped capability graph, MCP opt-in startup, isolated Docker agent closure, and LKW Docker build smoke. |
| LKW.1.7 | Live `local_workspace_application` HTTP smoke | Docker compose + `/health` + `/agents` + `/run` index/search/synthesize | **Partial / open** | Product host works, but full RAG-backed flow is blocked until LKW.1.9/LKW.1.10 are completed. |
| LKW.1.8 | Diagnose live RAG ingest failure | LKW Docker logs/runtime output + `rag.ingest_document`/retrieve path | **Diagnosed** | Platform-reusable: Qdrant point-id contract, hidden raw tool reason, and tenant scope consistency are now queued as explicit tasks. |
| LKW.1.9 | Fix Qdrant-compatible RAG ingest point ids | `intergrax/integrations/providers/vector_store/qdrant/rag_store.py` or `intergrax/rag/ingest/ingest_pipeline.py` + focused regression tests | **Next** | Platform-reusable: future Qdrant-backed Intergrax apps must not emit invalid point ids. |
| LKW.1.10 | Verify and fix tenant scope consistency for live RAG ingest/retrieve | Runtime request metadata, RAG ingest/retrieve scope, Qdrant tenant enforcement | **Queued after LKW.1.9** | Platform-reusable: tenant/workspace/user scope must be consistent for generated RAG applications. |
| LKW.1.11 | Re-run live HTTP smoke and close LKW.1.7 if full flow passes | Docker compose + `/run` index/search/synthesize + shadow verification | **Queued after LKW.1.9/LKW.1.10** | Product closeout plus platform proof checklist before moving to H1/LKW.2. |

### LKW.1.7 live smoke result — 2026-06-26

Status: **PARTIAL / OPEN**.

Observed live stack:

| Area | Result |
|------|--------|
| Repository | `development`, latest commit at smoke time: `e77a4fc9` |
| Docker | `local_workspace` healthy, `qdrant` running, `ollama` running |
| `/health` | `{"status":"ok"}` |
| `/v1/local_workspace/agents` | 3 LKW agents visible: index, search, synthesize |
| Index smoke | HTTP completed; `accepted=1`, `rejected=0`, `ingested=0`, `chunks=0` |
| Search smoke | HTTP completed; `local_search: search failed — retrieve_failed` |
| Synthesize smoke | Completed with manual evidence |
| Shadow write | `lkw-live-smoke-draft.md` written under `/data/shadow_workspaces/...` |
| Source file immutability | Original fixture unchanged; only untracked smoke fixture existed locally |
| Files changed by diagnostic | None |
| Commits created by diagnostic | None |

Conclusion:

- Docker/run parity is not the blocker anymore.
- HTTP route, agent registration/routing, manual-evidence synthesize, shadow write, and source immutability work.
- Full `index → search → synthesize` is **not proven** because live RAG ingest produces no chunks.
- Search failure is secondary until ingest produces retrievable chunks.

Primary failure category at LKW.1.7:

```text
RAG ingest / retrieve wiring
```

### LKW.1.8 diagnostic result — 2026-06-26

Status: **DIAGNOSED**.

Primary diagnosed blocker:

```text
Qdrant write failure: generated point id `ingest-lkw-live-smoke-0` is not a valid Qdrant point id. Qdrant accepts only unsigned integers or UUIDs.
```

Observed details:

- Runtime harness wiring is present enough for `rag.ingest_document` to reach the RAG pipeline.
- Vectorstore manager and embedding manager are configured.
- Fixture source file is visible and readable inside the container.
- Configured Ollama model `llama3.1:latest` is present.
- Loader, splitter, and embedding path run before the Qdrant upsert failure.
- Qdrant logs show no successful write for the failing ingest.
- `index_job` reports `accepted=1` but `ingested=0` because the tool response fails before successful upsert.

Diagnosed findings queued as tasks:

| Finding | Classification | Queued task |
|---------|----------------|-------------|
| Qdrant rejects generated string point ids | `Platform-reusable` | LKW.1.9 |
| HTTP `/run` hides raw tool reason/status/error | `Platform-reusable` | LKW-H1.1 / LKW-H1.2 |
| Tenant metadata can mismatch Qdrant store tenant (`expected 'default', got 'lkw-smoke'`) | `Platform-reusable` | LKW.1.10 |
| Need final live proof after fixes | Product + platform closeout | LKW.1.11 |

### LKW.1.9 implementation goal

Fix Qdrant-compatible point ids so live RAG ingest can store chunks in Qdrant.

Acceptance:

- [ ] Qdrant-backed ingest no longer sends invalid point ids such as `ingest-lkw-live-smoke-0`.
- [ ] Original/stable logical chunk id remains recoverable in metadata if provider id normalization is used.
- [ ] Focused regression test proves Qdrant provider receives valid point ids or normalizes invalid ids before upsert.
- [ ] LKW live index smoke produces `ingested>0` and `chunks>0`, unless another queued blocker is reached.
- [ ] Finding is reported as `Platform-reusable`, with any scaffold/RAG provider follow-up recorded.

Out of scope:

- Tenant scope refactor.
- HTTP diagnostic surface.
- Grafana/Tempo/OpenTelemetry Collector.
- Full LKW-H1 implementation.

### LKW.1.10 diagnostic/implementation goal

Verify and, if needed, fix tenant/workspace/user scope consistency between request body, request metadata, RAG ingest, RAG retrieve, and Qdrant tenant enforcement.

Acceptance:

- [ ] `tenant_id` source of truth is explicit for LKW live runs.
- [ ] Ingest and retrieve use compatible tenant/workspace/user filters.
- [ ] Qdrant tenant enforcement does not reject valid LKW smoke requests.
- [ ] Focused regression test covers tenant mismatch or confirms intended behavior.
- [ ] Any scaffold/env/runbook implication is classified and recorded.

Out of scope:

- Qdrant point-id compatibility, unless LKW.1.9 was not completed.
- Hosted observability stack.
- Graph pipeline / LKW.2.

### LKW.1.11 closeout goal

Re-run the live LKW HTTP proof after LKW.1.9 and LKW.1.10.

Acceptance:

- [ ] Docker stack healthy.
- [ ] `/health` succeeds.
- [ ] `/v1/local_workspace/agents` lists index/search/synthesize.
- [ ] `local.workspace.index` ingests fixture with `ingested>0` and `chunks>0`.
- [ ] `local.workspace.search` retrieves evidence referencing the fixture.
- [ ] `local.workspace.synthesize` writes only to shadow workspace.
- [ ] Original fixture remains unchanged.
- [ ] Product and platform acceptance criteria are updated before LKW.1 closeout.

### Product acceptance criteria

- [x] Docker stack can run the LKW application host.
- [x] `/health` responds successfully.
- [x] `/v1/local_workspace/agents` lists `local.workspace.index`, `local.workspace.search`, and `local.workspace.synthesize`.
- [x] `POST /v1/local_workspace/run` reaches the index agent.
- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` ingests at least one chunk from the fixture.
- [ ] Follow-up search returns answer/evidence referencing ingested content.
- [x] Synthesize with `shadow_workspace: true` writes artifact under shadow root when evidence is supplied.
- [x] Original user files are not modified.
- [x] No Slack, tray, watcher, or OS service required.
- [ ] `uv run pytest` agent + host smoke green for final closeout.

### Platform acceptance criteria

- [ ] Platform proof checklist in §0a is completed.
- [x] Every discovered defect/pattern/gap from LKW.1.8 is classified as `Platform-reusable` and queued.
- [x] Reusable Docker/build/run lessons are reflected in Docker templates/docs or recorded as follow-ups.
- [ ] Reusable Qdrant/RAG provider id handling is fixed or recorded as a blocking follow-up.
- [ ] Reusable tenant/workspace/user scope handling is fixed or recorded as a blocking follow-up.
- [ ] Reusable env/settings lessons are reflected in shared settings/scaffold/docs or recorded as a blocking follow-up.
- [ ] Reusable agent/application patterns are reflected in scaffold templates/docs or recorded as a blocking follow-up.
- [ ] Any dependency/profile lesson is reflected in `pyproject.toml` or recorded as a blocking follow-up.
- [ ] Live diagnostic/inspection lessons are reflected in LKW-H1 or recorded as a blocking follow-up.

### Observability decision for LKW.1

Do **not** add Grafana, Tempo, OpenTelemetry Collector, or a hosted/external observability backend as a prerequisite for LKW.1 closeout.

Reason:

- LKW.1 must first prove the local product path with minimum moving parts.
- LKW-H1 explicitly covers local trace/evidence inspection without requiring an external dashboard.
- Full observability stack can be introduced later when the product proof path is useful and stable enough to justify operational complexity.

However, the LKW.1.7/LKW.1.8 results show that minimal local diagnosability is required before the RAG blocker can be resolved. At minimum, the operator must be able to inspect:

- selected agent;
- step id;
- invoked tool id;
- tool input summary;
- raw tool status;
- raw tool `reason`/error;
- RAG ingest/retrieve summary;
- shadow artifact path.

This requirement feeds directly into LKW-H1.

### Out of scope (LKW.1)

- Tray UI.
- Slack.
- File watcher.
- OS service installer.
- Full `local.workspace.*` skill bundle, except any minimal stub explicitly needed for LKW.1 tests.
- Hosted observability dashboard.
- Grafana/Tempo/OpenTelemetry Collector as a blocker for live proof.
- Broad harness refactor unrelated to LKW acceptance or platform propagation.

---

## 4. LKW-H1: live trace/evidence inspection for LKW runs

### Goal

Make one real LKW run inspectable without reading internal runtime code, and ensure the trace/evidence pattern is reusable by future applications.

LKW-H1 is **not** the hosted observability stack. It is the minimum local inspection surface needed for a developer/operator to understand a run. Grafana, Tempo, and an OpenTelemetry Collector remain optional future operational infrastructure unless a later task explicitly scopes them.

### Known diagnosed input from LKW.1.8

`LKW.1.8` proved that the platform can hide the exact raw tool failure from the HTTP run response. The operator had to inspect logs/runtime behavior to discover the Qdrant point-id error. This is `Platform-reusable` because every future application needs a minimal way to see tool status and reason/error during local proof runs.

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
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | `applications/local_workspace_application/...tests` | Planned | Update generated app test pattern if reusable |

### Acceptance criteria

- [ ] A reviewer can see what happened in an LKW run from task submission to terminal result.
- [ ] Tool calls, policy decisions, RAG evidence, raw tool reason/error, and shadow artifact path are visible.
- [ ] No hosted dashboard or external observability backend is required.
- [ ] Platform proof checklist in §0a is completed.

---

## 5. LKW.2: graph pipeline + local workspace skills

### Tasks

| ID | Task | Module | Owner | Platform propagation |
|----|------|--------|-------|----------------------|
| LKW.2.1 | Add `intergrax/skills/providers/local/` bundle | Tier-0 skills | Tier-0 | Update skill scaffold/catalog docs if pattern is reusable |
| LKW.2.2 | Add `skill_ids` to local agent contracts | `agents/local_*` contracts | Tier-2 | Update agent scaffold to generate correct `skill_ids` pattern if needed |
| LKW.2.3 | Enable `skill_bundles=["harness", "local"]` | `host/environment_profile.py` | Tier-3 | Update app scaffold/environment templates if bundle pattern is generic |
| LKW.2.4 | Add graph/pipeline capability `local.workspace.pipeline` | manifest / graph spec | Tier-1/3 | Update app scaffold or graph docs if this becomes canonical multi-agent pipeline pattern |

### Acceptance criteria

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

Allowed claim vocabulary:

| Claim type | Allowed wording |
|------------|-----------------|
| Architecture maturity | Strong architecture baseline / high architectural maturity |
| Harness baseline | Core harness proof path available |
| Live product proof | In progress through LKW |
| Production-proven claim | Not claimed until live product, provider, deployment, security, and adoption evidence exist |

### LKW-H3 — packaging/adoption simplification

| ID | Task | Module | Acceptance |
|----|------|--------|------------|
| LKW-H3.1 | Define minimal developer first-run path for LKW and scaffolded apps | README / BUILD_AND_DEPLOY / LKW docs / scaffold docs | New developer can run host, index fixture, search, and synthesize from documented commands |
| LKW-H3.2 | Decide optional dependency split | `pyproject.toml` / docs | Minimal install story is clear; heavy optional stacks are documented or split |
| LKW-H3.3 | Propagate adoption lessons to application scaffold | `intergrax/scaffold/` | Next generated product application inherits the improved env/build/deploy documentation pattern |

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
| LKW.1.6-F1 | Legacy application Dockerfiles | Pre-scaffold application Dockerfiles should receive build-time factory smoke (same closure pattern as generated apps). |
| LKW.1.6-F2 | `attestation_demo` agent COPY | `attestation_demo` `COPY agents/ ./agents/` must be documented as a demo exception or narrowed to the required agent subset. |
| LKW.1.6-F3 | `architecture_health_wiring` global catalog | Remains governance-only; must not become default product application startup wiring. |

### Watchlist rule

Do not refactor these components because they look wide. Refactor only when LKW exposes a measurable implementation, testing, or maintenance cost.

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
| E1e | LKW.1 closeout: full live `index → search → synthesize → shadow write` passes | LKW.1.11 |
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

# Host + agents (every PR touching LKW)
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Dev run (backend only)
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020

# Docker proof path once stable enough for CI
applications/local_workspace_application/docker/build-docker.sh
docker run --rm --env-file applications/local_workspace_application/.env -p 8020:8020 local_workspace-application

# Current LKW.1.7+ Docker compose path
docker compose -f applications/local_workspace_application/docker/docker-compose.yml up -d --build
curl -sS http://127.0.0.1:8020/health
curl -sS http://127.0.0.1:8020/v1/local_workspace/agents
```

Add narrower test commands next to the implementation PR once exact runtime/scaffold/Docker test modules are known.

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
- Current LKW.1.8 findings: Qdrant point-id compatibility, tenant scope consistency, and live diagnostic visibility are platform pressure points; full hosted observability is not a prerequisite for LKW.1 closeout.

---

## 13. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion or platform propagation requirement;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Do not stop platform propagation when LKW reveals a reusable pattern. In that case, update the platform/scaffold/deploy surface in the same iteration or record a blocking follow-up before moving to the next LKW wave.
