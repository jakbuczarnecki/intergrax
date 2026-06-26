# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in the architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** (T6) · **LKW.1.1–LKW.1.6 Closed** · **LKW.1.7 Partial/Open** · **LKW.1.8 Diagnosed** · **LKW.1.9 Completed** · **LKW.1.10 Completed / live HTTP blocked by LKW.1.11** · **Active queue: LKW.1.11 → LKW.1.12 → LKW-H1**

Latest live proof snapshot: **2026-06-26 — LKW.1.10 COMPLETED IN SCOPE / LIVE HTTP BLOCKED BY LKW.1.11**. Docker, HTTP routing, agent listing, manual-evidence synthesize, shadow write, and source-file immutability work. The Qdrant point-id blocker from LKW.1.8 is fixed. Tenant scope consistency from LKW.1.10 is fixed: `RuntimeRequest.tenant_id` / HTTP body tenant is authoritative, `metadata.tenant_id` cannot conflict, and `collection_id` remains `workspace_id`. Focused tenant/RAG/Qdrant/LKW tests passed (`21 passed`), Qdrant tenant isolation still passes, and direct `perform_rag_ingest` with `tenant_id=lkw-smoke` produced `used=True`, `num_chunks=1`. Live HTTP still returns `ingested=0`, `chunks=0` because `RuntimeToolGateway` uses an `IdempotentToolInvoker` registry that does not contain `rag.ingest_document` even though `ApplicationToolWiring.registry` does. This produces `unknown_capability_tool:rag.ingest_document` and `total_tool_calls=0`. This is a **Platform-reusable execution wiring blocker**, not an observability-only H1 issue. It must be fixed before LKW.1 closeout.

Platform register: [`docs/intergrax_runtime_architecture.md` §6.3a LKW.*](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)

Principle: **local backend daemon** · **thin frontends** · **Slack optional** · **shadow writes only** · **LKW proves the platform**

---

## 0. Product boundary reminder

| | Backend (`lkw-host`) | Frontend (clients) |
|---|---------------------|-------------------|
| **Runs on** | localhost daemon | Tray / Cursor / Slack / curl |
| **Contains** | Nexus, agents, RAG, index, policy, trace | UI + HTTP calls only |
| **Must not** | — | RAG, LLM, direct file index, agent loops |

See [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) §4.

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
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW-H0 | **Active — LKW.1.10 completed, LKW.1.11 next** | Critical |
| LKW-H1 | LKW live trace/evidence inspection | LKW.1.11/LKW.1.12 | **Queued after live execution blockers** | High |
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
| LKW.1.7 | Live `local_workspace_application` HTTP smoke | Docker compose + `/health` + `/agents` + `/run` index/search/synthesize | **Partial / open** | Product host works, but full RAG-backed flow is blocked until LKW.1.11 is completed. |
| LKW.1.8 | Diagnose live RAG ingest failure | LKW Docker logs/runtime output + `rag.ingest_document`/retrieve path | **Diagnosed** | Platform-reusable: Qdrant point-id contract, hidden raw tool reason, tenant scope, and runtime tool registry parity were queued as explicit tasks. |
| LKW.1.9 | Fix Qdrant-compatible RAG ingest point ids | `intergrax/integrations/providers/vector_store/qdrant/rag_store.py` + `tests/unit/integrations/providers/vector_store/test_qdrant_point_id_normalization.py` | **Completed** | Platform-reusable fix committed in `855737a6`; Qdrant point ids are normalized and `logical_id` is preserved. |
| LKW.1.10 | Fix tenant scope consistency for live RAG ingest/retrieve | `agents/lkw_shared/runtime_helpers.py`, `intergrax/tools/providers/rag/scope.py`, local index/search steps, RAG services/tests | **Completed / live HTTP blocked by LKW.1.11** | Platform-reusable: tenant source of truth is explicit; direct RAG ingest works with `tenant_id=lkw-smoke`; live path now exposes runtime gateway registry mismatch. |
| LKW.1.11 | Fix runtime tool gateway registry parity for catalog tools | `RuntimeToolGateway`, `IdempotentToolInvoker`, application tool wiring / catalog bootstrap path | **Next** | Platform-reusable: tools declared by application wiring must be invokable by the live runtime gateway. |
| LKW.1.12 | Re-run live HTTP smoke and close LKW.1.7 if full flow passes | Docker compose + `/run` index/search/synthesize + shadow verification | **Queued after LKW.1.11** | Product closeout plus platform proof checklist before moving to H1/LKW.2. |

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
- Full `index → search → synthesize` is **not proven** because the live runtime path still cannot execute `rag.ingest_document` through the same registry advertised by application wiring.
- Search failure is secondary until ingest produces retrievable chunks.

Primary failure category at LKW.1.7 after LKW.1.10:

```text
Runtime tool gateway / catalog registry parity
```

### LKW.1.8 diagnostic result — 2026-06-26

Status: **DIAGNOSED**.

Primary diagnosed blocker:

```text
Qdrant write failure: generated point id `ingest-lkw-live-smoke-0` is not a valid Qdrant point id. Qdrant accepts only unsigned integers or UUIDs.
```

Observed details:

- Runtime harness wiring is present enough for `rag.ingest_document` to reach the RAG pipeline in direct execution.
- Vectorstore manager and embedding manager are configured.
- Fixture source file is visible and readable inside the container.
- Configured Ollama model `llama3.1:latest` is present.
- Loader, splitter, and embedding path run before the Qdrant upsert failure.
- Qdrant logs show no successful write for the failing ingest.
- `index_job` reports `accepted=1` but `ingested=0` because the tool response fails before successful upsert.

Diagnosed findings queued as tasks:

| Finding | Classification | Queued task |
|---------|----------------|-------------|
| Qdrant rejects generated string point ids | `Platform-reusable` | LKW.1.9 — completed in `855737a6` |
| Tenant metadata can mismatch Qdrant store tenant (`expected 'default', got 'lkw-smoke'`) | `Platform-reusable` | LKW.1.10 — completed in scope |
| Runtime gateway registry does not expose `rag.ingest_document` although application wiring has it | `Platform-reusable` | LKW.1.11 |
| HTTP `/run` hides raw tool reason/status/error | `Platform-reusable` | LKW-H1.1 / LKW-H1.2 |
| Need final live proof after execution blockers | Product + platform closeout | LKW.1.12 |

### LKW.1.9 implementation result — 2026-06-26

Status: **COMPLETED IN SCOPE**.

Commit:

```text
855737a6
```

Changed files:

- `intergrax/integrations/providers/vector_store/qdrant/rag_store.py`
- `tests/unit/integrations/providers/vector_store/test_qdrant_point_id_normalization.py`

Result:

- Qdrant provider now normalizes invalid logical point ids before upsert.
- `logical_id` preserves the original logical chunk id in payload metadata.
- Qdrant upsert succeeds for the previously failing ingest path.
- Direct container ingest with `tenant_id=default` returns `used=True`, `chunks=1`, and `vector_ids=['ingest-lkw-live-smoke-0']`.
- Live HTTP smoke then reached the tenant mismatch blocker, which was handled in LKW.1.10.

Tests:

```text
uv run pytest tests/unit/integrations/providers/vector_store/test_qdrant_point_id_normalization.py -q
→ 7 passed

uv run pytest tests/unit/integrations/providers/vector_store/test_qdrant_chroma.py \
  tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py::test_vectorstore_tenant_isolation_contract[qdrant] -q
→ 11 passed
```

Acceptance:

- [x] Qdrant-backed ingest no longer sends invalid point ids such as `ingest-lkw-live-smoke-0`.
- [x] Original/stable logical chunk id remains recoverable in metadata as `logical_id`.
- [x] Focused regression test proves Qdrant provider normalizes invalid ids before upsert.
- [x] LKW live index smoke reaches the next queued blocker instead of the Qdrant point-id failure.
- [x] Finding is reported as `Platform-reusable`.
- [x] No tenant scope fix included.
- [x] No HTTP diagnostic/H1 fix included.

Out of scope completed as expected:

- Tenant scope refactor moved to LKW.1.10.
- HTTP diagnostic surface remains in LKW-H1.
- Grafana/Tempo/OpenTelemetry Collector not introduced.
- Full LKW-H1 not started.

### LKW.1.10 implementation result — 2026-06-26

Status: **COMPLETED IN SCOPE / LIVE HTTP BLOCKED BY LKW.1.11**.

Changed files reported by implementation:

- `agents/lkw_shared/runtime_helpers.py` — `resolve_request_scope()`
- `agents/local_indexer/steps/index_job.py`
- `agents/local_search/steps/search_job.py`
- `intergrax/tools/providers/rag/scope.py`
- `intergrax/tools/providers/rag/ingest_service.py`
- `intergrax/tools/providers/rag/service.py`
- `tests/unit/tools/providers/rag/test_rag_scope.py`
- `agents/local_indexer/tests/test_index_job.py`
- `agents/local_search/tests/test_search_job.py`
- `applications/local_workspace_application/local_workspace_application_tests/test_lkw_acceptance_index_search_synthesize.py`

Root cause fixed:

- LKW previously allowed `metadata.tenant_id` to drive or conflict with RAG tenant scope.
- Qdrant store could be configured as `default` while RAG metadata carried `lkw-smoke`, causing tenant enforcement to reject ingest.
- `RuntimeRequest.tenant_id` / HTTP body tenant is now the authoritative tenant source.
- `metadata.tenant_id` must not conflict with the authoritative tenant.
- `collection_id` maps to `workspace_id` and remains a separate dimension.

Verification:

```text
uv run pytest tests/unit/tools/providers/rag/test_rag_scope.py \
  tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py::test_vectorstore_tenant_isolation_contract[qdrant] \
  agents/local_indexer/tests/test_index_job.py \
  agents/local_search/tests/test_search_job.py \
  applications/local_workspace_application/local_workspace_application_tests/test_lkw_acceptance_index_search_synthesize.py -q
→ 21 passed
```

Additional verification:

```text
direct perform_rag_ingest with tenant_id=lkw-smoke → used=True, num_chunks=1
```

Live HTTP result after tenant fix:

```text
accepted=1
rejected=0
ingested=0
chunks=0
blocker=unknown_capability_tool:rag.ingest_document
```

Interpretation:

- Tenant scope is no longer the blocker.
- The live path does not reach RAG because `RuntimeToolGateway` dispatches through an `IdempotentToolInvoker` registry that does not contain `rag.ingest_document`.
- `ApplicationToolWiring.registry` contains the tool, so this is a runtime gateway / application wiring registry parity bug.
- `total_tool_calls=0` is a symptom of dispatch failure, not proof that no tool was intended.

Acceptance:

- [x] `tenant_id` source of truth is explicit.
- [x] `metadata.tenant_id` cannot conflict with authoritative request tenant.
- [x] Ingest and retrieve use compatible tenant/workspace/user scope in direct/unit paths.
- [x] Qdrant tenant enforcement no longer rejects valid direct LKW/RAG ingest.
- [x] Focused regression tests pass.
- [x] Qdrant tenant isolation test still passes.
- [x] Live index smoke reaches the next queued blocker instead of tenant mismatch.
- [x] No Qdrant point-id changes included.
- [x] No HTTP diagnostic/H1 changes included.

### LKW.1.11 implementation goal

Fix live runtime tool gateway / catalog registry parity so tools declared by application wiring are invokable by the runtime gateway.

Known blocker from LKW.1.10 live HTTP smoke:

```text
unknown_capability_tool:rag.ingest_document
```

Root question:

```text
Why does ApplicationToolWiring.registry contain rag.ingest_document while the RuntimeToolGateway / IdempotentToolInvoker registry used by live /run does not?
```

Acceptance:

- [ ] Live `RuntimeToolGateway` and `IdempotentToolInvoker` use the same effective tool registry/catalog as `ApplicationToolWiring.registry` for configured application tools.
- [ ] `rag.ingest_document` is invokable through the live HTTP `/run` index path.
- [ ] `unknown_capability_tool:rag.ingest_document` no longer occurs for configured LKW tools.
- [ ] `total_tool_calls` is non-zero when a configured catalog tool is invoked.
- [ ] Focused regression test covers application wiring registry → runtime gateway invoker parity.
- [ ] Fix is classified as `Platform-reusable` and not LKW-only.
- [ ] Any scaffold/application-host implication is updated or recorded as a blocking follow-up.

Out of scope:

- Tenant scope refactor; completed in LKW.1.10.
- Qdrant point-id compatibility; completed in LKW.1.9.
- HTTP diagnostic surface / H1, except minimal assertions needed to prove execution.
- Hosted observability stack.
- Graph pipeline / LKW.2.

### LKW.1.12 closeout goal

Re-run the live LKW HTTP proof after LKW.1.11.

Acceptance:

- [ ] Docker stack healthy.
- [ ] `/health` succeeds.
- [ ] `/v1/local_workspace/agents` lists index/search/synthesize.
- [ ] `local.workspace.index` invokes `rag.ingest_document` through the live runtime gateway.
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
- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` invokes `rag.ingest_document` through the live runtime gateway.
- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` ingests at least one chunk from the fixture.
- [ ] Follow-up search returns answer/evidence referencing ingested content.
- [x] Synthesize with `shadow_workspace: true` writes artifact under shadow root when evidence is supplied.
- [x] Original user files are not modified.
- [x] No Slack, tray, watcher, or OS service required.
- [ ] `uv run pytest` agent + host smoke green for final closeout.

### Platform acceptance criteria

- [ ] Platform proof checklist in §0a is completed.
- [x] Every discovered defect/pattern/gap from LKW.1.8–LKW.1.10 is classified as `Platform-reusable` and queued or completed.
- [x] Reusable Docker/build/run lessons are reflected in Docker templates/docs or recorded as follow-ups.
- [x] Reusable Qdrant/RAG provider id handling is fixed in LKW.1.9.
- [x] Reusable tenant/workspace/user scope handling is fixed in LKW.1.10.
- [ ] Reusable runtime gateway / application tool registry parity is fixed in LKW.1.11.
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

However, the LKW.1.7–LKW.1.10 results show that minimal local diagnosability is required after execution blockers are fixed. At minimum, the operator must be able to inspect:

- selected agent;
- step id;
- invoked tool id;
- tool input summary;
- raw tool status;
- raw tool `reason`/error;
- RAG ingest/retrieve summary;
- shadow artifact path.

This requirement feeds directly into LKW-H1, but it must not be used to bury execution blockers such as `unknown_capability_tool`.

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

### Known diagnosed input from LKW.1.8–LKW.1.10

`LKW.1.8`, `LKW.1.9`, and `LKW.1.10` proved that the platform can hide exact raw tool failures from the HTTP run response. The operator had to inspect logs/runtime behavior to discover the Qdrant point-id error, the tenant mismatch, and the runtime gateway registry mismatch. This is `Platform-reusable` because every future application needs a minimal way to see tool status and reason/error during local proof runs. H1 must improve visibility, but it must not replace the LKW.1.11 execution wiring fix.

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

Full task breakdown: [`ARCHITECTURE.md`](docs/ARCHITECTURE.md) §15.2.

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
| E1f | LKW.1 closeout: full live `index → search → synthesize → shadow write` passes | LKW.1.12 |
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
- Current LKW.1 findings: Qdrant point-id compatibility is fixed in LKW.1.9; tenant scope consistency is fixed in LKW.1.10; runtime gateway / application tool registry parity is the next platform execution blocker in LKW.1.11; live diagnostic visibility remains queued for LKW-H1; full hosted observability is not a prerequisite for LKW.1 closeout.

---

## 13. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion or platform propagation requirement;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Do not stop platform propagation when LKW reveals a reusable pattern. In that case, update the platform/scaffold/deploy surface in the same iteration or record a blocking follow-up before moving to the next LKW wave.
