# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15 and [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md)  
**Do not diverge:** architecture decisions live in the architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** (T6) · **Active queue: LKW-H0 → LKW.1**

Platform register: [`docs/intergrax_runtime_architecture.md` §6.3a LKW.*](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated)

Principle: **local backend daemon** · **thin frontends** · **Slack optional** · **shadow writes only** · **LKW drives harness hardening**

---

## 0. Product boundary reminder

| | Backend (`lkw-host`) | Frontend (clients) |
|---|---------------------|-------------------|
| **Runs on** | localhost daemon | Tray / Cursor / Slack / curl |
| **Contains** | Nexus, agents, RAG, index, policy, trace | UI + HTTP calls only |
| **Must not** | — | RAG, LLM, direct file index, agent loops |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.

---

## 0a. LKW-driven hardening rule

Harness work in this plan is allowed only when it directly supports one of the following:

- LKW.1 acceptance criteria;
- LKW.2 graph pipeline acceptance criteria;
- local filesystem safety boundary;
- strict/product policy behavior;
- trace/evidence clarity for a real LKW run;
- first-run or adoption usability;
- a concrete implementation or testing bottleneck discovered while implementing LKW.

Do not start broad harness refactors from this plan. `NexusLoop` constructor width and `StepKernelContext` width are tracked as deferred watchlist items only.

Canonical decision record: [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md).

---

## 1. Wave queue

| ID | Title | Depends | Status | Priority |
|----|-------|---------|--------|----------|
| LKW.0 | Scaffold + architecture v2 | — | **Done** | — |
| LKW.3 | `filesystem.*` + allowlist | LKW.0 | **Done** | — |
| LKW-H0 | Minimal runtime hardening for product proof | LKW.0 | **Active** | Critical |
| LKW.1 | Domain UAEP: ingest + search + synthesize stub | LKW-H0 | **Next active** | Critical |
| LKW-H1 | LKW live trace/evidence inspection | LKW.1 | Planned | High |
| LKW.2 | Graph pipeline + `local.workspace.*` skills | LKW.1, LKW-H1 | Planned | High |
| LKW.4 | Background ingest queue (`message_bus`) | LKW.1 | Planned | Medium |
| LKW.5 | `LKW_DATA_HOME` + Chroma persistence | LKW.1 | Planned | High |
| LKW.6 | OS daemon + interaction intake router | LKW.1 | Planned | High |
| LKW.6b | Slack Socket Mode (optional) | LKW.6 | Planned | Medium |
| LKW.7 | File watcher + incremental index | LKW.4, LKW.5 | Planned | Medium |
| LKW.8 | Tray thin client | LKW.6 | Deferred | Low |
| LKW-H2 | Evidence/maturity wording cleanup | LKW.1 | Planned | Medium |
| LKW-H3 | Packaging/adoption simplification | LKW.1 or LKW.2 | Planned | Medium |
| LKW-W | Deferred architecture watchlist | LKW proof pain only | Deferred | Watch |

---

## 2. Active wave — LKW-H0: minimal runtime hardening for product proof

This is not a broad harness refactor wave. These tasks are allowed because they directly improve safety, bounded execution, and diagnosability for LKW.1.

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| LKW-H0.1 | Strict/product runtime must not silently default-allow when policy wiring is missing | runtime policy / kernel wiring | Tier-1 |
| LKW-H0.2 | Add `max_steps` boundary regression test | runtime kernel or ACP session tests | Tier-1 |
| LKW-H0.3 | Emit diagnostic/runtime event for post-finalization hook failure | Nexus lifecycle / runtime events | Tier-1 |

### Acceptance criteria

- [ ] Strict/product configuration fails closed or emits explicit configuration violation when required policy wiring is missing.
- [ ] Dev/test permissive policy behavior remains available only when explicitly selected and visible.
- [ ] Regression test proves whether `max_steps=N` permits exactly N steps and rejects step N+1.
- [ ] Finalization/lifecycle hook failure is visible in trace/diagnostics and is not silently swallowed.
- [ ] Existing gate and affected runtime tests remain green.

### Out of scope (LKW-H0)

- `NexusLoop` constructor refactor.
- `StepKernelContext` decomposition.
- Hosted observability product.
- Packaging split.
- New product features outside LKW safety and diagnosability.

---

## 3. Next active wave — LKW.1: Domain UAEP proof

### Goal

Deliver the first real LKW product proof:

```text
POST /v1/local_workspace/run
  -> local.workspace.index using metadata.source_paths
  -> rag.ingest_document
  -> local.workspace.search
  -> rag.retrieve with evidence
  -> local.workspace.synthesize
  -> workspace.write_file under shadow root
```

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/` `on_next_step` / cognitive pattern hooks | Tier-2 |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/` `on_next_step` / cognitive pattern hooks | Tier-2 |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/` `on_next_step` / cognitive pattern hooks | Tier-2 |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | `applications/.../tests/` or `tests/acceptance/` | Tier-3 |

### Acceptance criteria

- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` completes.
- [ ] Follow-up search returns answer referencing ingested content.
- [ ] Synthesize with `shadow_workspace: true` writes artifact under shadow root.
- [ ] Original user files are not modified.
- [ ] No Slack, tray, watcher, or OS service required.
- [ ] `uv run pytest` agent + host smoke green.

### Out of scope (LKW.1)

- Tray UI.
- Slack.
- File watcher.
- OS service installer.
- `local.workspace.*` skill bundle, except any minimal stub explicitly needed for LKW.1 tests.
- Broad harness refactor.

---

## 4. LKW-H1: live trace/evidence inspection for LKW runs

### Goal

Make one real LKW run inspectable without reading internal runtime code.

### Required inspection fields

For every LKW.1 proof run, the operator should be able to inspect:

- submitted task and capability;
- task id and run id;
- selected agent;
- step sequence;
- invoked tools and outcomes;
- policy decisions;
- RAG ingest/retrieve evidence;
- shadow workspace artifact path;
- terminal outcome;
- diagnostics from non-fatal lifecycle/finalization failures.

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| LKW-H1.1 | Define minimal LKW trace/evidence inspection contract | LKW host/debug docs or tests | Tier-3 |
| LKW-H1.2 | Ensure LKW run emits/records tool, policy, RAG, and shadow artifact evidence | runtime events + LKW host | Tier-1/3 |
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | `applications/local_workspace_application/...tests` | Tier-3 |

### Acceptance criteria

- [ ] A reviewer can see what happened in an LKW run from task submission to terminal result.
- [ ] Tool calls, policy decisions, RAG evidence, and shadow artifact path are visible.
- [ ] No hosted dashboard or external observability backend is required.

---

## 5. LKW.2: graph pipeline + local workspace skills

### Tasks

| ID | Task | Module | Owner |
|----|------|--------|-------|
| LKW.2.1 | Add `intergrax/skills/providers/local/` bundle | Tier-0 skills | Tier-0 |
| LKW.2.2 | Add `skill_ids` to local agent contracts | `agents/local_*` contracts | Tier-2 |
| LKW.2.3 | Enable `skill_bundles=["harness", "local"]` | `host/environment_profile.py` | Tier-3 |
| LKW.2.4 | Add graph/pipeline capability `local.workspace.pipeline` | manifest / graph spec | Tier-1/3 |

### Acceptance criteria

- [ ] Single `POST /v1/local_workspace/run` with `capability=local.workspace.pipeline` can run index → search → synthesize without manual capability selection.
- [ ] Tool access is resolved through `skill_ids`, not ad-hoc allowlists in agent code.
- [ ] Existing LKW.1 index/search/synthesize direct capabilities still pass.

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
| LKW-H3.1 | Define minimal developer first-run path for LKW | README / BUILD_AND_DEPLOY / LKW docs | New developer can run host, index fixture, search, and synthesize from documented commands |
| LKW-H3.2 | Decide optional dependency split | `pyproject.toml` / docs | Minimal install story is clear; heavy optional stacks are documented or split |

Potential packaging direction:

```text
intergrax-core
intergrax-lab
intergrax-lkw
intergrax-rag
intergrax-all
```

Do not start packaging before LKW.1 has a useful proof path.

---

## 7. Deferred architecture watchlist

These items are real architectural pressure points, but they must not block LKW.1.

| ID | Topic | Current decision | Trigger for action |
|----|-------|------------------|-------------------|
| LKW-W1 | `NexusLoop` constructor width | Accept as composition-root pressure | Refactor only if LKW requires repeated custom wiring, makes tests brittle, or forces duplicated bootstrap logic |
| LKW-W2 | `StepKernelContext` width | Accept as kernel execution-context pressure | Refactor only if unrelated concerns start changing together or test setup becomes excessive |

### Watchlist rule

Do not refactor these components because they look wide. Refactor only when LKW exposes a measurable implementation, testing, or maintenance cost.

---

## 8. Remaining product waves (summary)

Full task breakdown: [`ARCHITECTURE.md`](ARCHITECTURE.md) §15.2.

| ID | Key deliverables |
|----|------------------|
| LKW.4 | `message_bus` background ingest queue |
| LKW.5 | `LKW_DATA_HOME` in settings, Chroma path under `data/chroma/` |
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
| E2 | Search at desk via MCP | LKW.1 |
| E3 | Pipeline report | LKW.2 |
| E4 | Install → pick folders → persistent index | LKW.5, LKW.6, LKW.8 |
| E5 | Auto-index new file | LKW.7 |
| E6 | Slack search (optional) | LKW.6b |

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
```

Add narrower test commands next to the implementation PR once exact runtime test modules are known.

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
- One wave per PR unless operator batches explicitly.
- Every harness change in this track must cite the LKW acceptance criterion or watchlist trigger that justifies it.

---

## 13. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Continue only when the next change makes LKW safer, more observable, easier to run, or closer to a real user-facing proof.
