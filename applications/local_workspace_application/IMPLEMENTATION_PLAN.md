# Local Knowledge Workspace (LKW) — Implementation Plan

**Derived from:** [`ARCHITECTURE.md`](ARCHITECTURE.md) §15, [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md), and [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md)  
**Do not diverge:** architecture decisions live in the architecture documents; this file schedules implementation waves only.

Status: **LKW.0 Done** · **LKW.3 Done** (T6) · **Active queue: LKW-H0 → LKW.1**

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

Every task in this wave must also run the platform proof checklist in §0a.

### Tasks

| ID | Task | Module | Owner | Platform propagation |
|----|------|--------|-------|----------------------|
| LKW-H0.1 | Strict/product runtime must not silently default-allow when policy wiring is missing | runtime policy / kernel wiring | Tier-1 | Update shared config/scaffold guidance if unsafe defaults are generic |
| LKW-H0.2 | Add `max_steps` boundary regression test | runtime kernel or ACP session tests | Tier-1 | Update generated guidance only if step-limit semantics are exposed to app/agent authors |
| LKW-H0.3 | Emit diagnostic/runtime event for post-finalization hook failure | Nexus lifecycle / runtime events | Tier-1 | Propagate generic diagnostic/event pattern to runtime docs/templates if applicable |

### Acceptance criteria

- [ ] Strict/product configuration fails closed or emits explicit configuration violation when required policy wiring is missing.
- [ ] Dev/test permissive policy behavior remains available only when explicitly selected and visible.
- [ ] Regression test proves whether `max_steps=N` permits exactly N steps and rejects step N+1.
- [ ] Finalization/lifecycle hook failure is visible in trace/diagnostics and is not silently swallowed.
- [ ] Existing gate and affected runtime tests remain green.
- [ ] Platform proof checklist in §0a is completed for each task.

### Out of scope (LKW-H0)

- `NexusLoop` constructor refactor.
- `StepKernelContext` decomposition.
- Hosted observability product.
- Full packaging split.
- New product features outside LKW safety and diagnosability.

---

## 3. Next active wave — LKW.1: Domain UAEP proof

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

### Tasks

| ID | Task | Module | Owner | Platform propagation |
|----|------|--------|-------|----------------------|
| LKW.1.1 | Indexer steps: path validation + `rag.ingest_document` loop | `agents/local_indexer/` `on_next_step` / cognitive pattern hooks | Tier-2 | Update agent scaffold/docs if this becomes the canonical tool-invocation pattern |
| LKW.1.2 | Search steps: `rag.retrieve` + evidence formatting | `agents/local_search/` `on_next_step` / cognitive pattern hooks | Tier-2 | Update evidence/result patterns if reusable by generated agents |
| LKW.1.3 | Synthesizer stub: shadow `workspace.write_file` | `agents/local_synthesizer/` `on_next_step` / cognitive pattern hooks | Tier-2 | Update scaffold guidance for shadow-write outputs if generic |
| LKW.1.4 | Acceptance test: fixture doc ingest → search cites source | `applications/.../tests/` or `tests/acceptance/` | Tier-3 | Add scaffold/test template if this becomes the canonical app acceptance pattern |
| LKW.1.5 | Env/settings parity check | `.env.example`, `host/settings.py`, docs | Tier-3/platform | Ensure app settings pattern can inform scaffolded app settings |
| LKW.1.6 | Docker/run parity check | Dockerfile, compose, build/run docs | Tier-3/platform | Keep generated Docker/build docs aligned with real LKW execution |

### Product acceptance criteria

- [ ] `POST /v1/local_workspace/run` with `metadata.source_paths` + `capability=local.workspace.index` completes.
- [ ] Follow-up search returns answer referencing ingested content.
- [ ] Synthesize with `shadow_workspace: true` writes artifact under shadow root.
- [ ] Original user files are not modified.
- [ ] No Slack, tray, watcher, or OS service required.
- [ ] `uv run pytest` agent + host smoke green.

### Platform acceptance criteria

- [ ] Platform proof checklist in §0a is completed.
- [ ] Reusable env/settings lessons are reflected in shared settings/scaffold/docs or recorded as a blocking follow-up.
- [ ] Reusable Docker/build/run lessons are reflected in Docker templates/docs or recorded as a blocking follow-up.
- [ ] Reusable agent/application patterns are reflected in scaffold templates/docs or recorded as a blocking follow-up.
- [ ] Any dependency/profile lesson is reflected in `pyproject.toml` or recorded as a blocking follow-up.

### Out of scope (LKW.1)

- Tray UI.
- Slack.
- File watcher.
- OS service installer.
- `local.workspace.*` skill bundle, except any minimal stub explicitly needed for LKW.1 tests.
- Broad harness refactor unrelated to LKW acceptance or platform propagation.

---

## 4. LKW-H1: live trace/evidence inspection for LKW runs

### Goal

Make one real LKW run inspectable without reading internal runtime code, and ensure the trace/evidence pattern is reusable by future applications.

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

| ID | Task | Module | Owner | Platform propagation |
|----|------|--------|-------|----------------------|
| LKW-H1.1 | Define minimal LKW trace/evidence inspection contract | LKW host/debug docs or tests | Tier-3 | Promote reusable inspection contract to platform docs if generic |
| LKW-H1.2 | Ensure LKW run emits/records tool, policy, RAG, and shadow artifact evidence | runtime events + LKW host | Tier-1/3 | Update event/trace scaffold or docs if reusable |
| LKW-H1.3 | Add smoke/assertion for inspectable LKW run output | `applications/local_workspace_application/...tests` | Tier-3 | Update generated app test pattern if reusable |

### Acceptance criteria

- [ ] A reviewer can see what happened in an LKW run from task submission to terminal result.
- [ ] Tool calls, policy decisions, RAG evidence, and shadow artifact path are visible.
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

---

## 13. Stop conditions

Stop broad harness work when:

- the change cannot be tied to an LKW acceptance criterion or platform propagation requirement;
- the change only improves conceptual elegance;
- the change starts a platform-wide refactor before LKW.1 is demonstrable;
- the change makes LKW harder to run locally;
- the change introduces new abstractions without a product proof requirement.

Do not stop platform propagation when LKW reveals a reusable pattern. In that case, update the platform/scaffold/deploy surface in the same iteration or record a blocking follow-up before moving to the next LKW wave.
