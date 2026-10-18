# PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-ADR-01 — Strict reflex HTTP → sanctioned external-work invocation admission

| Field | Value |
|-------|-------|
| **Status** | **RATIFIED — SCOPED ARCHITECTURE DECISION / OPERATOR APPROVED** (2026-10-18; implementation not yet performed) |
| **Date** | 2026-10-18 |
| **Branch** | `development` |
| **START_HEAD / evidence SHA** | `d460186cc8929bb3689fe8afd9cb56294de3afe0` |
| **Parent** | **PROD-Q-R1-R7-Q** → integrated reflex external-work admission |
| **Blocker addressed (design)** | Strict reflex HTTP execution does not traverse sanctioned `ProviderInvocation` + `GovernedProviderInvocationDispatchGate` lifecycle in one request |
| **Production delta (this task)** | **0** — documentation only |
| **Related ADRs** | [`PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md`](PROD_Q_R1_ADR_03_DURABLE_PROVIDER_INVOCATION_STORE.md) (RATIFIED) · [`PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md`](PROD_Q_R1_R7_Q_P3_A2_HTTP_CALLER_AUTHENTICATION_AND_TENANT_AUTHORIZATION_ADR.md) · [`PROD_Q_R1_DISPATCH_Q1_PARENT_GATE_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_DISPATCH_Q1_PARENT_GATE_QUALIFICATION_EVIDENCE.md) |
| **Qualification authority** | [`PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md`](PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md) · [`PROD_Q_R1_STRICT_HTTP_INVOCATION_IDENTITY_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_STRICT_HTTP_INVOCATION_IDENTITY_QUALIFICATION_EVIDENCE.md) · [`PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md`](PROD_Q_R1_R7_Q_QUALIFICATION_INVENTORY_AND_GATE_PLAN.md) · [`PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md`](PROD_Q_R1_R7_Q_SECTION_C_REMAINING_GATES_RECONCILIATION.md) |
| **Does not close** | **PROD-Q-R1-R7-Q** · **R7-Q** · **PROD-Q** · **Package P** (interpretation **A** remains scoped) · any global **FRZ-*** PASS · **DIM6/DIM17** (OP-01…04) |

**Competing ADR check:** No ratified ADR assigns reflex HTTP to a second invocation store or alternate production dispatcher. **ADR-03** and **DISPATCH-Q1** define sanctioned composition; they do **not** wire the reflex agent entry. **No conflict — proceed.**

**Program posture (unchanged):** Package P interpretation **A** = scoped dual-path evidence; interpretation **B** = **NOT PROVEN**. Gate B and Package D **CLOSED — SCOPED**. D1–D3 pending. **R7-Q** and **PROD-Q** remain **BLOCKED**.

---

## 1. Verified current-HEAD facts (@ `d460186cc…`)

### 1.1 Root-cause code (confirmed)

| Location | Fact |
|----------|------|
| `agents/external_contractor_adapter/external_work_adapter.py` | `adapt_from_step_metadata()` constructs `ExternalWorkAdapter(integration, authorization_boundary=…)` **without** `invocation_dispatch` (L1347). |
| Same file | `_tracked_execute()` (L1053–L1072): if `self._invocation_dispatch is not None`, requires `provider_invocation` and calls `dispatch_after_intent_persisted`; **else** calls `_dispatch_provider()` → direct provider `execute()`. |
| `agents/external_contractor_adapter/steps/domain_job.py` | Reflex step `run_domain_job()` → `adapt_from_step_metadata()`; receives `authorization_boundary` from agent injection only. |
| `agents/external_contractor_adapter/external_contractor_adapter_agent.py` | `act()` → `run_domain_job(..., authorization_boundary=self._authorization_boundary)` — no orchestrator or dispatch port. |
| `applications/governed_contractor_application/host/production_external_work_composition.py` | `build_governed_external_work_production_runtime()` builds `GovernedProviderInvocationDispatchGate`, injects into `ExternalWorkAdapter` as `invocation_dispatch` and `reliability_aware_invocation_dispatch` (L311–L335). |
| `applications/governed_contractor_application/host/provider_invocation_lifecycle.py` | `GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted()` — `admit_provider_invocation_intent` then `execute()` (L157–L201). |
| `applications/governed_contractor_application/host/orchestrator.py` | `GovernedExternalWorkOrchestrator._new_invocation()` mints canonical `ProviderInvocation`; `create()` passes `provider_invocation=invocation` into `adapter.create_and_map()` (L313–L335). |
| `intergrax/contracts/provider_invocation_store.py` | Canonical `ProviderInvocationStore` port. |
| `applications/governed_contractor_application/serving/fastapi_router.py` | Strict HTTP: `GovernedContractorRunService.run_task()` → `HostTaskExecutionPort.execute(task)` — no HTTP `invocation_id`; capability `external_contractor.adapt` (L109–L121). |

### 1.2 Qualification finding (reconciled)

[`PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md`](PROD_Q_R1_PACKAGE_P_INDEPENDENT_EVIDENCE_RECONCILIATION.md) §2–§4: strict HTTP and sanctioned `build_governed_external_work_production_runtime` are **independently proven**; **integrated** reflex HTTP → PIS → provider in **one** request is **NOT PROVEN** @ HEAD.

---

## 2. Mandatory root-cause qualification

### 2.1 Which external-work operations have meaningful side effects?

| Action constant | Provider mutation? | Admission class |
|-----------------|-------------------|-----------------|
| `ACTION_CREATE_EXTERNAL_WORK` (`external_work.create`) | **Yes** — `create_work` | Meaningful side effect (`MeaningfulSideEffectKind.MUTATION`) via `_run_meaningful_side_effect` in `create_and_map` |
| `ACTION_ACCEPT_QUOTE` (`external_work.accept_quote`) | **Yes** — forward acceptance | Meaningful side effect when acceptance evidence supplied |
| `ACTION_CANCEL_EXTERNAL_WORK` (`external_work.cancel`) | **Yes** — cancel | Meaningful side effect on cancel paths |
| Enrich / `get_quote` / correlate / read status | **No** (observational) | Documented in adapter module docstring and `_OBSERVATIONAL_PROVIDER_METHODS`; not gated as mutation |

**Not every adapter call is mutating;** reflex `adapt_from_step_metadata` may surface continuation blockers without provider mutation when acceptance is absent (quote-forward path).

### 2.2 Which require durable invocation admission under existing contracts?

Per **ADR-03**, **DISPATCH-Q1**, and **FRZ-REL-05** product semantics: **all mutating external-work operations** on the **strict production** graph must pass:

1. Governance authorization (`MeaningfulSideEffectAuthorizationBoundary`),
2. Canonical `ProviderInvocation` identity (host-minted, not HTTP-supplied),
3. `ProviderInvocationStore` intent admission **before** provider `execute()`,
4. Durable outcome via orchestrator / lifecycle helpers.

Observational calls do **not** require PIS rows.

### 2.3 Is the direct `_dispatch_provider()` fallback legitimate?

| Context | Legitimate? | Rationale |
|---------|-------------|-----------|
| Unit / lab tests | **Yes** | `adapter_test_wiring.allow_adapter(..., invocation_dispatch=None)`; tests assert policy without PIS |
| `offline_demo.py` | **Scoped non-production** | Explicit demo composition |
| Strict production reflex HTTP | **No** | Violates single sanctioned admission graph documented in DISPATCH-Q1 |

### 2.4 Can strict production reach the fallback for a qualifying side effect?

| Claim | Classification |
|-------|----------------|
| **Code-level bypass exists** | **CONFIRMED** — reflex chain builds adapter with `invocation_dispatch=None`; if `authorization_boundary` is injected (production host composition via `compose_governed_contractor_host_runtime`), `_run_meaningful_side_effect` can authorize and call `_dispatch_provider()` without PIS |
| **Runtime reachability on qualified strict HTTP fixture** | **NOT PROVEN** in committed qualification tests (H01 dual-path only); static call graph supports reachability when strict host wires boundary + integration without binding reflex to `build_governed_external_work_production_runtime` |

**Distinction:** mandatory **IN-SCOPE BLOCKER** is the **architectural seam** (reflex not on sanctioned lifecycle); **integrated E2E failure** is **TRACKED FREEZE DEBT** until implementation + qual (interpretation **B**).

### 2.5 Existing configuration enforcing deny before fallback?

| Mechanism | Effect |
|-----------|--------|
| Missing `authorization_boundary` | **DENY** — `side_effect_authorization_boundary_missing` (`external_work_adapter.py` L908–L924) |
| `invocation_dispatch` set + `provider_invocation is None` | **DENY** — `provider_invocation_required_for_durable_dispatch` (L1055–L1058) |
| `invocation_dispatch is None` | **No PIS gate** — direct provider after governance ALLOW |
| Strict HTTP caller auth (P3-A2) | **HTTP admission only** — does not bind PIS |
| `build_governed_external_work_production_runtime` | **Fails closed** without durable `provider_invocation_store` — applies only when that builder is used |

**Gap:** no fail-closed rule today that forbids **strict production agent reflex** from mutating without dispatch port.

---

## 3. Closed-world inventory (call graph)

### 3.1 Strict HTTP reflex path (before — @ HEAD)

```text
POST /v1/governed_contractor/run
  → GovernedContractorRunService.run_task
       [applications/governed_contractor_application/serving/fastapi_router.py]
  → authorize_strict_production_serving_http_caller (when strict_serving_tenant_binding set)
  → Task(capability=external_contractor.adapt)
  → HostTaskExecutionPort.execute
       [intergrax/runtime/execution/host_task.py — protocol]
  → StrategyExecutionRouter.execute (host orchestration)
       [intergrax/runtime/execution/strategy_router.py]
  → Agent reflex / UAEP → ExternalContractorAdapterAgent.act
       [agents/external_contractor_adapter/external_contractor_adapter_agent.py]
  → run_domain_job
       [agents/external_contractor_adapter/steps/domain_job.py]
  → adapt_from_step_metadata
       [agents/external_contractor_adapter/external_work_adapter.py]
  → ExternalWorkAdapter (no invocation_dispatch)
  → create_and_map → _run_meaningful_side_effect → _tracked_execute
  → [BYPASS EDGE] _dispatch_provider() when invocation_dispatch is None
  → ExternalWorkIntegration.create_work / …
```

| Edge | Caller | Callee | Contract | Semantic owner | Composition owner | Authority | On failure |
|------|--------|--------|----------|----------------|-------------------|-----------|------------|
| HTTP run | FastAPI route | `GovernedContractorRunService.run_task` | `GovernedContractorRunRequestV1` | GCA serving | `factory.py` / mount | P3-A2 HTTP caller | 401/400/502 |
| Execute task | Run service | `HostTaskExecutionPort` | `Task` | Runtime execution | Harness `build_environment_host_task_execution` | Root admission + tenant binding | Propagates / 502 |
| Agent act | Router | `ExternalContractorAdapterAgent.act` | `AgentStepContext` | Tier-2 agent | `agent_builders.py` | Agent contract | Domain summary error |
| Domain job | Agent | `adapt_from_step_metadata` | metadata map | Adapter | **None at composition** (helper) | Governance boundary only | Structured `ExternalWorkAdapterResult` |
| Side effect | Adapter | `_tracked_execute` | `ProviderInvocationDispatchPort` optional | ER + GCA lifecycle | **Sanctioned only via** `production_external_work_composition.py` | PIS + gate when wired | Persistence / deny |

### 3.2 Sanctioned external-work path (parallel — proven separately)

```text
build_governed_external_work_production_runtime
  [applications/governed_contractor_application/host/production_external_work_composition.py]
  → GovernedExternalWorkOrchestrator.create / accept / cancel
       [applications/governed_contractor_application/host/orchestrator.py]
  → _new_invocation → ProviderInvocation
  → ExternalWorkAdapter(..., invocation_dispatch=GovernedProviderInvocationDispatchGate)
  → _tracked_execute → dispatch_after_intent_persisted → provider
  → persist_provider_invocation_outcome (orchestrator finalize)
```

| Symbol | Owner file | Role |
|--------|------------|------|
| `GovernedContractorRunRequestV1` | `applications/governed_contractor_application/serving/schemas.py` | HTTP body — no `invocation_id` |
| `GovernedProviderInvocationDispatchGate` | `provider_invocation_lifecycle.py` | Ordinary dispatch admission |
| `ProviderInvocationRecoveryRepeatPort` | `intergrax/runtime/enterprise_reliability/provider_invocation_recovery.py` | Recovery repeat only — not ordinary HTTP retry |
| `MeaningfulSideEffectAuthorizationBoundary` | `intergrax/runtime/policy/meaningful_side_effect_authorization.py` | Governance permission |
| `ProviderInvocation` | `intergrax/contracts/provider_invocation.py` | Durable invocation identity |
| `ProviderInvocationStore` | `intergrax/contracts/provider_invocation_store.py` | Intent/outcome persistence |

### 3.3 Tests relying on optional direct dispatch (`invocation_dispatch=None`)

| Test / fixture | Path |
|----------------|------|
| `agents/external_contractor_adapter/tests/fakes/adapter_test_wiring.py` | `allow_adapter(..., invocation_dispatch=None)` default |
| `tests/unit/agents/external_contractor_adapter/test_pg_fix_a_side_effect_boundary.py` | `ExternalWorkAdapter` without dispatch — policy only |
| `test_adapt_from_step_metadata_*` | Reflex metadata entry without PIS |
| `test_gr6_r2_production_decision_requirement.py` | Adapter with boundary, no gate |
| **Contrast:** `test_prod_q_r1_strict_http_invocation_identity_qualification.py::test_h08_*` | Asserts **sanctioned** runtime adapter **has** gate — not reflex chain |

---

## 4. Architecture options

### 4.1 Candidate A — Reuse sanctioned external-work orchestration (preferred)

Reflex qualifying mutations delegate to **`GovernedExternalWorkOrchestrator`** (same adapter instance and gate as `build_governed_external_work_production_runtime`), not to bare `adapt_from_step_metadata` → orphan adapter.

| Pros | Cons |
|------|------|
| Single invocation minting (`_new_invocation`) | Requires host composition to expose orchestrator (or narrow port) to Tier-2 agent |
| Reuses outcome finalization and host execution store | Reflex response shape must map `OrchestratorStepResult` → domain summary |
| No second store / gate | Operator must ratify agent↔host wiring boundary |

### 4.2 Candidate B — Fail-closed typed boundary (complement to A)

Strict production composition **rejects** agent wiring that permits mutating external-work without `ProviderInvocationDispatchPort` + durable store. Qualifying operations without admission return structured deny — **no** silent fallback.

| Pros | Cons |
|------|------|
| Mechanical regression gate | Does not alone integrate HTTP; must pair with A |
| Aligns with FRZ-PRD-08 | Requires new composition validation symbol |

### 4.3 Candidate C — Other sanctioned architecture

**Not required.** Injecting `invocation_dispatch` only into `adapt_from_step_metadata` without orchestrator-level `ProviderInvocation` minting **rejected** (violates identity ownership). HTTP-supplied `invocation_id` **rejected**. Second dispatch gate **rejected**.

### 4.4 Recommended decision (pending operator ratification)

**Ratify A + B:** strict reflex HTTP → **`GovernedExternalWorkOrchestrator`** for mutating paths; strict composition **fail-closed** when reflex agent lacks sanctioned production external-work runtime handle.

**RATIFIED WIRING RULE:** GCA host injects a **narrow, platform-defined, strongly typed external-work operation port** into Tier-2 agent composition; no dependency from the agent on `applications/` or `GovernedExternalWorkProductionRuntime`. Reuse the existing orchestrator behind this port. Exact signature must be checked against current contract ownership in IMPL-01; architecture conflicts require STOP, not an untyped callback.

---

## 5. Identity and lifecycle (mandatory)

| # | Decision |
|---|----------|
| 1 | **Who creates `ProviderInvocation`?** — **`GovernedExternalWorkOrchestrator._new_invocation`** (host, GCA). Reflex must not mint in adapter metadata. |
| 2 | **Association** — `task_id`, `run_id` from active execution (`AgentStepContext` / governed task binding); `tenant_id` / `principal_id` from metadata + strict HTTP principal binding; `execution_id` from orchestrator canonical identity gate (`_require_canonical_execution_identity`). |
| 3 | **Retries / recovery** — Same `invocation_id` on ordinary retry only via gate admission rules; **recovery** uses `ProviderInvocationRecoveryRepeatPort` / GR-7-A7 — not HTTP replay. |
| 4 | **Typed carrier** — `ProviderInvocation` passed into `ExternalWorkAdapter.create_and_map(..., provider_invocation=…)`; store keyed by `invocation_id`; metadata `META_PROVIDER_INVOCATION_ID` for correlation only. |
| 5 | **Intent vs execute** — `GovernedProviderInvocationDispatchGate.dispatch_after_intent_persisted`: `admit_provider_invocation_intent` **before** `execute()`. |
| 6 | **Persistence failure** — No provider mutation if intent not admitted; adapter returns `provider_invocation_intent_persistence_failed`; orchestrator maps to host state deny. |
| 7 | **UNKNOWN outcome** — Classified per `classify_provider_invocation_status`; **no** ordinary re-dispatch via gate (IG-F01, H03); recovery decision-only. |
| 8 | **Ordinary vs recovery** — Ordinary: dispatch gate only. Recovery: `applications/governed_contractor_application/host/provider_invocation_recovery.py` + `ProviderInvocationRecoveryRepeatPort`. |
| 9 | **Final outcome recorder** — Orchestrator `persist_provider_invocation_outcome` + execution store state transitions. |
| 10 | **Authoritative execution state** — `GovernedExecutionStore` host states + durable PIS outcome — not HTTP response body alone. |

**Explicit:** Identical HTTP JSON bodies **≠** identical `invocation_id` (H04). No provider exactly-once claim.

---

## 6. Governance and security invariants (design targets)

| Invariant | Design support |
|-----------|----------------|
| Governance policy authoritative | Existing `MeaningfulSideEffectAuthorizationBoundary` before `_tracked_execute` |
| Execution cannot widen permission | No HTTP `invocation_id`; orchestrator mint only |
| Proposal ≠ permission ≠ execution | Continuation surfaces without acceptance remain non-mutating |
| Production side effects require durable admission | A + B |
| No alternate production dispatcher | Reuse `GovernedProviderInvocationDispatchGate` only |
| Strict vs lab separation | Lab keeps `adapt_from_step_metadata` without orchestrator when composition omits strict handle |
| Missing authority fail-closed | B: composition validation + runtime deny |
| Tenant isolation | PIS tenant keys + strict serving tenant port (existing) |
| Observability evidence-only | No change |
| Recovery no silent duplicate effects | GR-7-A7 repeat port unchanged |

**Gap:** If operator rejects host-orchestrator injection into Tier-2 agent → **STOP — ARCHITECTURE DECISION REQUIRED** (new host port contract).

---

## 7. Enterprise design audit matrix (@ HEAD + target)

| Dimension | Requirement | Symbols / files | Risk @ HEAD | Target enforcement |
|-----------|-------------|-----------------|-------------|-------------------|
| Boundaries | No forbidden imports | Agent must not import `applications/` stores | Reflex bypasses app lifecycle while using app boundary | Orchestrator port injected from Apps host only |
| Communication | Sanctioned ports | `HostTaskExecutionPort`, `ProviderInvocationDispatchPort` | Reflex skips dispatch port | Inject orchestrator via composition |
| Ownership | One semantic owner | PIS: ER contract; orchestration: GCA | Two entrypoints to same adapter semantics | Single mutating entry: orchestrator |
| Contracts | Typed interfaces | `ProviderInvocation`, `GovernedContractorRunRequestV1` | Untyped metadata-only reflex | Orchestrator `create()` args |
| Typing | No pseudo-contracts | Pydantic models | — | Preserve |
| Pluginability | Provider via `ExternalWorkIntegration` | Unchanged | — | Unchanged |
| Replaceability | Store/provider swappable | `ProviderInvocationStore` | — | Unchanged |
| Security | No bypass | FRZ-SEC-04 | Reflex mutation without PIS | B gate |
| Governance | Auth separate from execution | `MeaningfulSideEffectAuthorizationBoundary` | Auth without PIS | A |
| Persistence | Intent before effect | `admit_provider_invocation_intent` | Bypass edge L1072 | Remove reachability in strict |
| Recovery | Separate authority | `ProviderInvocationRecoveryRepeatPort` | — | Unchanged |
| Observability | Evidence only | Reliability observer | — | Unchanged |
| Traceability | Identity continuity | `invocation_id` in PIS | Broken on reflex | A |
| Regression | Mechanical gates | IG-F01, architecture tests | No reflex-specific gate | New static gate §9 |
| Fail-closed | Missing authority | Partial (boundary only) | dispatch optional | B |

---

## 8. Implementation blueprint (child task — **do not implement in this ADR task**)

**Proposed child:** `PROD-Q-R1-R7-Q-REFLEX-EXTERNAL-WORK-ADMISSION-IMPL-01`

### 8.1 File-level changes

| File | Owner | Symbol | Change | Contract |
|------|-------|--------|--------|----------|
| `applications/governed_contractor_application/host/governed_contractor_host_runtime_composition.py` | GCA host | `GovernedContractorHostRuntimeComposition` | Add optional `external_work_production_runtime: GovernedExternalWorkProductionRuntime \| None` (or protocol with `.orchestrator`) | Reuse `GovernedExternalWorkProductionRuntime` from `production_external_work_composition.py` |
| `applications/governed_contractor_application/host/factory.py` | GCA | `create_governed_contractor_backend_app` | When `production_mode` + `provider_invocation_store` durable: call `build_governed_external_work_production_runtime`, attach to `resolved_host_runtime` | Existing builder only — **no new store** |
| `applications/governed_contractor_application/host/agent_builders.py` | GCA | `build_external_contractor_adapter_factory` | Pass orchestrator/runtime handle into agent | Forbidden: agent importing SQLite store |
| `agents/external_contractor_adapter/external_contractor_adapter_agent.py` | Tier-2 | `ExternalContractorAdapterAgent` | Hold optional orchestrator port; `act()` forward to domain job | Protocol defined in `intergrax/contracts` **only if** needed to avoid Apps import in agents — **prefer** callback port owned by Apps passed as callable |
| `agents/external_contractor_adapter/steps/domain_job.py` | Tier-2 | `run_domain_job` | If strict runtime injected: map metadata → `orchestrator.create` / accept / cancel; else legacy `adapt_from_step_metadata` | No `applications` import — injection only |
| `agents/external_contractor_adapter/external_work_adapter.py` | Tier-2 | `adapt_from_step_metadata` | Document lab/non-strict; optional **fail-closed** when `require_durable_dispatch=True` injected | Do not mint `ProviderInvocation` here |
| `intergrax/applications/_shared/strict_production_*` or GCA architecture test module | Platform | New gate test | Assert strict GCA factory wires reflex agent with production external-work runtime when external work enabled | Mirror `test_production_composition_architecture_gate.py` patterns |
| `tests/qualification/prod_q/test_prod_q_r1_strict_http_invocation_identity_qualification.py` | Qual | New `test_h09_*` (name TBD) | Single HTTP POST → PIS row + `create_calls==1` on fake provider via **reflex** | Real path — minimal mocks beyond existing fixture |

**Forbidden:** HTTP `invocation_id`; second gate; recovery for ordinary retry; direct store in `domain_job`; weakening lab tests that use `invocation_dispatch=None` without strict flag.

### 8.2 Before / after graphs

**Before (strict production — two graphs):**

```text
[HTTP] → HostTaskExecutionPort → reflex → adapt_from_step_metadata → Adapter(dispatch=None) → provider
[Qual only] build_governed_external_work_production_runtime → Orchestrator → Adapter(dispatch=Gate) → PIS → provider
```

**After (target — unified):**

```text
[HTTP] → HostTaskExecutionPort → reflex → run_domain_job(strict runtime)
  → GovernedExternalWorkOrchestrator.create|…
  → Adapter(dispatch=Gate, provider_invocation=minted)
  → PIS admit → provider → outcome durable
```

**Removed edge:** strict production reachability of `ExternalWorkAdapter._tracked_execute` → `_dispatch_provider()` for `CREATE` / `ACCEPT` / `CANCEL`.

### 8.3 Migration

| Consumer | Rule |
|----------|------|
| Lab / unit tests | Keep `adapt_from_step_metadata` without orchestrator when runtime handle is `None` |
| `offline_demo.py` | Unchanged (non-strict) |
| Provider plugins | Unchanged — still `ExternalWorkIntegration` |
| Strict HTTP qualification | Add integrated H09+; do not weaken H01–H08 |

---

## 9. Future implementation Definition of Done

Deterministic tests (child task) must prove:

1. One strict HTTP request → sanctioned external-work lifecycle end-to-end.
2. `ProviderInvocation` identity canonical and typed.
3. Intent durably persisted before dispatch.
4. Exactly one ordinary dispatch admission per `invocation_id` (gate + store).
5. Fake provider via `ExternalWorkIntegration`.
6. Durable outcome readable from PIS.
7. Tenant/principal isolation (extend H05 pattern).
8. Governance DENY → zero provider calls.
9. Persistence failure → zero provider calls.
10. Missing strict dispatch authority → fail-closed (composition or runtime).
11. Duplicate replay → no unauthorized provider repeat (IG-F01 class).
12. UNKNOWN → no ordinary gate retry.
13. Recovery via `ProviderInvocationRecoveryRepeatPort` only.
14. Non-strict composition compatible.
15. Static architecture gate blocks direct strict reflex dispatch.
16. Tests exercise **one** graph, not dual mocked pipelines.

**pytest (this ADR task):** `0` — no production or test code changed; qualification deferred to IMPL child.

---

## 10. FRZ mapping (local contribution only — global rows stay OPEN)

| FRZ | Relevance | @ HEAD | After IMPL (expected) |
|-----|-----------|--------|------------------------|
| **FRZ-PRD-08** | Production bypass count | **OPEN** — reflex bypass seam | Target: remove strict reflex bypass |
| **FRZ-SEC-01** | Tenant isolation | Partial scoped (H05) | Extend integrated path |
| **FRZ-SEC-04** | Auth bypass | HTTP chain scoped; PIS bypass on reflex | Close reflex PIS bypass class |
| **FRZ-REL-03** | Idempotency ownership | Gate scoped; HTTP re-entry NOT PROVEN | Orchestrator-owned `invocation_id` on reflex |
| **FRZ-REL-05** | External-op lifecycle | Composition PASS; reflex NOT PROVEN | Integrated termination semantics |
| **FRZ-CTR-*** / **FRZ-BND-*** / **FRZ-OWN-*** | Layer ownership | Agent must not own PIS | Orchestrator injection |
| **FRZ-REG-02** / **FRZ-REG-03** | Regression gates | IG-F01 partial | New reflex integration gate |

**No global PASS** may be recorded from this ADR alone.

---

## 11. Findings classification

| ID | Class | Summary |
|----|-------|---------|
| **BP-REFLEX-01** | **IN-SCOPE BLOCKER** | Reflex adapter entry bypasses `GovernedProviderInvocationDispatchGate` + minted `ProviderInvocation` |
| **BP-01** (Package P) | **TRACKED FREEZE DEBT** | Integrated HTTP→PIS→provider NOT PROVEN — addressed by IMPL + qual |
| **ADR-03 §2.1 dispatch replay** | **TRACKED FREEZE DEBT** | Equivalent intent + gate `execute()` — orthogonal; not widened by this ADR |
| **DIM6/DIM17** | **TRACKED FREEZE DEBT** | OP-01…04 — unchanged |
| **R7-Q / PROD-Q** | **BLOCKED** | Parent cannot close until integrated path qualified |

---

## 12. Impact

| Artifact | Impact |
|----------|--------|
| **Package P** | Interpretation **A** unchanged; **B** becomes implementable after IMPL + operator qual |
| **DIM20** | Dependency on integrated proof — **OPEN** until child completes |
| **R7-Q** | Unblocks integrated E2E gate design; remains **NOT QUALIFIED** |
| **PROD-Q** | **BLOCKED** |

---

## 13. Operator ratification — 2026-10-18

**Formal architecture decision: RATIFIED — SCOPED.** Operator explicitly approved the four choices below following independent GitHub review of ADR commit `c918c2fff0f41f064690025de3deb8da471b58ff`:

1. **A+B required together:** the existing sanctioned `GovernedExternalWorkOrchestrator` controls strict reflex mutating external-work, and strict composition must fail closed without sanctioned durable invocation admission.
2. **Typed seam:** use a **narrow platform-defined strongly typed port injected from GCA host** into Tier-2. The agent must not import `applications/`, store implementations, or host runtime concrete types. The port may expose only the required create/accept/cancel operations; no generic `Callable[..., Any]`, `object`, reflection, or metadata pseudo-contract. Exactly-one semantic owner and exactly-one composition owner remain mandatory.
3. **All three mutation classes in one IMPL-01:** create, accept_quote, cancel. Observational reads remain separately classified. Non-strict lab/demo direct behavior can remain only outside strict production.
4. **Package P interpretation A unchanged:** earlier scoped acceptance is not retroactively broadened. Integrated interpretation B is mandatory for the **R7-Q integrated gate**, not for historical scoped Package P closure.

**Implementation authorization is bounded by these invariants, not a preapproval of any particular code patch.** If a narrow port requires a new semantic owner, widens downstream authority, or conflicts with a ratified architecture, **STOP — ARCHITECTURE DECISION REQUIRED** before writing production code. IMPL-01 must supply tests and independent audit. This ratification does **not** close Package P, DIM20, R7-Q, PROD-Q or any global FRZ criterion.

---

## 14. Audit commands (@ ADR authoring)

```text
git fetch origin development
git rev-parse HEAD  → d460186cc8929bb3689fe8afd9cb56294de3afe0
git merge-base --is-ancestor d460186cc8929bb3689fe8afd9cb56294de3afe0 HEAD  → OK
git status --short  → clean
pytest  → 0 (documentation-only task)
```

---

**Status statement:** This architecture decision is **RATIFIED — SCOPED** by operator on 2026-10-18. IMPL-01 remains unimplemented/unqualified; **PROD-Q-R1-R7-Q**, **R7-Q**, **DIM20** and global FRZ remain open.
