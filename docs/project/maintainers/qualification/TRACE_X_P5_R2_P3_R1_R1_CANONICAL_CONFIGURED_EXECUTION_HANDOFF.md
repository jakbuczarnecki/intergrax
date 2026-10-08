# TRACE-X-P5-R2-P3-R1-R1 — Canonical Configured Execution Handoff (architecture lock)

| Field | Value |
|---|---|
| **Task** | `TRACE-X-P5-R2-P3-R1-R1` — CONFIGURE_EXISTING Canonical Execution Handoff Reconciliation |
| **Parent** | `TRACE-X-P5-R2-P3-R1` → `TRACE-X-P5-R2-P3` → `TRACE-X-P5-R2` → `TRACE-X-P5` → `TRACE-X` |
| **START_HEAD** | `61cfc96a42cef104819b90ca36a7dab7decfeba1` |
| **Disposition** | **READY FOR AUDIT** (not CLOSED) |
| **Production delta** | **0** (docs / architecture only) |
| **FRZ-TRC-11** | **OPEN** (scoped evidence only; no PASS promotion) |
| **P5-GAP-04** | **IMPLEMENTATION IN PROGRESS** |

## Canonical stage (unchanged @ START_HEAD)

| Stage | Status |
|---|---|
| TRACE-X | CURRENT |
| TRACE-X-P5 | CURRENT / BLOCKED ON R2 |
| TRACE-X-P5-R2 | CURRENT / P3 NEXT |
| TRACE-X-P5-R2-P3 | NEXT / REQUIRED / NOT ENTERED |
| TRACE-X-P5-R2-P3-R1 | Host composition evidence (upstream); handoff debt = this child |
| P4 | NOT ENTERED |

## Root causes reconciled

| Blocker | Root cause @ `61cfc96a…` | Architecture resolution |
|---|---|---|
| **R2-P3-CANONICAL-DISPATCH-COMPOSITION-09** | `build_production_marketplace_qualified_capability_execution_dispatch` builds a **Marketplace-only** `QualifiedCapabilityExecutionBindingHandlerRegistry` + dispatch | Tier-3 **host registry composition** aggregates CodeCraft + Marketplace handlers → **one** `build_qualified_capability_execution_dispatch_service`; Marketplace module returns **handler/composition only**; dispatch helper **TO BE REMOVED / NOT SANCTIONED** |
| **R2-P3-AW-TO-CANONICAL-DISPATCH-REACHABILITY-10** | Governed fulfillment `inner_dispatch` may be marketplace-private dispatch; resume → adapter → dispatch not proven on unified registry | Same **one** canonical dispatch wired as `inner_dispatch` in `build_worker_recovery_governed_fulfillment_wiring` |
| **R2-P3-CONFIGURE-EXISTING-ENTRY-SEAM-11** | `WorkerCapabilityFulfillmentCoordinator.fulfill` calls only `WorkerCapabilityRecoveryPort.coordinate_recovery`; **never** invokes sole producer `WorkerCapabilityAcquisitionDecisionService.decide`; recovery coordinator **never** emits `CONFIGURE_EXISTING_REQUIRED` | **Fulfillment recovery adapter** (new implementation surface) owns entry: `decide` → map `CONFIGURE_EXISTING` → `CONFIGURE_EXISTING_REQUIRED` + `worker_acquisition_decision`; canonical UCA recovery delegated only when `decide` routes to `ACQUIRE_CAPABILITY` |

## Current-HEAD behavior (evidence, not normative)

| ID | Observation |
|---|---|
| A | `WorkerCapabilityAcquisitionDecisionService.decide` can emit `CapabilityAcquisitionDisposition.CONFIGURE_EXISTING` for `EXISTING_CONFIGURATION` candidates (`capability_acquisition_service.py`). |
| B | `WorkerCapabilityFulfillmentRequest` does **not** carry `WorkerCapabilityAcquisitionDecision`. |
| C | `WorkerCapabilityRecoveryOutcome.worker_acquisition_decision` exists and is consumed by `_fulfill_configure_existing`. |
| D | `WorkerCapabilityRecoveryCoordinator.coordinate_recovery` does **not** call `WorkerCapabilityAcquisitionDecisionService` and **never** sets phase `CONFIGURE_EXISTING_REQUIRED`. |
| E | After successful configured fulfillment, `_fulfill_configure_existing` calls `_fulfill_qualified` only when `recovery.phase is QUALIFICATION_COMPLETE`; otherwise **FAIL_CLOSED** — violates P0 §1B.11 Option A. |
| F | `_fulfill_qualified` requires `recovery.acquisition_result` and `recovery.qualification_result`; CONFIGURE_EXISTING entry does not populate them on current production recovery path. |
| G | `build_production_marketplace_qualified_capability_execution_dispatch` is **not** a neutral wrapper — it owns a single-handler registry (`uca6c_marketplace_qualified_execution_composition.py`). |

---

## Q1 — Decision handoff (exactly one typed seam)

**Chosen seam:** `WorkerCapabilityRecoveryOutcome.worker_acquisition_decision: WorkerCapabilityAcquisitionDecision | None`

**Rules:**

- The **only** producer of `CONFIGURE_EXISTING` disposition remains `WorkerCapabilityAcquisitionDecisionService.decide`.
- Fulfillment **must not** re-run discovery, re-select candidates, or construct a second decision object.
- `WorkerCapabilityFulfillmentRequest` **must not** duplicate the decision (no parallel field on the request DTO in the sanctioned design).
- The fulfillment recovery adapter copies the **exact** `WorkerCapabilityAcquisitionDecision` instance returned inside `WorkerCapabilityAcquisitionResult.decision` onto `WorkerCapabilityRecoveryOutcome.worker_acquisition_decision` when mapping to `CONFIGURE_EXISTING_REQUIRED`.

**Forbidden:** inferring CONFIGURE_EXISTING from `REALIZATION_REQUIRED` discovery alone without a decision object; reading `configuration_ref` without the decision’s `selected_candidate`.

---

## Q2 — Configured recovery phase owner

**Owner:** `WorkerCapabilityFulfillmentRecoveryAdapter` (new module in implementation wave; name normative) implementing `WorkerCapabilityRecoveryPort`.

**Call path:**

```text
WorkerCapabilityFulfillmentCoordinator.fulfill(request)
  → WorkerCapabilityFulfillmentRecoveryAdapter.coordinate_recovery(request.acquisition_request, …)
      → WorkerCapabilityAcquisitionDecisionService.decide(request, decided_at=…)
      → if result.decision.disposition is CONFIGURE_EXISTING:
            return WorkerCapabilityRecoveryOutcome(
              phase=CONFIGURE_EXISTING_REQUIRED,
              provenance=<derived from need + discovery_correlation_id>,
              worker_acquisition_decision=result.decision,
              decided_at=…,
            )
      → elif request.recovery_decision.strategy is ACQUIRE_CAPABILITY:
            return WorkerCapabilityRecoveryCoordinator.coordinate_recovery(…)
      → else: map other dispositions to existing phases / FAIL_CLOSED (no CONFIGURE_EXISTING re-decision)
```

`WorkerCapabilityRecoveryCoordinator` **must not** become a second CONFIGURE_EXISTING decision owner; it may only receive delegated canonical UCA recovery.

---

## Q3 — Acquisition / qualification facts

**Pilot / sanctioned P3-R5 vertical slice (Marketplace `database.*` + RELATIONAL_STORE CONFIGURE_EXISTING):**

| Fact | Source | Contract |
|---|---|---|
| **Qualification** | **C — reuse** | `CapabilityQualificationResult` already on `WorkerCapabilityRecoveryOutcome` from the **canonical UCA acquisition + qualification segment** of the same fulfillment episode (same ports as `WorkerCapabilityRecoveryCoordinator`: `CapabilityAcquisitionCoordinatorPort.acquire` → `CapabilityQualificationCoordinatorPort.qualify` via `build_acquisition_qualification_request`). |
| **Acquisition result** | **C — reuse** | `CapabilityAcquisitionResult` on the same `WorkerCapabilityRecoveryOutcome` from that UCA segment. |
| **CONFIGURE_EXISTING decision** | **Producer** | `WorkerCapabilityAcquisitionDecision` on `worker_acquisition_decision` from `decide()` — addresses **integration configuration** only; does **not** replace acquisition/qualification facts. |

**Episode ordering (normative for pilot):**

1. Fulfillment recovery adapter ensures UCA acquisition + qualification facts are present on the recovery outcome **when the worker need requires Marketplace qualified tool execution** (marketplace strategy / handoff evidence).
2. Adapter then runs `decide()`; on `CONFIGURE_EXISTING`, sets phase `CONFIGURE_EXISTING_REQUIRED` while **retaining** `acquisition_result` and `qualification_result` on the same outcome object (phase names orchestration step; facts are not discarded).

**Pure CONFIGURE_EXISTING-only episode** (configuration discovery candidate, **no** UCA acquisition_result / qualification_result): **fail-closed** at `_fulfill_qualified` / resume construction — **no** synthesized acquisition or qualification objects. A future architecture child may define a typed post-INT-CONFIG qualification subject; **out of scope** for this lock.

**Not chosen:** **A** (qualification before configuration only from discovery — no `CapabilityQualificationResult` contract on configuration layer). **B** (post-configuration `qualify()` without an existing subject projector from `ConfiguredCapabilityBinding`) — **no** such projector exists on HEAD.

---

## Q4 — Direct Option A continuation

After `WorkerConfiguredCapabilityFulfillmentService.fulfill_configure_existing` returns non-`None` `ExecutionIntegrationConfigurationAdoption`:

**Exact next call:**

```text
WorkerCapabilityFulfillmentCoordinator._fulfill_qualified(
  request,
  recovery=recovery,  # same outcome; acquisition_result + qualification_result retained
  decided_at=decided_at,
  integration_configuration_adoption=adoption,
)
```

**Invariants (P0 §1B.11):**

- **No** `coordinate_recovery()` rediscovery gate after INT-CONFIG success.
- **No** `_fulfill_realization_required` / generic UCA realization reconcile for INT-CONFIG.
- **No** new `decide()` invocation on the continuation edge.
- Phase check `recovery.phase is QUALIFICATION_COMPLETE` **must be removed** on this edge (implementation wave); continuation is authorized by successful adoption + presence of qualification facts, not by phase enum equality.

---

## Q5 — `WorkerQualifiedCapabilityResumeRequest` field sources (CONFIGURE_EXISTING path)

| Field | Source |
|---|---|
| `worker_instance_id` | `WorkerCapabilityFulfillmentRequest.worker_instance_id` |
| `worker_need_id` | `derive_worker_capability_need_id(request.acquisition_request.need)` |
| `recovery_decision_id` | `request.acquisition_request.need.recovery_decision_id` |
| `provenance` | `recovery.provenance` (implementation may append INT-CONFIG evidence refs; **must not** replace canonical IDs) |
| `acquisition_result` | `recovery.acquisition_result` (**required**; fail-closed if missing) |
| `qualification_result` | `recovery.qualification_result` (**required**; fail-closed if missing / not `QUALIFIED`) |
| `resume_operation_id` | `derive_worker_capability_resume_operation_id(recovery_decision_id=…, qualification_request_id=qualification.qualification_request_id)` |
| `tenant_id` | `request.tenant_id` |
| `task_id` | `request.task_id` |
| `requested_at` | fulfillment `decided_at` |
| `requested_authority_scopes` | `request.requested_authority_scopes` |
| `run_id` / `attempt_id` | `request.run_id` / `request.attempt_id` |
| `integration_configuration_adoption` | `WorkerConfiguredCapabilityFulfillmentResult.adoption` (exact binding from realization) |

**Binding source:** `QualifiedCapabilityBindingService.bind` via `WorkerQualifiedCapabilityResumeCoordinator` using `qualified_capability_subject_from_result(qualification_result)` and `MarketplaceToolQualifiedCapabilityBindingProvider` when strategy/evidence match marketplace handoff (unchanged UCA-6C binding port).

---

## Q6 — One canonical dispatch

**Normative composition shape:**

```text
build_production_codecraft_qualified_capability_execution_handler(...) → CodeCraftQualifiedCapabilityExecutionHandler
build_production_marketplace_qualified_capability_execution_handler(...) → MarketplaceToolQualifiedCapabilityExecutionHandler
        ↓
QualifiedCapabilityExecutionBindingHandlerRegistry((codecraft_handler, marketplace_handler, …))
        ↓
build_qualified_capability_execution_dispatch_service(handler_registry=registry, runtime_policy_admission=…, integration_configuration_pinning=…)
        ↓
QualifiedCapabilityExecutionDispatchService  (sole Execution dispatch authority)
        ↓
GovernedTaskScopedQualifiedCapabilityExecutionDispatchService (AW inner_dispatch wrapper)
```

`QualifiedCapabilityExecutionBindingHandlerRegistry` is **routing only** — not Execution authority.

**Classification:** `build_production_marketplace_qualified_capability_execution_dispatch` = **TO BE REMOVED / NOT SANCTIONED** (private single-handler registry + dispatch).

---

## Q7 — Tier-3 ownership

**Sanctioned owner:** new Application-shared composition module:

`intergrax/applications/_shared/uca6c_qualified_capability_execution_host_composition.py` (name normative)

**Owns:** aggregating handler artifacts from `uca6c_codecraft_qualified_execution_composition` and `uca6c_marketplace_qualified_execution_composition` into **one** registry + **one** `build_qualified_capability_execution_dispatch_service` call; wiring pinning store / resolution from marketplace composition artifact where required.

**Must not own:** execution semantics, recovery, qualification, or INT-CONFIG realization.

**Disposition:** **No STOP** — existing Tier-3 shared composition pattern (`uca6c_*_qualified_execution_composition.py`) can host registry aggregation without a second framework.

---

## After graph (normative)

```text
WorkerCapabilityAcquisitionDecisionService.decide
  → WorkerCapabilityFulfillmentRecoveryAdapter (WorkerCapabilityRecoveryPort)
  → WorkerCapabilityRecoveryOutcome(
        phase=CONFIGURE_EXISTING_REQUIRED,
        worker_acquisition_decision=<exact decision>,
        acquisition_result=<UCA segment, pilot>,
        qualification_result=<UCA segment, pilot>,
      )
  → WorkerCapabilityFulfillmentCoordinator._fulfill_configure_existing
  → WorkerConfiguredCapabilityFulfillmentService.fulfill_configure_existing
  → ExistingCapabilityConfigurationOpportunityReadPort.read_exact
  → ExistingCapabilityConfigurationRealizationPort.realize
  → ExecutionIntegrationConfigurationAdoption
  → WorkerCapabilityFulfillmentCoordinator._fulfill_qualified (Option A; no rediscovery)
  → WorkerQualifiedCapabilityResumeCoordinator.resume
  → WorkerQualifiedCapabilityExecutionEngineAdapter
  → GovernedTaskScopedQualifiedCapabilityExecutionDispatchService
  → QualifiedCapabilityExecutionDispatchService (ONE registry: CodeCraft + Marketplace + …)
  → MarketplaceToolQualifiedCapabilityExecutionHandler (Pattern A / projection)
  → ToolRuntime Governance
```

---

## Ownership matrix

| Concern | Owner |
|---|---|
| CONFIGURE_EXISTING decision | `WorkerCapabilityAcquisitionDecisionService.decide` |
| CONFIGURE_EXISTING_REQUIRED outcome | `WorkerCapabilityFulfillmentRecoveryAdapter` |
| CONFIGURE_EXISTING sequencing | `WorkerCapabilityFulfillmentCoordinator` |
| Decision carry | `WorkerCapabilityRecoveryOutcome.worker_acquisition_decision` |
| INT-CONFIG | `ExistingCapabilityConfigurationRealizationPort` |
| Adoption DTO | `WorkerConfiguredCapabilityFulfillmentService` |
| Qualification facts (pilot) | Reused `CapabilityQualificationResult` on recovery outcome (UCA segment) |
| Binding | `QualifiedCapabilityBindingService` + domain providers |
| Resume | `WorkerQualifiedCapabilityResumeCoordinator` |
| Handler registry aggregation | `uca6c_qualified_capability_execution_host_composition` (new) |
| Dispatch authority | `build_qualified_capability_execution_dispatch_service` (runtime) |
| Governed task scope | `GovernedTaskScopedQualifiedCapabilityExecutionDispatchService` |

---

## Failure matrix (fail-closed)

| Condition | Disposition / outcome |
|---|---|
| Missing `worker_acquisition_decision` on CONFIGURE_EXISTING path | `FAIL_CLOSED` |
| Decision disposition ≠ `CONFIGURE_EXISTING` | `FAIL_CLOSED` / configured fulfillment deny |
| Missing / invalid `configuration_ref` on candidate | `FAIL_CLOSED` / `REALIZATION_FAILED` per configured service |
| Configured fulfillment port unavailable | `FAIL_CLOSED` |
| INT-CONFIG deny / failure | `REALIZATION_FAILED` or `FAIL_CLOSED` |
| Missing `acquisition_result` or `qualification_result` at resume | `FAIL_CLOSED` |
| Qualification not `QUALIFIED` | `QUALIFICATION_FAILED` |
| Binding failure / no provider | Resume binding outcomes → fulfillment `BINDING_FAILED` / `FAIL_CLOSED` |
| Authority admission deny | Execution rejected; no provider I/O |
| Tenant mismatch (request / opportunity / binding / adoption / resume / dispatch) | 0 materialization, 0 pin, 0 provider I/O |
| Missing handler for execution target | Dispatch fail-closed |
| Unsupported tool/category for configured adoption | Handler fail-closed |

No fallback to non-qualified execution.

---

## Tenant continuity

```text
decide context / need
  → fulfillment request.tenant_id
  → opportunity.tenant_id
  → realization request.tenant_id
  → configured binding.tenant_id
  → adoption (via binding)
  → resume request.tenant_id
  → dispatch request.tenant_id
  → Execution tenant
  → pin tenant
```

Any mismatch before provider materialization: **materialization = 0, pin = 0, provider I/O = 0**.

---

## Authority separation

| Layer | Independent |
|---|---|
| CONFIGURE_EXISTING decision | Proposal / candidate selection only |
| INT-CONFIG realization | Governance mutation authorization |
| Execution admission | `WorkerExecutionAdmissionPort` |
| ToolRuntime | Operation authorization |

Adoption **must not** imply permission at downstream layers.

---

## Implementation file map (next wave only)

| File | Change |
|---|---|
| `intergrax/autonomous_work/worker_capability_fulfillment_recovery_adapter.py` | **NEW** — `decide` + phase mapping + UCA delegation |
| `intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py` | Option A continuation; remove erroneous phase gate |
| `intergrax/autonomous_work/worker_recovery_governed_fulfillment_composition.py` | Wire adapter + canonical `inner_dispatch` |
| `intergrax/applications/_shared/uca6c_qualified_capability_execution_host_composition.py` | **NEW** — unified handler registry + dispatch |
| `intergrax/applications/_shared/uca6c_marketplace_qualified_execution_composition.py` | Deprecate/remove private dispatch builder (after host composition exists) |
| Tests | Extend configured fulfillment + composition + production flow gates |

---

## Applicable FRZ (evidence only)

| FRZ | Role |
|---|---|
| **FRZ-TRC-11** | OPEN — primary |
| FRZ-OWN-01, FRZ-OWN-02, FRZ-OWN-03, FRZ-OWN-05 | Ownership |
| FRZ-CTR-01, FRZ-CTR-02 | Contracts |
| FRZ-GOV-05 | Governance ordering |
| FRZ-EXE-01, FRZ-EXE-02 | Single execution authority |
| FRZ-TEN-* | Tenant continuity (local PASS architecture; no global promotion) |

---

## Tests (this child)

```text
uv run pytest -p no:xdist \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_production_flow_gates.py \
  tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_production_composition_gates.py
```

Optional static gate file: `tests/qualification/trace_x/test_trace_x_p5_r2_p3_r1_r1_handoff_architecture_gates.py` (when added).

---

## Unresolved / audit notes

- **Pure CONFIGURE_EXISTING-only** episodes without UCA acquisition/qualification facts remain **fail-closed** until a future typed post-INT-CONFIG qualification subject is architected (explicitly **not** invented here).
- P3-R1 host composition doc still references marketplace-private dispatch; superseded for dispatch ownership by this lock.
- `build_production_marketplace_qualified_capability_execution_dispatch` remains in tree until implementation wave removes it.

**Status:** `TRACE-X-P5-R2-P3-R1-R1` = **READY FOR AUDIT**
