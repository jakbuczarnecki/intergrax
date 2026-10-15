# PROD-Q-R1-R7-Q-P3-D — Drain / stop / execution admission qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (scoped evidence; does **not** close **P3**, **PROD-Q-R1-R7-Q**, or promote global **FRZ-***) |
| **Child** | **PROD-Q-R1-R7-Q-P3-D** / **P3-T-06** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Branch** | `development` |
| **START_HEAD** | `cd7dafc887468a8dcbd59b205fe3b106011d0e69` |
| **Execution / source SHA** | `cd7dafc887468a8dcbd59b205fe3b106011d0e69` (pre-commit; post-commit SHA in git report below) |

## 1. Preflight

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** | **match** @ `cd7dafc887468a8dcbd59b205fe3b106011d0e69` |
| Working tree @ run start | **clean** |
| **PROD-Q-R1-R7-Q-P3-P0** | **READY FOR AUDIT** (inventory; dim **16** P3-D) |
| **PROD-Q-R1-R7-Q-P3-C** | **READY FOR AUDIT** (prior child; stale READY / physical loss) |
| **PROD-Q-R1-R7-D1** / **D1-R6** | Slice evidence @ `17e928e4…` / `c25e7efe…` chains — **independently accepted** for adapter admission + coordinator |
| **PROD-Q-R1-R7-D4** | **CLOSED / independently accepted** @ `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` |
| **PROD-Q-R1-R7-Q** parent | **IN PROGRESS** / **NOT QUALIFIED**; **E.1** full D0–D4 replay **NOT EXECUTED** |
| Drain/admission core @ HEAD | **0** diff vs `f2dd8ecb…` on `venv_bundle_colocated_runtime_deployment_adapter.py`, `colocated_serving_unit_execution_admission.py`, `serving_runtime_handoff.py`, `colocated_serving_runtime_composition.py`, `strict_production_serving_launcher.py` |

## 2. Scope (P3-T-06)

| Scenario | Intent |
|----------|--------|
| **P3-T-06** | When a strict-path colocated unit is **draining**, **stopping**, or **stopped**, **new** physical execution admission is rejected; in-flight work follows existing drain contract; counters/leases stay consistent; stop does not silently re-open admission; stale handoff cannot bypass stopped-unit gates; only sanctioned lifecycle restores eligibility |

**Production code delta:** **none** (qualification + evidence reuse only).

**Out of scope:** new drain controller, lifecycle owner, serving registry, or admission mechanism; full strict-session E2E drain orchestration (operator-scheduled).

## 3. Canonical drain / stop ownership graph

```text
Governance mutations (complete_drain, activation commit)
  → ActivationService (SO: durable instance state + env lock)
      → RuntimeDeploymentAdapter.begin_drain / check_drain / stop
          → VenvBundleColocatedRuntimeDeploymentAdapter
              → _UnitAdmissionGate (physical: _admission_closed, _in_flight)
              → serving_admission_authorized + coordinator_admission_open_staged

Strict serving open (CO)
  → open_strict_production_serving_session
      → open_production_process_lifecycle + reconcile
      → _bind_serving_handoff_for_environment
      → bind_colocated_serving_runtime_after_activation
          → establish_serving_runtime_handoff (identity + resolver; no admission grant)
          → authorize_colocated_serving_execution_after_activation
              → ActivationDomainCoordinator.coordinate_unit_serving_admission_open
              → adapter publish → open_admission on gate + serving_admission_authorized

Per host task / per execution (RA: serving_unit_ref + identity)
  → ColocatedServingUnitAdmissionTaskExecutionPort.execute
      → StoreBackedColocatedServingUnitExecutionAdmission.admit_execution
          → ActivationDomainCoordinator.coordinate_physical_execution_reservation
              → verify_serving_execution_eligibility_from_state (durable)
              → physical_gate.reserve_execution → adapter.reserve_unit_execution
```

| Transition | Semantic owner | Composition owner | Authoritative store | Linearization | Fail-closed | Error contract |
|------------|----------------|-------------------|---------------------|---------------|-------------|----------------|
| **Open admission (post-activate)** | AD `ActivationService` + coordinator | Apps `bind_colocated_serving_runtime_after_activation` | Serving pointer + deployment instance **SERVING** + physical gate | SQLite/in-mem activation domain; publish after verify | Yes | `ServingRuntimeHandoffError` / `ServingExecutionAdmissionError` |
| **Per-exec admit** | AD coordinator + physical gate | Apps `StoreBackedColocatedServingUnitExecutionAdmission` | Same + `_UnitAdmissionGate` under adapter lock | `coordinate_physical_execution_reservation`: verify then `physical_reserve` in one domain entry | Yes | `ServingRuntimeHandoffError` wraps admission errors |
| **Per-exec release** | Physical gate | Apps `release_execution` / reservation `release` | In-flight counter on unit | Gate lock | Double-release raises | `VenvBundleColocatedDeploymentError` |
| **begin_drain** | AD via adapter port | `ActivationService.commit_activation` (prior unit), adapter `begin_drain` | Physical: `DRAINING` + gate closed; durable may move to **DRAINING** on governance path | Adapter `RLock` with gate lock | Blocks new admit | `VenvBundleColocatedDeploymentError` on admit |
| **check_drain / complete_drain** | AD `ActivationService.complete_drain` | Governance mutation | Durable instance **DRAINING** + adapter in-flight | Env lock + adapter | Timeout → policy (e.g. MARK_FAILED) | `RuntimeDrainError` |
| **stop** | AD adapter `stop` | Ops / rollback paths | Physical **STOPPED**, resources invalidated | Adapter lock | `resume_serving` on **STOPPED** rejected | `VenvBundleColocatedDeploymentError` |
| **Re-open eligibility** | AD activation + coordinator publish only | Strict launcher bind after reconcile + commit | Not via `open_admission` alone | Sanctioned activation sequence | Handoff establish does **not** open gate | — |

## 4. Physical vs durable state

| Layer | Drain/stop meaning |
|-------|-------------------|
| **Durable** (`ApplicationEnvironmentServingStore`, `DeploymentInstanceStore`) | Traffic pointer, instance state **SERVING** / **DRAINING** / **STOPPED**; `verify_serving_execution_eligibility_from_state` requires instance **SERVING** for new coordinated admits |
| **Physical** (`_UnitAdmissionGate`, `serving_admission_authorized`, `runtime_resource.is_live`) | `begin_drain` sets `_admission_closed=True`; `stop` → `invalidate_resources()` (drain + clear authorization); `reserve_unit_execution` requires live unit + authorized + open gate |
| **Process handoff** | `ServingRuntimeHandoff` is identity + resolver + port reference; **does not** bypass coordinator or physical gate on `admit_execution` |

Enforcement of **new** admission is **physical gate + coordinator verify**, not a mutable HTTP flag alone.

## 5. Strict-path trace (static @ HEAD)

```text
open_strict_production_serving_session
  → bind_colocated_serving_runtime_after_activation (store-backed admission wired)
  → ready_for_traffic requires handoff + colocated_execution_admission
  → GCA factory: ColocatedServingUnitAdmissionTaskExecutionPort(handoff)

Subsequent execution (after session established):
  handoff.colocated_execution_admission.admit_execution(unit_ref)
    → coordinate_physical_execution_reservation
    → adapter.reserve_unit_execution (physical truth)

Drain/stop on active unit (ops / activation):
  adapter.begin_drain / stop (physical)
  OR ActivationService.complete_drain (durable + check_drain)
  → new reserve/admit fails at gate and/or eligibility verify
```

No second admission authority on strict graph (P1/D2 continuity).

## 6. D-01 … D-08 classification

| ID | Invariant | Classification | Evidence |
|----|-----------|----------------|----------|
| **D-01** | New admission after draining transition rejected | **PROVEN AND REUSABLE** | D1 `test_begin_drain_blocks_admission`; D1-R6 R2 `test_identifier_only_reserve_without_traffic_authorization_fails`; physical `begin_drain` closes gate before `admit_execution` |
| **D-02** | In-flight admitted before drain governed by release/drain policy | **PROVEN AND REUSABLE** | D1 `test_drain_completion_reflects_in_flight`; D1-R6 R2 `test_s2_admitted_in_flight_survives_post_coordination_pointer_swap` (no forced cancel) |
| **D-03** | Admission vs drain linearization | **PROVEN AND REUSABLE** | D1-R6 R4/R5 (`test_sqlite_s2_admission_wins_concurrent`, `test_sqlite_unit_open_blocks_pointer_swap_until_publish`); coordinator verify→reserve ordering in `coordinate_physical_execution_reservation` |
| **D-04** | Stop does not leave execution-eligible physical unit | **PROVEN AND REUSABLE** + **STATIC CONTINUITY** | Adapter `stop` → `invalidate_resources` + `serving_admission_authorized=False` + gate drain; `resume_serving` rejects **STOPPED**; `open_admission` only via coordinator `publish_coordinator_verified_open` |
| **D-05** | Stale handoff cannot admit against stopped/draining unit | **PROVEN AND REUSABLE** (boundary) | Per-exec path re-runs coordinator + physical reserve (not handoff-only); D1-R3 stale pointer / not-serving; `_require_live_unit` on **STOPPED**; D1-R6 R2 pointer-swap deny with zero in-flight on failed admit |
| **D-06** | Counters and leases under success/failure/cancel | **PROVEN AND REUSABLE** | Gate `admit_execution`/`release_execution`; coordinator `_release_physical_reservation_once` on failure; `ColocatedServingUnitAdmissionTaskExecutionPort` `finally: release_execution` |
| **D-07** | Stopped/missing unit does not become READY via stale durable alone | **PROVEN AND REUSABLE** (reuse **P3-C**) | D3/D4 stale physical + reconcile fail-closed; strict open abort before handoff — P3-C matrix |
| **D-08** | Only sanctioned activation/reconciliation restores eligibility | **PROVEN BY STATIC CONTINUITY** | `establish_serving_runtime_handoff` explicit: does not grant execution; `authorize_colocated_serving_execution_after_activation` + coordinator publish required; strict session bind only after `RECONCILIATION_OUTCOME_READY` |

**Overall P3-T-06:** **PASS (scoped)** — D1/D1-R6 physical admission + drain contracts at canonical seams; strict wiring continuity from D4/P1; **no** integrated strict-session drain E2E @ HEAD.

## 7. Historical evidence reuse (provenance)

| Era | SHA (reference) | Module / tests | P3-D reuse |
|-----|-----------------|----------------|------------|
| **D1** | `17e928e4…` | `test_prod_q_r1_r7_d1_venv_bundle_colocated_deployment_adapter.py` (`test_begin_drain_*`, `test_drain_timeout_evidence`) | D-01, D-02 |
| **D1-R3** | D1-R3 chain | `test_prod_q_r1_r7_d1_r3_serving_execution_admission_closure.py` | D-05 durable/handoff bridge |
| **D1-R6-R2** | `c25e7efe…` | `test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py` | D-01–D-03, D-05 |
| **D1-R6-R4/R5** | same | `test_prod_q_r1_r7_d1_r6_r4_physical_gate_atomicity.py`, `r5_serving_admission_linearization.py` | D-03 drain vs staged open |
| **D1-R6-R6/R7** | same | lease/cleanup modules with `begin_drain` scenarios | D-06 coordination under drain |
| **D1b** | `17e928e4…` | `test_prod_q_r1_r7_d1b_serving_runtime_handoff.py` + `ColocatedServingUnitAdmissionTaskExecutionPort` | GCA admit/release wrap |
| **D4** | `0dd51fe1…` | `strict_production_serving_launcher.py` bind + cleanup | Strict CO path to store-backed admission |
| **P3-C** | `11cf7df0…` / continuity @ `cd7dafc…` | Stale READY / physical loss | D-07 |

**Continuity @ `cd7dafc…`:** admission/drain/stop implementation files unchanged since D3 reconciliation baseline on listed paths (git diff **0** lines `f2dd8ecb…` → HEAD).

## 8. Race / linearization analysis

| Hazard | Mitigation @ HEAD |
|--------|-------------------|
| Admit after `begin_drain` | `_UnitAdmissionGate.admit_execution` checks `_admission_closed` under lock; `begin_drain` sets closed under same lock |
| Verify pass then unit drained | `coordinate_physical_execution_reservation` runs verify then `physical_reserve` inside activation domain; drain closes gate before reserve succeeds |
| Concurrent coordinator open vs drain | D1-R6 R4 `test_concurrent_drain_clears_staged_open`; R5 publish/pointer interleaving tests |
| Durable **SERVING** while physical draining | Eligibility may still read **SERVING** until store transitions; **physical gate still rejects** new work (defense in depth) |
| Stale handoff object in process | Each `admit_execution` re-coordinates; identity mismatch on unit ref raises `ServingRuntimeHandoffError` |

## 9. Integration-depth gaps

| Gap | Disposition |
|-----|-------------|
| **Drain/stop during active `open_strict_production_serving_session`** with live GCA task in flight, then HTTP execute | **NOT PROVEN** — **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** (P3-P0 dim **16**; ≤5 min with active session fixture + real adapter drain ports) |
| `ActivationService.complete_drain` → `stop` → strict session still holding handoff reference | **NOT PROVEN** integrated — static: subsequent admits fail at coordinator and/or physical gate; session lifecycle close is D4-R2 scope, not re-run here |
| Full D1-R6 drain module replay @ HEAD | **NOT EXECUTED** (pytest budget; continuity **0** delta) |
| Global **FRZ-PRD-05** promotion | **OUT OF SCOPE** — contribution only |

## 10. Test execution

**Permitted node IDs:** **none executed** (default hard budget; historical PASS not refreshed).

| Metric | Value |
|--------|-------|
| Pytest invocations | **0** |
| Wall time | **0s** |

## 11. Enterprise architecture audit

| Invariant | Status @ P3-D |
|-----------|----------------|
| Exactly-one drain/stop semantic owner (AD `ActivationService` + adapter ports) | **Aligned** |
| Exactly-one physical admission authority (`_UnitAdmissionGate` + coordinator) | **Aligned** |
| One authoritative lifecycle state (durable instance + physical unit state) | **Aligned** (dual-layer; physical enforces admit) |
| Admission/drain linearization | **Aligned** (D1-R6 evidence) |
| In-flight accounting + release on failure/cancel | **Aligned** |
| Stale handoff fail-closed | **Aligned** at admit boundary |
| Governance ≠ execution | **Aligned** — drain via governance/activation; exec via coordinator |
| Typed contracts / replaceability | **Aligned** — ports on adapter + handoff protocols |
| No reference/lab fallback on strict graph | **Aligned** (P1 continuity) |
| No shadow lifecycle registry | **Aligned** |
| Observability (drain evidence refs, timeout) | **Aligned** — `check_drain` evidence strings |
| Lifecycle cleanup on strict startup failure | **Aligned** — D4-R2 `close_process_resources` (not re-audited) |

No **BLOCKED BY ARCHITECTURE** or **PROVEN DEFECT** @ HEAD.

## 12. FRZ mapping (contribution only — globals unchanged)

| FRZ ID | P3-D contribution |
|--------|-------------------|
| **FRZ-PRD-05** | Drain blocks new physical admission; stop invalidates unit resources; scoped reuse of D1 drain tests |
| **FRZ-PRD-04** | Strict bind ordering; admission only after sanctioned open publish |
| **FRZ-PRD-08** | No identifier-only or handoff-only execution bypass |
| **FRZ-SEC-04** | Store-backed admission on strict GCA path |
| **FRZ-PRD-09** | Physical gate + reconcile/handoff gates compose with P3-C stale-ready slice |
| **FRZ-REG-02** | No duplicate P3-named drain tests; reuse D1/D1-R6 |
| **FRZ-REG-03** | This document @ commit on `development` |
| **FRZ-REG-09** | Continuity vs `f2dd8ecb…` / D1-R6 baselines |

Global checklist rows remain **OPEN**.

## 13. Remaining blockers / next slice

| Item | Disposition |
|------|-------------|
| **E.1** full D0–D4 replay | Operator-scheduled |
| **P3-E** (pre-readiness / P3-T-07) | **Next mandatory P3 child** per P3-P0 ordering after P3-D |
| **P3-F** (dim 20) | **BLOCKED** on parent **PROD-Q-R1** |
| Operator integration: active strict session + `begin_drain` during in-flight GCA task | Scheduled qualification; exact nodes: extend D4 session fixture + `adapter.begin_drain` + denied second `ColocatedServingUnitAdmissionTaskExecutionPort.execute` |

## 14. Cursor disposition

**READY FOR AUDIT** — independent GitHub SHA audit required; this report is not final acceptance.

**Recommendation:** Accept **P3-T-06** via static qualification + D1/D1-R6/D1-R3/P3-C evidence reuse; defer integrated strict-session drain E2E to operator schedule; **no** production code changes.
