# PROD-Q-R1-R7-Q-C-DIM16-F02-ADR-01 — Fail-Closed Physical Execution Admission on Strict Session Shutdown

| Field | Value |
|-------|-------|
| **Status** | **DECISION CANDIDATE — NOT APPROVED** · **READY FOR AUDIT** (architecture preparation only) |
| **Implementation** | **BLOCKED — ARCHITECTURE DECISION REQUIRED** until operator ratifies this record |
| **Revision** | **R0** |
| **Date** | 2026-10-17 |
| **Branch** | `development` |
| **Source SHA (exact)** | `4def1fde38b25a56b9defc4453ffded46545c456` |
| **Audited baseline** | `4def1fde38b25a56b9defc4453ffded46545c456` (= **HEAD** at authoring) |
| **Parent program** | **PROD-Q-R1-R7-Q** → **PROD-Q-R1-R7-Q-C** (Package D) |
| **Controlling task** | `PROD-Q-R1-R7-Q-C-DIM16-F02-FAIL-CLOSED-ADMISSION-CONTRACT-DECISION` |
| **Parent architecture** | [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) (**R6** — DEC-09 physical A1 owner + coordinator admission; **does not** specify dual-failure shutdown ordering) |
| **Qualification evidence** | [`PROD_Q_R1_R7_Q_C_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_C_QUALIFICATION_EVIDENCE.md) §E · [`PROD_Q_R1_R7_Q_C_PACKAGE_D_INDEPENDENT_ACCEPTANCE_READINESS.md`](PROD_Q_R1_R7_Q_C_PACKAGE_D_INDEPENDENT_ACCEPTANCE_READINESS.md) §5 |
| **Does not close** | **PROD-Q-R1-R7-Q** · **PROD-Q-R1-R7** · **PROD-Q** · **DIM16** · any global **FRZ-*** PASS |
| **Production delta (this task)** | **0** |

**Competing ADR check:** No accepted ADR defines fail-closed physical execution admission when **both** sanctioned session-close drain hooks fail. [`ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md`](../architecture/ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md) governs **recovery resume admission**, not strict serving shutdown. This record is the bounded child decision candidate; it does **not** supersede **PROD-Q-R1-R7-ADR-01** deployment authority.

---

## 1. Confirmed defect @ source SHA

`StrictProductionServingSession.close()` (`intergrax/applications/_shared/strict_production_serving_launcher.py`):

1. Invokes `begin_colocated_serving_execution_drain_on_strict_session_close` (revision-store–dependent Apps composition helper).
2. On `ColocatedServingAdmissionTopologyError` / `VenvBundleColocatedDeploymentError`, invokes `revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close`.
3. When **both** raise, appends errors but still calls `close_process_resources()` if cleanup does not fail.
4. Sets `self._closed = True` **only** when `shutdown_errors` is empty — dual failure leaves `session._closed is False` while `lifecycle._process_closed is True`.
5. Does **not** establish a third, independent physical admission-deny path.

**Negative qualification (committed, failing):**

- `tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py::test_c_dim16_f02_post_close_01_execution_deny_after_dual_shutdown_failure`
- Provenance: F02-shaped close with patched drain + emergency revoke failures; post-close HTTP **200** + inner delegate invoked; expected **502** and no inner execution.
- `test_c_dim16_f02_drain_and_emergency_revocation_both_fail` — documents `unit.admission.admission_closed() is False` after dual failure.

**Invariant gap:**

> Once strict serving shutdown begins, no new execution may be admitted to the retiring serving session or physical serving unit, even if normal drain and emergency revocation operations fail.

---

## 2. Before ownership / composition / execution graph

```text
StrictProductionServingSession.close()          [Apps — composition owner for session shutdown]
  ├─ begin_colocated_serving_execution_drain_on_strict_session_close
  │     └─ revision_store.get_revision → deployment_adapter.begin_drain(revision, unit_ref)
  │           └─ VenvBundle…begin_drain_for_bound_serving_unit → _UnitAdmissionGate.begin_drain()
  ├─ (on failure) revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close
  │     └─ revision present → same begin_drain path; revision absent → begin_drain_for_bound_serving_unit only
  ├─ ProductionProcessLifecycle.close_process_resources()   [Apps — process slot / PIS / stores]
  └─ session._closed := (shutdown_errors empty)

HTTP POST → ColocatedServingUnitAdmissionTaskExecutionPort.execute   [Apps — per-request wrapper]
  └─ StoreBackedColocatedServingUnitExecutionAdmission.admit_execution
        └─ ActivationDomainCoordinator.coordinate_physical_execution_reservation
              └─ verify_serving_execution_eligibility_from_state + physical_reserve()
                    └─ VenvBundleColocatedUnitPhysicalExecutionGate.reserve_execution
                          └─ adapter.reserve_unit_execution → _UnitAdmissionGate.admit_execution
```

**Semantic owners (unchanged @ HEAD):**

| Concern | Owner | Contract |
|---------|-------|----------|
| Physical execution admission (in-flight gate) | **Agent Distribution** — `VenvBundleColocatedRuntimeDeploymentAdapter` + `_UnitAdmissionGate` | Adapter-owned; checked in `reserve_unit_execution` |
| Canonical serving eligibility | **AD** — `ActivationDomainCoordinator` + `verify_serving_execution_eligibility_from_state` | `ServingExecutionAdmissionRequest` |
| Session shutdown orchestration | **Apps** — `StrictProductionServingSession.close` | Public session API |
| Per-request execution handoff | **Apps** — `ColocatedServingUnitAdmissionTaskExecutionPort` | `ServingRuntimeHandoff.colocated_execution_admission` |
| Process lifecycle / slot | **Apps** — `ProductionProcessLifecycle` | Not an execution admission authority |

---

## 3. Decision questions (D01–D12) — evidence @ SHA

| ID | Answer |
|----|--------|
| **D01 Authority** | **AD** physical gate (`_UnitAdmissionGate`, `reserve_unit_execution`) is the **only** mechanism that denies new physical in-flight slots. Coordinator eligibility is necessary but **not sufficient** when the gate remains open (`serving_admission_authorized` + open gate still admit). |
| **D02 Composition** | **Apps** `StrictProductionServingSession.close` invokes AD drain/revoke helpers; it must not become execution authority. |
| **D03 Ordering** | **Before** new execution can be denied after failed hooks: `_UnitAdmissionGate._admission_closed` must be **True** (or equivalent protocol-level deny). **Before** `close_process_resources()`: fail-closed physical admission closure for the **bound** `serving_unit_ref` must be **attempted** and **must not depend** on revision-store success alone. Process cleanup must not imply admission closure today. |
| **D04 Fail-closed** | When drain **and** emergency revoke raise, an **independent** bound-unit physical admission closure must still run (adapter-owned), errors aggregated, admission remain closed for new requests. |
| **D05 Public contract** | `RuntimeDeploymentAdapter.begin_drain` / unit-scoped drain semantics + `ServingRuntimeHandoff` identity (`serving_unit_ref`); **not** `ProductionProcessLifecycle._process_closed` as HTTP guard. Optional: documented session-close ordering on `StrictProductionServingSession` (Apps public type). |
| **D06 In-flight** | Operations already admitted via `_UnitAdmissionGate` / `ColocatedUnitExecutionReservation` complete under existing drain semantics (`check_drain`, in-flight count); shutdown begin-deny applies to **new** reservations only. |
| **D07 Concurrency** | Linearization: `_UnitAdmissionGate` lock + coordinator `run_under_activation_domain` / SQLite IMMEDIATE snapshot for eligibility; race: request between shutdown start and gate close — **fix must close gate before or atomically with** shutdown entry (adapter lock), not after fallible revision lookup. |
| **D08 Recovery** | After dual failure + `close_process_resources`, `session._closed` is False but `_process_closed` True — **retry `close()` is rejected**; ASGI lifespan may skip `session.close()`. New session must **not** inherit stale handoff; new activation/open path required. |
| **D09 Ownership release** | **Yes, today** `close_process_resources()` can run while gate open — **not permitted** under target invariant; independent gate enforcement required. |
| **D10 Errors** | Preserve `ExceptionGroup` / chained causes for drain, revoke, cleanup; **do not** set `_closed = True` on partial admission failure; do not swallow original shutdown errors. |
| **D11 Isolation** | `begin_drain_for_bound_serving_unit(serving_unit_ref)` is unit-scoped (`test_c_dim16_f06_emergency_revocation_unit_isolation`). |
| **D12 Replaceability** | Behavior must be expressible via **`RuntimeDeploymentAdapter`** unit drain port, not private `_UnitAdmissionGate` reads from Apps. |

---

## 4. Candidate comparison (A / B / C)

### Candidate A — Harden existing physical admission authority (AD)

| Dimension | Assessment |
|-----------|------------|
| Semantic owner | AD adapter + gate (unchanged) |
| Composition owner | Apps still calls adapter |
| Public contract | Extend/document adapter: **idempotent fail-closed admission close** is first effect of any shutdown drain entry |
| Layer boundary | **Low** — stays in AD |
| Authority changes | **None** |
| Concurrency | Gate lock already serializes admit vs `begin_drain` |
| Failure/recovery | If adapter unreachable/raises before gate mutation, gap remains unless orchestration also guarantees bound-unit close |
| In-flight | Unchanged |
| Replaceability | **Strong** — protocol-level |
| Production files | `venv_bundle_colocated_runtime_deployment_adapter.py`; possibly `deployment.py` docstring |
| Regression | F02-POST-CLOSE-01; dim16-01/04/06/F03 |
| Blockers | **Composition layer can fail before adapter** (missing revision in `begin_colocated_serving_execution_drain_on_strict_session_close`); patches in qual simulate total helper failure — production needs orchestration ordering **with** A |
| ADR | Amendment to deployment drain semantics under **ADR-01** footprint |

### Candidate B — Strengthen session-close orchestration (Apps)

| Dimension | Assessment |
|-----------|------------|
| Semantic owner | AD gate (unchanged); Apps orders calls |
| Composition owner | `StrictProductionServingSession.close` + `colocated_serving_runtime_composition.py` |
| Public contract | **Session-close ordering invariant** documented; new sanctioned helper e.g. `close_bound_colocated_physical_admission_fail_closed_on_strict_session_close` calling `begin_drain_for_bound_serving_unit` **first** |
| Layer boundary | **Low** — Apps invokes public adapter API only |
| Authority changes | **None** (no HTTP guard, no second gate) |
| Concurrency | Must call bound-unit close **before** `close_process_resources` and **before** fallible revision path |
| Failure/recovery | Gate closed even when revision drain fails; errors still raised for operator visibility |
| In-flight | Unchanged |
| Replaceability | Uses `RuntimeDeploymentAdapter` / bound unit ref from handoff |
| Production files | `strict_production_serving_launcher.py`, `colocated_serving_runtime_composition.py` |
| Regression | F02-POST-CLOSE-01 (primary) |
| Blockers | Requires operator approval of **ordering contract**; `isinstance(..., VenvBundle...)` in helpers today — replaceability gap already present @ HEAD |
| ADR | **This record (mandatory)** |

### Candidate C — Couple store-backed admission to lifecycle termination

| Dimension | Assessment |
|-----------|------------|
| Semantic owner | **Ambiguous** — Apps admission reads lifecycle/session private flags |
| Composition owner | Apps |
| Public contract | Would require new typed “serving session terminating” port on handoff |
| Layer boundary | **High** — cross-layer lifecycle import or private field |
| Authority changes | **Risk of shadow authority** at HTTP/task layer |
| Concurrency | TOCTOU unless same linearization as physical gate |
| Failure/recovery | `_process_closed` without gate close = false deny |
| Replaceability | **Weak** — lifecycle coupling |
| Blockers | Violates D05/D13; **not recommended** |
| ADR | Would require broader contract — **rejected as primary** |

### Comparison summary

| | A | B | C |
|---|:---:|:---:|:---:|
| Preserves single physical admission owner | ✓ | ✓ | ✗ |
| Closes gap when composition helpers fail before adapter | partial | **✓** | ✗ |
| No HTTP-specific guard | ✓ | ✓ | ✓ |
| No duplicate gate | ✓ | ✓ | ✗ |
| Replaceability | ✓ | ✓ (with existing adapter pin) | ✗ |

---

## 5. Proposed preferred decision (candidate — not approved)

**Prefer Candidate B**, implemented **together with** minimal **Candidate A** hardening:

1. **Ordering (B):** At the start of `StrictProductionServingSession.close` (when `handoff.colocated_execution_admission` is present), invoke a sanctioned Apps helper that calls **only** `deployment_adapter.begin_drain_for_bound_serving_unit(handoff.identity.serving_unit_ref)` (or protocol equivalent) **before** revision-dependent drain and **before** `close_process_resources()`. Treat admission closure as **must-attempt**; aggregate errors from subsequent canonical drain/revoke without reopening admission.

2. **Idempotency (A):** Adapter `begin_drain_for_bound_serving_unit` remains safe to call multiple times; `begin_drain` on revision path must not reopen admission after closure.

3. **Explicit non-goals:** No new HTTP 502 branch; no check of `_process_closed` in `ColocatedServingUnitAdmissionTaskExecutionPort`; no coordinator bypass.

**If operator rejects ordering amendment:** status remains **STOP — ARCHITECTURE DECISION REQUIRED**; F02 blocker persists.

---

## 6. Public contract consequences (if ratified)

| Contract | Change |
|----------|--------|
| `StrictProductionServingSession.close()` | Documented: **fail-closed physical admission closure** for bound unit precedes process resource cleanup; dual hook failure does not leave admission open. |
| `begin_colocated_serving_execution_drain_on_strict_session_close` | May become second-phase **lifecycle/drain evidence** rather than sole admission closer. |
| `RuntimeDeploymentAdapter.begin_drain` | Clarify: unit-scoped admission deny is mandatory effect; revision argument is correlation, not admission prerequisite. |
| `ServingRuntimeHandoff` / coordinator | **No** new execution authority; eligibility rules unchanged. |

---

## 7. Concurrent request vs shutdown

| Phase | Behavior (target) |
|-------|-------------------|
| Request wins race before shutdown | Admitted under existing coordinator + gate rules |
| Shutdown starts | Bound-unit gate closed under adapter lock → subsequent `reserve_unit_execution` raises → `ServingRuntimeHandoffError` → HTTP **502** (existing mapping) |
| Request during drain errors | Gate already closed → deny even if revision drain raises |

**Linearization point:** `_UnitAdmissionGate._lock` inside `begin_drain` / `admit_execution`, invoked from adapter before fallible Apps composition returns.

---

## 8. In-flight execution

Existing drain semantics: in-flight work decrements via `release_execution`; `check_drain` observes in-flight count. Shutdown **denies new** admissions; does **not** require killing in-flight tasks in this slice (aligned with `test_c_dim16_01`, D1 drain tests).

---

## 9. Recovery and retry

| State after dual failure | Current HEAD | Target |
|--------------------------|--------------|--------|
| `session._closed` | False | False (close incomplete) |
| `lifecycle._process_closed` | True | True |
| Admission gate | **Open (defect)** | **Closed** |
| Retry `close()` | Rejected | Rejected — operator must open **new** session |
| Reuse unit | Qual uses same adapter instance — production must not re-authorize without new activation |

---

## 10. Error propagation

Preserve two-error `ExceptionGroup` for F02; add admission-close failure as additional context only if bound-unit close itself fails. **F03** triple-failure ordering unchanged in intent. Do not mark session closed successfully when shutdown_errors non-empty.

---

## 11. Tenant and serving-unit isolation

Fail-closed closure is keyed by `handoff.identity.serving_unit_ref` only (`test_c_dim16_f06`). Tenant binding on HTTP path unchanged (`StrictProductionServingTenantAuthorityTaskExecutionPort`).

---

## 12. Replaceability

Target implementation must route through **`RuntimeDeploymentAdapter`** drain APIs, not direct `_UnitAdmissionGate` mutation from Apps. Long-term: reduce `isinstance(VenvBundleColocatedRuntimeDeploymentAdapter)` in composition helpers via protocol extension — **tracked freeze debt**, not in this decision’s implementation scope.

---

## 13. F02 exit-criteria matrix (F02-01 … F02-15)

| ID | Requirement | Historical evidence @ SHA | After fix evidence needed |
|----|-------------|---------------------------|---------------------------|
| F02-01 | Normal shutdown denies new executions | **VALID** — `test_c_dim16_01` | Re-run @ impl SHA |
| F02-02 | Drain fail + successful emergency revoke denies | **VALID** — `test_c_dim16_04` | Re-run |
| F02-03 | Drain + revoke fail still denies admission | **FAIL** — F02 unit + POST-CLOSE-01 | **NEW** — same tests must PASS |
| F02-04 | HTTP after failed close never reaches inner | **FAIL** — POST-CLOSE-01 | **NEW** |
| F02-05 | Concurrent shutdown/request ordering documented | **NOT PROVEN** | Qual + §7 record |
| F02-06 | In-flight policy explicit | **PARTIAL** — dim16-01 drain state | Reconfirm |
| F02-07 | Cleanup fail preserves deny invariant | **NOT PROVEN** — F03 focuses errors | F03 + admission assert extension |
| F02-08 | Repeated/partial close cannot resurrect admission | **PARTIAL** — dim16-07 reopen | Re-run + new asserts |
| F02-09 | New session no stale authorization | **NOT PROVEN** at DIM16 depth | P2 identity continuity reuse |
| F02-10 | Sibling units unaffected | **VALID** — `test_c_dim16_f06` | Re-run |
| F02-11 | Tenant boundary unchanged | **VALID** — P3-A scoped | No regression |
| F02-12 | All shutdown errors observable | **VALID** — F02/F03 ExceptionGroup | Preserve |
| F02-13 | No second admission authority | **Design constraint** | Architecture gate review |
| F02-14 | Public contracts + replaceability | **PARTIAL** — VenvBundle isinstance | Document amendment |
| F02-15 | Regression auditable @ exact SHA | **VALID** — failing test committed | PASS @ impl SHA |

---

## 14. Qualification-boundary debt (autouse fixture)

`test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py` — `@pytest.fixture(autouse=True) _isolate_production_process_lifecycle_slot` reads `production_process_lifecycle._active_production_process_lifecycle` and `_process_closed` (**private module slot**).

| Item | Disposition |
|------|-------------|
| Classification | **TRACKED FREEZE DEBT** (qualification boundary) |
| This task | **Do not expand** private-slot pattern |
| Implementation dependency | **Not blocking** — F02 POST-CLOSE uses `_dim10_http_fixture` + public `session.close()`; fixture debt is parallel hygiene (cf. D4-R1) |
| Future | Owned session registry or sanctioned qual boundary per strict HTTP evidence docs |

---

## 15. Proposed implementation file scope (post-ratification only)

| File | Role |
|------|------|
| `intergrax/applications/_shared/strict_production_serving_launcher.py` | Order fail-closed admission step |
| `intergrax/applications/_shared/colocated_serving_runtime_composition.py` | Sanctioned bound-unit fail-closed helper |
| `intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py` | Idempotent admission close (if needed) |
| `intergrax/agent_distribution/deployment.py` | Protocol documentation only (optional) |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py` | F02/F02-POST-CLOSE-01 must PASS |

**Forbidden:** HTTP router changes as primary fix; private gate mutation from Apps; new admission owner.

---

## 16. Operator decision required

| # | Question |
|---|----------|
| 1 | Ratify **Candidate B + minimal A** ordering as canonical session-close semantics? |
| 2 | Accept **contract amendment** to `StrictProductionServingSession.close` / drain helper responsibilities? |
| 3 | Authorize bounded implementation child (single pytest module replay, test-process budget per operator rules)? |

---

## 17. Applicable FRZ (scoped — no promotion)

| FRZ | Mapping @ SHA |
|-----|----------------|
| **FRZ-PRD-05** | **FAIL / OPEN** for full shutdown qualification — dual-failure admission gap breaks “shutdown/resource cleanup qualified” at Package D depth |
| **FRZ-PRD-08** | **PARTIAL** — post-close execution reaches inner path (bypass of intended physical deny) |
| **FRZ-REG-03** | **PASS (scoped)** — negative test **detects** defect (failing regression committed); fulfillment requires test PASS after fix |
| **FRZ-SEC-04** | **NOT PROVEN** for this slice — no new auth bypass; execution after failed shutdown is reliability/admission |
| **FRZ-REL-*** | **PARTIAL** — resource release vs admission inconsistency |
| **FRZ-TEN-*** | **NOT IMPACTED** — unit/tenant scope unchanged |
| **FRZ-CTR / OWN / RPL** | **PASS (design)** — no new owner if B+A ratified |

---

## 18. Enterprise audit matrix @ SHA

| Criterion | Result |
|-----------|--------|
| Hard layer boundaries | **PASS** (design target) |
| Communication direction | Apps → AD ports; no reverse authority |
| Exactly-one semantic ownership (admission) | **PASS** (AD physical gate) |
| Exactly-one composition ownership (session close) | **PASS** (Apps session) |
| Contracts over implementations | **PARTIAL** — VenvBundle isinstance in close helpers |
| Strong typed boundaries | **PASS** |
| Pluginability | **PARTIAL** — strict A1 pin |
| Structural replaceability | **PARTIAL** |
| No bypasses / shadow authorities | **FAIL** @ HEAD — open gate after shutdown |
| Fail-closed admission | **FAIL** @ HEAD for F02 path |
| Tenant / unit isolation | **PASS** |
| Observability non-authoritative | **PASS** |
| Traceability / evidence continuity | **PASS** — SHA + failing test |
| Regression protection | **PASS** — committed negative |
| Recovery contracts | **PARTIAL** — session retry blocked |

---

## 19. Roadmap disposition (canonical tracker)

| Item | Recommended status |
|------|-------------------|
| This ADR slice | **READY FOR AUDIT** |
| **PROD-Q-R1-R7-Q-C** Package D | Remains **READY FOR AUDIT** with **DIM16/F02 IN-SCOPE BLOCKER** |
| **PROD-Q** | **CURRENT / BLOCKED** (unchanged) |
| Next proposed action | Operator ratifies **ADR-01**; then bounded implementation child with **one** targeted `pytest` on dim16 module |

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
