# PROD-Q-R1-R7-Q-C-DIM16-F02-ADR-01 — Fail-Closed Physical Execution Admission on Strict Session Shutdown

| Field | Value |
|-------|-------|
| **Status** | **DECISION CANDIDATE — NOT APPROVED** · **READY FOR AUDIT** (architecture preparation only) |
| **Implementation** | **BLOCKED** until operator ratifies this record |
| **Revision** | **R1** (contract closure — `PROD-Q-R1-R7-Q-C-DIM16-F02-ADR-01-R1-CONTRACT-CLOSURE`) |
| **Date** | 2026-10-17 |
| **Branch** | `development` |
| **Source SHA (exact)** | `d3be032d4ddbbbbb0f2b00ffbd93bfd38c1f3d11` |
| **Audited baseline** | `d3be032d4ddbbbbb0f2b00ffbd93bfd38c1f3d11` (= **HEAD** @ R1 amendment) |
| **Prior revision** | **R0** @ `4def1fde38b25a56b9defc4453ffded46545c456` — preferred **B + minimal A** without protocol-level admission-close port or first-deny-failure semantics |
| **Parent program** | **PROD-Q-R1-R7-Q** → **PROD-Q-R1-R7-Q-C** (Package D) |
| **Parent architecture** | [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) (**R6** — DEC-09 physical A1 owner + coordinator admission) |
| **Qualification evidence** | [`PROD_Q_R1_R7_Q_C_QUALIFICATION_EVIDENCE.md`](PROD_Q_R1_R7_Q_C_QUALIFICATION_EVIDENCE.md) §E · [`PROD_Q_R1_R7_Q_C_PACKAGE_D_INDEPENDENT_ACCEPTANCE_READINESS.md`](PROD_Q_R1_R7_Q_C_PACKAGE_D_INDEPENDENT_ACCEPTANCE_READINESS.md) §5 |
| **Does not close** | **PROD-Q-R1-R7-Q** · **PROD-Q-R1-R7** · **PROD-Q** · **DIM16** · any global **FRZ-*** PASS |
| **Production delta (this task)** | **0** |

**Competing ADR check:** No accepted ADR supersedes this bounded child. Recovery admission remains [`ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md`](../architecture/ADR_ENTERPRISE_CHECKPOINT_CONSISTENCY_AND_RECOVERY_ADMISSION.md).

---

## R1 — Unresolved findings closed in this revision

| ID | R0 gap | R1 resolution |
|----|--------|----------------|
| **U1** | `begin_drain_for_bound_serving_unit` is concrete-adapter API; **not** guaranteed by `RuntimeDeploymentAdapter` | **Protocol extension (§5)** — typed `close_serving_unit_execution_admission`; venv `begin_drain_for_bound_serving_unit` becomes implementation detail |
| **U2** | Fail-closed ordering implied gate closure when composition helpers fail, but not when **admission-close itself** fails before gate mutation | **Explicit first-deny-failure policy (§8)** — no claim of deny without successful port completion; cleanup gated on successful admission close |

**R0 retained:** Apps orchestration ordering (**B**), single AD physical gate owner, no HTTP guard, no second admission authority, session incomplete on shutdown errors.

---

## 1. Confirmed defect @ source SHA

`StrictProductionServingSession.close()` (`intergrax/applications/_shared/strict_production_serving_launcher.py`):

1. Calls revision-dependent `begin_colocated_serving_execution_drain_on_strict_session_close` first.
2. On failure, calls `revoke_colocated_serving_execution_admission_fail_closed_on_strict_session_close`.
3. When **both** raise (including before adapter is reached), still runs `close_process_resources()` if cleanup succeeds.
4. Sets `_closed = True` only when `shutdown_errors` is empty — dual helper failure leaves `session._closed is False` while `lifecycle._process_closed is True`.
5. No **independent**, revision-agnostic, **protocol-level** admission-close step precedes fallible helpers.

**Negative qualification (committed, failing @ SHA):**

- `tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py::test_c_dim16_f02_post_close_01_execution_deny_after_dual_shutdown_failure`
- `test_c_dim16_f02_drain_and_emergency_revocation_both_fail` — `unit.admission.admission_closed() is False` after dual helper failure.

**Invariant (target, post-ratification implementation):**

> After strict serving shutdown **begins** (§7), no new physical execution reservation may succeed for the bound `serving_unit_ref`, even when canonical drain and emergency revocation helpers fail.

**Not claimed @ HEAD or by this ADR alone:** deny before the enforceable admission-close port completes successfully.

---

## 2. Before / target ownership and dependency graph

### 2.1 Before (@ `d3be032d…`)

```text
StrictProductionServingSession.close()                    [Apps — composition owner]
  ├─ begin_colocated_serving_execution_drain_on_strict_session_close
  │     ├─ isinstance(VenvBundleColocatedRuntimeDeploymentAdapter)
  │     ├─ revision_store.get_revision (may raise before adapter)
  │     └─ RuntimeDeploymentAdapter.begin_drain(revision, serving_unit_ref)
  │           └─ VenvBundle…begin_drain_for_bound_serving_unit → _UnitAdmissionGate.begin_drain()
  ├─ (on failure) revoke_colocated…fail_closed…
  │     └─ revision path OR begin_drain_for_bound_serving_unit (non-protocol)
  ├─ ProductionProcessLifecycle.close_process_resources()
  └─ session._closed := (shutdown_errors empty)

HTTP POST → ColocatedServingUnitAdmissionTaskExecutionPort
  └─ StoreBackedColocatedServingUnitExecutionAdmission.admit_execution
        └─ ActivationDomainCoordinator.coordinate_physical_execution_reservation
              └─ VenvBundleColocatedUnitPhysicalExecutionGate → reserve_unit_execution
                    └─ _UnitAdmissionGate.admit_execution
```

### 2.2 Target (after ratified implementation)

```text
StrictProductionServingSession.close()
  ├─ [NEW FIRST] close_colocated_serving_execution_admission_on_strict_session_close
  │     └─ RuntimeDeploymentAdapter.close_serving_unit_execution_admission(serving_unit_ref)
  │           └─ adapter-owned gate deny (revision-independent)
  ├─ begin_colocated_serving_execution_drain_on_strict_session_close   [drain evidence / lifecycle]
  ├─ (on failure) revoke_colocated…fail_closed…   [secondary; must not reopen admission]
  ├─ close_process_resources()   [only if admission close succeeded — §8.3]
  └─ session._closed per existing error aggregation

Drain completion (orthogonal):
  RuntimeDeploymentAdapter.check_drain(revision, serving_unit_ref, policy) → in-flight observation
```

| Concern | Semantic owner | Sanctioned composition owner |
|---------|----------------|------------------------------|
| Physical execution admission (in-flight gate) | **Agent Distribution** — `_UnitAdmissionGate` via `VenvBundleColocatedRuntimeDeploymentAdapter` | N/A (authority, not orchestration) |
| Admission-close port surface | **Agent Distribution** — `RuntimeDeploymentAdapter` extension | Injected `deployment_adapter` on lifecycle |
| Session shutdown orchestration | N/A | **Apps** — `StrictProductionServingSession.close` |
| Per-request execution handoff | **Apps** — `ColocatedServingUnitAdmissionTaskExecutionPort` | GCA strict factory |
| Coordinator eligibility | **AD** — `ActivationDomainCoordinator` | Apps admission wrapper |

**Governing deployment / physical admission authority:** [`PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md`](PROD_Q_R1_R7_ADR_01_PRODUCTION_DEPLOYMENT_AUTHORITY.md) DEC-01 A1 + DEC-09 coordinator ordering; this ADR is a **bounded child** for strict session shutdown only.

---

## 3. Protocol vs capability decision (A / B)

| Option | Verdict |
|--------|---------|
| **A — Extend `RuntimeDeploymentAdapter`** with a dedicated, revision-independent **admission-close** operation | **SELECTED** |
| **B — Separate AD capability Protocol** composed beside deployment adapter | **REJECTED** for this slice |

**Rationale:**

- Physical admission is already **deployment-adapter-owned** under parent ADR-01; a parallel capability contract would duplicate semantic ownership and invite shadow composition paths.
- `begin_drain(revision, …)` is **revision-correlated** and today conflates admission closure with DRAINING lifecycle side effects on the venv backend; strict shutdown needs **admission closure without durable revision lookup**.
- **Structural replaceability (FRZ-RPL-*):** one topology-neutral port with typed unsupported-topology failure beats `isinstance` + concrete `begin_drain_for_bound_serving_unit` for the **authoritative** deny transition.
- **Rejected:** forcing unrelated backends to fake in-process gates — backends **without** unit-scoped execution admission implement the port as **unsupported** (typed error) or documented no-op only where parent ADR already defines equivalent external behavior (not in strict A1 venv path).

**Apps orchestration (R0 Candidate B)** remains mandatory: composition calls the **new port first**, before fallible revision-store helpers — this is ordering, not a second admission owner.

---

## 4. Final contract shape (typed API)

### 4.1 Admission closure (new)

```python
def close_serving_unit_execution_admission(
    self,
    *,
    serving_unit_ref: str,
) -> None:
    """Deny new physical execution reservations for this serving unit; idempotent."""
```

| Field | Semantics |
|-------|-----------|
| **Semantic owner** | AD physical gate (`_UnitAdmissionGate.begin_drain` effect on venv A1) |
| **Composition owner** | Apps session-close helper → `deployment_adapter` |
| **Preconditions** | `serving_unit_ref` identifies bound unit for active strict handoff; adapter holds unit record (venv) or backend equivalent |
| **Postconditions** | New `reserve_unit_execution` / `admit_execution` **must** fail closed for that unit; **does not** by itself complete drain |
| **Idempotency** | Repeated calls are no-ops at gate level |
| **Failure semantics** | Propagate `VenvBundleColocatedDeploymentError` or port-specific deployment errors; **no gate mutation** if call fails before adapter commits deny |
| **Backend obligations** | Venv A1: set `_admission_closed = True` under gate lock; may set DRAINING metadata consistent with existing `begin_drain_for_bound_serving_unit` |
| **Unsupported topology** | Raise typed `ColocatedServingAdmissionTopologyError` (or sanctioned deployment error) — strict colocated path already requires venv adapter @ HEAD |
| **Tenant / unit scope** | `serving_unit_ref` only; no cross-unit effect (`test_c_dim16_f06`) |

### 4.2 Drain completion (existing, clarified)

| Operation | Role |
|-----------|------|
| `begin_drain(revision, serving_unit_ref)` | Lifecycle / evidence path; **must not reopen** admission after `close_serving_unit_execution_admission`; on venv, delegates admission close if not already closed |
| `check_drain(revision, serving_unit_ref, policy)` | **Observation only** — in-flight count, timeout policy; independent of whether admission already closed |
| `stop` / `resume_serving` | Unchanged parent ADR semantics; `resume_serving` **must not** reopen admission without sanctioned coordinator re-authorization |

### 4.3 Conflation forbidden

Admission closure and drain completion **must not** be merged into a single Apps concept. Session close **must** invoke admission close explicitly; drain helpers remain for revision-correlated evidence and emergency paths.

---

## 5. Impact on adapters and callers

| Component | Change |
|-----------|--------|
| `deployment.py` | Add protocol method + document separation |
| `FakeInMemoryRuntimeDeploymentAdapter` | Implement deny bit per `serving_unit_ref` (test double) |
| `VenvBundleColocatedRuntimeDeploymentAdapter` | `close_serving_unit_execution_admission` ≡ gate close; `begin_drain_for_bound_serving_unit` calls it |
| `colocated_serving_runtime_composition.py` | New sanctioned helper using **protocol only** for first deny; reduce reliance on concrete method for authoritative close |
| `strict_production_serving_launcher.py` | Order: admission close → drain → revoke → cleanup (conditional §8) |
| `ColocatedServingUnitAdmissionTaskExecutionPort` | **No** lifecycle/private gate reads |
| `ActivationDomainCoordinator` | **No** new owner |

---

## 6. Concurrency and ordering

**Linearization:** `_UnitAdmissionGate._lock` serializes `admit_execution` vs `begin_drain` / admission close; venv adapter `_lock` serializes unit lifecycle mutations.

| Term | Definition |
|------|------------|
| **Shutdown begins (enforceable)** | `close_serving_unit_execution_admission` **returns successfully** for the bound `serving_unit_ref` — gate deny committed under adapter lock |
| **Not shutdown begins** | First line of `StrictProductionServingSession.close()`, failed revision lookup, or helper raise **before** successful admission close |

| Case | New execution | Notes |
|------|---------------|-------|
| Admitted before successful admission close | Allowed to complete in-flight policy | Existing DIM16-02 |
| Attempt after successful admission close | **Denied** → `ServingRuntimeHandoffError` → HTTP **502** | POST-CLOSE-01 target |
| Concurrent race before close succeeds | May admit | Documented window; closes with gate lock |
| Repeated drain / revoke failures after close succeeds | Denied | Gate stays closed |
| `resume_serving` | Requires fresh coordinator authorization; gate closed on resume path today | No stale resurrection |
| Sibling units | Unaffected | F06 |

---

## 7. Fail-closed and failure-before-deny matrix

**Critical rule:** Fail-closed **execution deny** is guaranteed **only after** successful admission-close port completion. A failed admission-close attempt **does not** imply deny.

| # | Situation | Authoritative admission state | New execution | Cleanup allowed | Process slot release | Session complete | Retry `close()` | Error propagation | Operator |
|---|-----------|------------------------------|---------------|-----------------|----------------------|------------------|-----------------|-------------------|----------|
| 1 | Admission close **succeeded** | Closed | **No** | Yes (subject to F03) | After cleanup policy | Only if no shutdown errors | Rejected if `_process_closed` | Drain/revoke errors preserved | No |
| 2 | Close succeeded; drain continuation failed | Closed | **No** | Yes @ current F02 tests | Yes @ HEAD F02 | No | Rejected | ExceptionGroup | No |
| 3 | Admission close **failed before gate mutation** | **Unchanged** (may be open) | **Yes** until deny succeeds | **No** (R1 policy) | **No** | No | Allowed if slot not released | Admission-close error primary | **Yes** — deny not achieved |
| 4 | Partial / ambiguous failure | Treat as **3** unless adapter documents commit | Assume open | **No** | **No** | No | Per slot state | Original + close error | **Yes** |
| 5 | Close succeeded; cleanup failed | Closed | **No** | N/A (failed) | **No** / partial | No | Per F03 | Triple aggregate | Maybe |
| 6 | Caller interrupted mid-close | If close completed → 1; else → 3 | Per state | Per §8 | Per lifecycle flags | No | Per DIM16-08 | Preserve partial | Per case |
| 7 | Concurrent admit before close succeeds | Open until close | May succeed | — | — | — | — | — | No |
| 8 | Concurrent admit after close succeeds | Closed | **Denied** | — | — | — | — | 502 path | No |

**Case 3 architecture note:** @ HEAD, cleanup runs after helper failures without gate close — **defect**. Target implementation **withholds** `close_process_resources()` when admission close fails. No invented fallback chain. If operator rejects withholding cleanup, status returns to **STOP — ARCHITECTURE DECISION REQUIRED**.

---

## 8. Cleanup, partial close, retry, recovery

- **Partial close:** `session._closed is False` with possible `_process_closed is True` remains valid when shutdown errors non-empty (existing DIM16-F02).
- **Retry:** Second `close()` rejected when `_process_closed` — operator must start **new** session with new activation; no stale handoff reuse.
- **Recovery:** After dual **helper** failure with **successful** admission close, gate **closed** even if `_closed` False — POST-CLOSE-01 target.
- **In-flight:** Existing `release_execution` + `check_drain`; shutdown does not kill in-flight tasks in this slice (DIM16-01/02).

---

## 9. F02 exit criteria (F02-01 … F02-15)

| ID | Requirement | Evidence @ `d3be032d…` | Post-implementation |
|----|-------------|------------------------|---------------------|
| F02-01 | Normal shutdown denies new executions | **VALID** — `test_c_dim16_01` | Re-run |
| F02-02 | Drain fail + successful emergency revoke denies | **VALID** — `test_c_dim16_04` | Re-run |
| F02-03 | Drain + revoke fail still denies admission | **FAIL** — F02 + POST-CLOSE-01 | PASS after impl |
| F02-04 | HTTP after failed close never reaches inner | **FAIL** — POST-CLOSE-01 | PASS |
| F02-05 | Concurrent shutdown/request documented | **NOT PROVEN** | §6 + qual |
| F02-06 | In-flight policy explicit | **PARTIAL** — dim16-02 | Reconfirm |
| F02-07 | Cleanup fail preserves deny | **NOT PROVEN** | F03 + gate assert |
| F02-08 | Partial close cannot resurrect admission | **PARTIAL** — dim16-07 | Re-run |
| F02-09 | New session no stale authorization | **NOT PROVEN** | P2 continuity |
| F02-10 | Sibling units unaffected | **VALID** — `test_c_dim16_f06` | Re-run |
| F02-11 | Tenant boundary unchanged | **VALID** — P3-A | No regression |
| F02-12 | Shutdown errors observable | **VALID** — F02/F03 | Preserve |
| F02-13 | No second admission authority | **PASS (design R1)** | Arch review @ impl |
| F02-14 | Public contracts + replaceability | **PARTIAL** — isinstance @ HEAD | Protocol method |
| F02-15 | Regression auditable @ SHA | **VALID** — red test committed | PASS @ impl SHA |

**Static vs dynamic:** F02-05, F02-07, F02-09 may require evidence beyond 120s budget — operator authorization for extended qual.

---

## 10. Future implementation specification (not executed)

| File | Action |
|------|--------|
| `intergrax/agent_distribution/deployment.py` | Add `close_serving_unit_execution_admission` to Protocol + fake |
| `intergrax/agent_distribution/venv_bundle_colocated_runtime_deployment_adapter.py` | Implement; align `begin_drain_for_bound_serving_unit` |
| `intergrax/applications/_shared/colocated_serving_runtime_composition.py` | `close_colocated_serving_execution_admission_on_strict_session_close` |
| `intergrax/applications/_shared/strict_production_serving_launcher.py` | Order + conditional cleanup |
| `tests/qualification/prod_q/test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py` | POST-CLOSE-01 green; optional admission-close-fail case |

**Forbidden:** second HTTP guard; Apps mutating `_UnitAdmissionGate`; new governance approval; dynamic `getattr` / method probing.

**Architecture STOP:** second admission owner; cleanup after failed admission close without operator waiver; type-check-only adapter dispatch for authoritative deny.

**Bounded verification plan (later task):** `F02-POST-CLOSE-01` first; then F02/F03 + dim16-01/04/06/07; one `uv run pytest … -p no:xdist` ≤60s per command, ≤120s total; no broad qual module without operator OK.

---

## 11. Qualification-boundary debt

`test_prod_q_r1_r7_q_c_dim16_strict_shutdown_lifecycle_gate.py` autouse `_isolate_production_process_lifecycle_slot` — **TRACKED FREEZE DEBT**; not expanded in this task.

---

## 12. Operator decision required

| # | Question |
|---|----------|
| 1 | Ratify **protocol admission-close (A)** + **Apps first-call ordering (B)**? |
| 2 | Ratify **cleanup withheld** when admission close fails (§7 case 3)? |
| 3 | Authorize bounded implementation child (single dim16 pytest module, test-process budget)? |

---

## 13. FRZ mapping (scoped — no global promotion)

| FRZ | @ `d3be032d…` (current) | Intended post-implementation |
|-----|---------------------------|------------------------------|
| **FRZ-PRD-05** | **OPEN / PARTIAL** — F02 blocker | Scoped re-eval after POST-CLOSE PASS |
| **FRZ-PRD-08** | **PARTIAL** — post-close execution | PASS candidate after F02-04 |
| **FRZ-REG-03** | **PASS (scoped)** — red test detects defect | PASS after fix |
| **FRZ-SEC-04** | **NOT PROVEN** | Unchanged slice |
| **FRZ-REL-*** | **PARTIAL** | Admission vs slot consistency improved |
| **FRZ-TEN-*** | **NOT IMPACTED** | — |
| **FRZ-BND-*** | **PASS (design)** | Apps→AD port only |
| **FRZ-OWN-*** | **PASS (design)** | Single AD gate owner |
| **FRZ-CTR-*** | **PARTIAL→PASS (design R1)** | Typed port vs isinstance |
| **FRZ-TYP-*** | **PARTIAL** | Protocol extension |
| **FRZ-RPL-*** | **PARTIAL** | Protocol-level deny |

---

## 14. Enterprise audit matrix @ R1 decision

| Criterion | Verdict | Evidence |
|-----------|---------|----------|
| Hard layer boundaries | **PASS** | Design — Apps orchestrates, AD owns gate |
| Communication direction | **PASS** | Apps → `RuntimeDeploymentAdapter` |
| Exactly-one semantic ownership (admission) | **PASS** | AD gate |
| Exactly-one composition ownership (session close) | **PASS** | `StrictProductionServingSession` |
| Contracts over implementations | **PARTIAL** | R1 selects protocol; isinstance @ HEAD |
| Strong typing | **PASS** | Named port method |
| Pluginability | **PARTIAL** | Strict A1 pin @ HEAD |
| Structural replaceability | **PARTIAL** | Improves at impl |
| No bypass / shadow authorities | **FAIL** @ HEAD | Open gate after dual helper fail |
| No duplicated mechanisms | **PASS** | Single gate |
| Governance vs execution | **PASS** | Unchanged |
| Fail-closed execution | **FAIL** @ HEAD; **NOT PROVEN** post-impl | Red POST-CLOSE-01 |
| Concurrency safety | **PARTIAL** | Gate lock documented |
| In-flight / recovery | **PARTIAL** | §7–8 |
| Unit / tenant isolation | **PASS** | F06 / P3 |
| Observability non-authoritative | **PASS** | — |
| Traceability | **PASS** | SHA + failing test |
| Regression protection | **PASS** | Committed qual |

---

## 15. Roadmap disposition

| Item | Status |
|------|--------|
| This ADR R1 | **READY FOR AUDIT** |
| Package D | **READY FOR AUDIT** + **DIM16/F02 IN-SCOPE BLOCKER** until impl |
| **PROD-Q** | **CURRENT / BLOCKED** (unchanged) |
| **Next task** | `PROD-Q-R1-R7-Q-C-DIM16-F02-FAIL-CLOSED-ADMISSION-HARDENING` — implement §10 after operator ratification |

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**
