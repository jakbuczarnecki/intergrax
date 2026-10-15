# PROD-Q-R1-R7-Q-P3-B — DENY / CAS strict production qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (scoped evidence; does **not** close **P3**, **PROD-Q-R1-R7-Q**, or promote global **FRZ-***) |
| **Child** | **PROD-Q-R1-R7-Q-P3-B** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Branch** | `development` |
| **START_HEAD** | `88da0c903018ceb6d6a90ed80806fb1c39ac0965` |
| **Execution / source SHA** | `88da0c903018ceb6d6a90ed80806fb1c39ac0965` (pre-commit; post-commit SHA recorded in git report below) |

## 1. Preflight

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** | **match** @ `88da0c903018ceb6d6a90ed80806fb1c39ac0965` |
| Working tree @ run start | **clean** (no staged/unstaged tracked delta) |
| **PROD-Q-R1-R7-Q-P3-P0** | **READY FOR AUDIT** (inventory @ prior acceptance) |
| **PROD-Q-R1-R7-Q-P3-A** | **READY FOR AUDIT** (cross-tenant slice; **not** closed) |
| **PROD-Q-R1-R7-Q-P3-A2 ADR** | **PROPOSED — READY FOR AUDIT** (ratification = operator; **not** ratified here) |
| **P3-B admissibility** | **Next independent slice** per P3-P0 §E.2 ordering after P3-A track (parallel P3-A2 implementation evidence orthogonal) |
| D1/D2/D4 strict wiring invalidation | **none** @ HEAD — `deploy_activate_pipeline.py`, `strict_production_serving_launcher.py`, `serving_runtime_handoff.py`, `activation_domain_coordinator.py` unchanged vs execution SHA |

## 2. Scope

| Scenario | Intent |
|----------|--------|
| **P3-T-03** | Governance **DENY** cannot activate, physically prepare (unauthorized), commit traffic **READY**, or issue serving handoff; prior valid serving state not corrupted |
| **P3-T-04** | **CAS** / stale activation conflict — losing revision cannot become **READY** or valid handoff; single-authority state preserved |

**Production code delta:** **none** (qualification + continuity only).

## 3. Canonical code-path inventory

| Role | Owner / symbol | Path |
|------|----------------|------|
| Governance decision | `ControlPlaneMutationAuthorizationBoundary` + policy evaluator; `authorize_scoped_control_plane_mutation` | `intergrax/runtime/governance/control_plane_mutation_authorization.py`, `intergrax/agent_distribution/control_plane_governance.py` |
| Production governance bundle | `ProductionControlPlaneGovernance` | `intergrax/applications/_shared/production_control_plane_governance_wiring.py` |
| Tenant scope on mutation | `TenantScopedControlPlaneMutationEvaluator` | `control_plane_governance.py` |
| Orchestration / composition | `ProductionLifecycleLauncher.deploy_and_activate` → `run_deploy_and_activate` | `production_lifecycle.py`, `deploy_activate_pipeline.py` |
| Activation SO | `ActivationService` (`prepare_candidate`, `commit_activation`) | `intergrax/agent_distribution/activation.py` |
| CAS / linearization SO | `ActivationDomainCoordinator`; store `atomic_swap_serving_revision` / commit expected pointer | `activation_domain_coordinator.py`, durable serving stores |
| Physical prepare | `ActivationService.prepare_candidate` → adapter (coordinator-gated on strict A1 path) | activation + `venv_bundle_colocated_runtime_deployment_adapter.py` |
| Strict session CO | `open_strict_production_serving_session` | `strict_production_serving_launcher.py` |
| Handoff RA | `ServingRuntimeHandoff`, `establish_serving_runtime_handoff`, `authorize_colocated_serving_execution_after_activation` | `serving_runtime_handoff.py`, `colocated_serving_unit_task_execution.py` |

### Mutation ordering (`run_deploy_and_activate`)

1. Input validation (revision / digest / topology).
2. **`_authorize_admission`** — governance **before** durable candidate path side effects that imply admission.
3. Persist candidate + validated revision; register projection input.
4. **`activation.prepare_candidate`** — physical preparation (only after admission **ALLOW**).
5. **`_authorize_activation`** — governance **before** traffic commit.
6. **`activation.commit_activation`** — CAS / expected pointer + digest gates → **READY** authority.
7. Resolve serving projection for result.

Strict opener with `activation_intent`: `open_strict_production_serving_session` → lifecycle open → **`deploy_and_activate`** → reconcile → handoff bind (fail-closed exceptions → `StrictProductionServingLauncherError`).

### Fail-closed exceptions (slice)

- `ReferenceProductionLifecycleGovernanceBlockedError` on DENY / REQUIRE_HUMAN / ESCALATE (`deploy_activate_pipeline._enforce_authorization_result`).
- `StrictProductionServingLauncherError` wrapping governance block on strict activate path.
- `ServingRuntimeHandoffError` on stale pointer / revision mismatch at handoff and admission bind.
- Coordinator / store CAS failures on pointer swap and activation commit (D1-R6 suites).

## 4. Historical evidence reuse and continuity

| Source | Accepted baseline (evidence era) | P3-B reuse | Continuity @ `88da0c90…` |
|--------|----------------------------------|------------|---------------------------|
| **D2** | `test_d2_governance_deny_is_fail_closed` | P3-T-03 governance boundary | **STATIC + replay** — deny at `authorize_scoped_control_plane_mutation`; boundary wired on launcher |
| **D4** | `test_d4_governance_deny_blocks_activation` @ D4 evidence `0dd51fe1…` | P3-T-03 strict session + activation intent | **STATIC + replay** — `open_strict_production_serving_session` fails closed with deny governance |
| **D1b / D1-R3** | `test_stale_activation_pointer_denied` | P3-T-04 stale handoff | **STATIC + replay** |
| **D1-R6** | `r2`–`r8`, `r3_coordinator_scoped_gate_closure`, `r5_serving_admission_linearization` | P3-T-04 CAS / admission linearization | **STATIC** (modules unchanged @ HEAD) + **targeted replay** `test_s1_pointer_swap_wins_before_admission_check` |
| **P3-P0 inventory** | P3-B uncovered list | Integrated strict DENY/CAS → no handoff+ASGI | Still **NOT PROVEN** at integration depth (see §7) |

**Impact analysis:** P3-A / P3-A2-R2 @ HEAD touch HTTP tenant authority and qualification docs only; **no** change to `deploy_activate_pipeline` ordering, coordinator CAS ownership, or D2/D4 deny wiring → D1/D2/D4 negatives remain valid for P3-B claims at this SHA.

## 5. Invariant classification

| Invariant | Classification |
|-----------|----------------|
| DENY blocks control-plane mutation admission | **PROVEN AND REUSABLE** (D2 + pipeline static order) |
| DENY blocks strict session establishment with activation intent | **PROVEN AND REUSABLE** (D4 replay @ HEAD) |
| DENY prevents **READY** commit (no `commit_activation` after deny) | **PROVEN AND REUSABLE** (static: `_authorize_activation` precedes `commit_activation`; D4 session never reaches handoff) |
| DENY prevents **serving handoff** on strict path | **PROVEN AND REUSABLE** (D4) |
| DENY → zero **prepare** when denied at **admission** gate | **PROVEN AND REUSABLE** (static order + D4 path uses full deny before `deploy_and_activate` proceeds) |
| DENY with **activation-only** policy while admission ALLOW | **MISSING STRICT-PATH PROOF** — not exercised; architecture uses distinct mutation requests; operator-scheduled if split policies required |
| CAS pointer swap / stale revision blocks admission & handoff | **PROVEN AND REUSABLE** (D1-R6 r2 + D1-R3 replay) |
| Concurrent CAS under **strict launcher** wiring | **MISSING STRICT-PATH PROOF** — **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** (P3-P0 §B P3-B uncovered) |
| Integrated strict: DENY/CAS → no ASGI factory | **MISSING STRICT-PATH PROOF** — **DEFERRED** (P2 roster subset; not re-run per strict pytest budget) |

No **BLOCKED BY ARCHITECTURE** or **PROVEN DEFECT** @ HEAD.

## 6. P3-T-03 — Governance DENY

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Denied mutation cannot activate revision | **PASS (slice)** | D2 deny; D4 strict activate; activation commit not reached |
| 2 | No unauthorized physical prepare after denial | **PASS (slice)** | `run_deploy_and_activate`: `_authorize_admission` before `prepare_candidate`; D4 deny fails before session/handoff |
| 3 | No READY transition committed | **PASS (slice)** | `_authorize_activation` before `commit_activation`; D4 aborts session |
| 4 | No serving handoff issued | **PASS (slice)** | D4 `StrictProductionServingLauncherError` before successful handoff |
| 5 | Prior valid serving state not corrupted | **PASS (slice)** | D4 uses fresh storage; deny path does not mutate traffic pointer (session abort) |
| 6 | Denial evidence at correct boundary | **PASS (slice)** | `authorize_scoped_control_plane_mutation` / `ControlPlaneMutationAuthorizationBoundary`; strict wrapper maps to `StrictProductionServingLauncherError` |

**Overall P3-T-03:** **PASS (scoped)** — integration-only prepare side-effect counting and split admission/activation policy negatives **NOT PROVEN** (deferred).

## 7. P3-T-04 — CAS conflict / stale activation

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Competing / stale mutation loses via canonical CAS | **PASS (slice)** | D1-R6 `atomic_swap_serving_revision` + admission checks; static coordinator ownership |
| 2 | Losing revision cannot become traffic READY | **PASS (slice)** | Commit paths require expected pointer (D1-R6 / activation); stale pointer denied at bind |
| 3 | Losing revision cannot establish valid handoff | **PASS (slice)** | `test_stale_activation_pointer_denied`; `test_s1_pointer_swap_wins_before_admission_check` |
| 4 | Winning / prior valid state consistent | **PASS (slice)** | r2 s1: in_flight=0 after failed admit; no alternate READY for loser |
| 5 | Physical cleanup / compensation | **PASS (slice)** | D1-R6 r1/r7 compensation contracts — **STATIC** continuity; not re-run in this slice |
| 6 | No CAS bypass path | **PASS (slice)** | Handoff bind requires store-backed commit + coordinator; forged/stale evidence rejected in D1-R6 r1 |

**Overall P3-T-04:** **PASS (scoped)** — strict-launcher concurrent CAS integration **NOT PROVEN** (deferred).

## 8. Test execution (@ HEAD, strict budget)

**Allowed node IDs (only):**

```text
tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py::test_d2_governance_deny_is_fail_closed
tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_governance_deny_blocks_activation
tests/unit/applications/test_prod_q_r1_r7_d1_r3_serving_execution_admission_closure.py::test_stale_activation_pointer_denied
tests/unit/agent_distribution/test_prod_q_r1_r7_d1_r6_r2_physical_admission_final_closure.py::test_s1_pointer_swap_wins_before_admission_check
```

| Metric | Value |
|--------|-------|
| Result | **4 passed** |
| Wall time | **~3.8s** pytest (+ uv cold start ≤11s first invocation) |
| Heavy suites | **not run** (D0–D4 full, P2, P3-A, P3-A2 qual, strict E2E) |

## 9. Real / stub boundaries

| Layer | Boundary |
|-------|----------|
| Governance | Real `authorize_scoped_control_plane_mutation`; test `_DenyEvaluator` (policy stub, **canonical boundary**) |
| Stores | SQLite / in-memory durable stores per D1-R6/D4 fixtures |
| Physical adapter | Real `VenvBundleColocatedRuntimeDeploymentAdapter` where D4 path requires; r2 s1 uses coordinator-seeded fixture |
| HTTP / ASGI | **not exercised** in this slice |

## 10. Unauthorized effects

At scoped evidence depth: **no** path demonstrated where governance **DENY** or CAS stale conflict produces traffic **READY**, valid **ServingRuntimeHandoff**, or strict session success. Alternate production paths (`open_reference_*`, fake adapter on strict graph) remain excluded by P1/P2 continuity (not re-audited here).

## 11. Cleanup and recovery

Activation deny before commit: no serving pointer advance (D4). CAS loser: admission rejected; r2 documents in_flight accounting reset on failed admit. Full rollback/lease suites (D1-R6 r7) — **historical evidence**, not replayed.

## 12. Enterprise architecture audit matrix

| Invariant | SO | CO | RA | Status @ P3-B |
|-----------|----|----|-----|----------------|
| Governance ≠ execution admission | Governance boundary | Apps launcher | Store-backed eligibility | **Aligned** |
| Single CAS authority | AD stores + coordinator | Apps handoff | Expected pointer revisions | **Aligned** |
| Physical prepare after governance admission | AD activation | `run_deploy_and_activate` | Coordinator gate (A1) | **Aligned** |
| Strict handoff bind | Apps | GCA admission bridge | `ServingRuntimeHandoff` | **Aligned** |

## 13. FRZ mapping (contribution only — globals unchanged)

| FRZ ID | P3-B contribution |
|--------|-------------------|
| **FRZ-PRD-08** | Scoped negative evidence: deny / CAS cannot bypass to READY or handoff on canonical paths |
| **FRZ-SEC-04** | Unactivated / unauthorized activation cannot complete strict serving handoff (D4 deny continuity) |
| **FRZ-STA-03** | Explicit atomic swap / commit boundaries (D1-R6 static + r2 replay) |
| **FRZ-STA-05** | Stale pointer / revision rejected (D1-R3 + r2) |
| **FRZ-REG-02** | Reuses existing regression modules; **no new gate** added in this slice |
| **FRZ-REG-03** | Exact-SHA record: this document @ commit on `development` |
| **FRZ-REG-09** | Continuity analysis vs D2/D4/D1-R6 evidence baselines |

Global **FRZ-PRD-08** / **FRZ-SEC-04** rows in freeze checklist remain **OPEN** (P3-B is R7-Q wave-3 contribution only).

## 14. Remaining blockers / next slice

| Item | Disposition |
|------|-------------|
| Integrated strict DENY → no handoff + no ASGI | **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** |
| Concurrent CAS under strict launcher wiring | **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** |
| Split admission-ALLOW / activation-DENY policy | **NOT PROVEN** (low priority unless operator policy requires) |
| **P3-A2 ADR ratification** | Operator decision — **PROPOSED** |
| **Next mandatory P3 child (ordering)** | **PROD-Q-R1-R7-Q-P3-C** — stale READY / physical unit loss |
| **E.1** full D0–D4 replay | Operator-scheduled; blocks **R7-Q PASS** only |

## 15. Cursor disposition

**READY FOR AUDIT** — independent GitHub SHA audit required for closure; this report is not final acceptance.
