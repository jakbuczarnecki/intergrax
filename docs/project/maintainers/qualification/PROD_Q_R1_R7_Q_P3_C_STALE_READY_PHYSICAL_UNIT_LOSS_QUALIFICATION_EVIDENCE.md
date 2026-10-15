# PROD-Q-R1-R7-Q-P3-C — Stale READY / physical serving unit loss qualification evidence

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (scoped evidence; does **not** close **P3**, **PROD-Q-R1-R7-Q**, or promote global **FRZ-***) |
| **Child** | **PROD-Q-R1-R7-Q-P3-C** |
| **Parent** | **PROD-Q-R1-R7-Q-P3** — **NOT CLOSED** |
| **Branch** | `development` |
| **START_HEAD** | `11cf7df00cbf7a76c5701d28171f5770a7e8096b` |
| **Execution / source SHA** | `11cf7df00cbf7a76c5701d28171f5770a7e8096b` (pre-commit; post-commit SHA in git report below) |

## 1. Preflight

| Check | Result |
|-------|--------|
| Branch | `development` |
| HEAD vs **EXPECTED_START_HEAD** | **match** @ `11cf7df00cbf7a76c5701d28171f5770a7e8096b` |
| Working tree @ run start | **clean** |
| **PROD-Q-R1-R7-Q-P3-P0** | **READY FOR AUDIT** (inventory) |
| **PROD-Q-R1-R7-Q-P3-B** | **READY FOR AUDIT** @ `11cf7df00…` (prior child on same HEAD chain) |
| **PROD-Q-R1-R7-D3** | **READY FOR AUDIT** — evidence implementation @ `f2dd8ecbbff8561a50ec10321ddef62d345b65a9` |
| **PROD-Q-R1-R7-D4** | **CLOSED / independently accepted** — evidence HEAD `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` |
| **PROD-Q-R1-R7-Q** parent | **IN PROGRESS** / **NOT QUALIFIED**; **E.1** full D0–D4 replay **NOT EXECUTED** |
| Reconciliation / attestation invalidation @ HEAD | **none** — `production_process_serving_reconciliation.py`, `production_process_lifecycle.py` (reconcile loop), `activation.py` (`reconcile_traffic_serving_physical_unit_after_process_restart`) **zero diff** `f2dd8ecb…` → HEAD |

## 2. Scope

| Scenario | Intent |
|----------|--------|
| **P3-T-05** | Durable traffic **READY** in store + lost / empty / non-attestable physical serving unit → **no** valid strict serving session, **no** handoff, **no** traffic-ready strict path |

**Production code delta:** **none** (qualification + continuity only).

## 3. Ownership and trust graph

| Role | Owner / symbol | Path |
|------|----------------|------|
| Traffic **READY** durable authority | `ActivationService.commit_activation` + `ApplicationEnvironmentServingStore` traffic pointer | `intergrax/agent_distribution/activation.py`, AD stores |
| Physical attestation SO | `ActivationService._attest_ready_deployment_instance` via `reconcile_traffic_serving_physical_unit_after_process_restart` | `activation.py` (DEC-08) |
| Reconciliation CO | `reconcile_production_serving_environment_on_process_open` | `production_process_serving_reconciliation.py` |
| Mandatory targets RA | `resolve_mandatory_production_serving_reconciliation_targets` ← `list_traffic_serving_records()` | same |
| Process-open orchestration | `open_production_process_lifecycle` reconcile loop before `_commit_production_active_slot` | `production_process_lifecycle.py` |
| Strict serving composition CO | `open_strict_production_serving_session` | `strict_production_serving_launcher.py` |
| Handoff RA | `establish_serving_runtime_handoff` / `_bind_serving_handoff_for_environment` | `serving_runtime_handoff.py`, strict launcher |
| GCA strict ASGI admission | `create_governed_contractor_strict_production_asgi_app` + session guards | `applications/governed_contractor_application/host/strict_production_serving.py` |

**Invariant:** Process restart does **not** infer physical readiness from historical durable **READY**; reconciliation **re-attests** via `RuntimeDeploymentAdapter` on the canonical path.

## 4. Real physical reconciliation path (closed-world)

```text
Durable traffic_serving_revision_id (READY in store)
  → open_production_process_lifecycle
      → resolve_mandatory_production_serving_reconciliation_targets
      → reconcile_production_serving_environment_on_process_open
          → reconcile_traffic_serving_physical_unit_after_process_restart
          → RuntimeDeploymentAdapter attestation (empty/stale adapter → RuntimeActivationError)
      → ProductionServingReconciliationError → slot abort + store cleanup (fail-closed)
  → open_strict_production_serving_session
      → open_production_process_lifecycle (same reconcile on open; maps to StrictProductionServingLauncherError)
      → lifecycle.reconcile_serving_environment (second reconcile on success path only)
      → _bind_serving_handoff_for_environment → readiness.ready_for_traffic
```

**Fail-closed types (slice):** `RuntimeActivationError` / `RuntimeActivationConflict` → `ProductionServingReconciliationError` → `StrictProductionServingLauncherError` on strict opener; lifecycle open uses `raise_startup_failure_with_cleanup` without committing the production slot.

**Cleanup:** Reconciliation failure before slot commit releases opening slot (`_abort_production_opening_slot`) and owned stores (`release_lifecycle_owned_stores`); D3-R1 `test_d3_r1_reconciliation_failure_releases_opening_slot` documents slot recovery. Strict opener closes partial lifecycle on any startup failure (`close_process_resources`).

## 5. Durable READY vs physical state

| Layer | Meaning |
|-------|---------|
| Durable | `ApplicationEnvironmentServingRecord.traffic_serving_revision_id` + pointer revision — survives process exit |
| Physical | Deployment instance + adapter attestation table — **not** assumed after restart |
| Reconcile | Mandatory when durable traffic record exists; attestation failure **does not** clear durable READY silently in these negative paths (no false promotion; recovery is operator re-activate path — out of P3-C slice) |

## 6. D3/D4 historical evidence and provenance

| Evidence era | SHA (canonical) | Module / tests | P3-C reuse |
|--------------|-----------------|----------------|------------|
| **D3** | `f2dd8ecbbff8561a50ec10321ddef62d345b65a9` | `test_prod_q_r1_r7_d3_production_restart_reconciliation.py`, `test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py` | Lifecycle open + reconcile pipeline |
| **D3-R1** | same chain | Mandatory targets without caller hints; slot release on reconcile fail | Recovery after failed open |
| **D4** | `0dd51fe1c7fdd9f4f93d4abdf26f89bea879709e` | `test_prod_q_r1_r7_d4_strict_production_gca_launcher.py` | Strict session after serve + fresh empty adapter |
| **P2 integrated roster** | `8e43b160447ca43de6631fe2a1b7f99948b64f2f` | P2 record lists D4 stale replay **PASS** | Cross-reference only |
| **D0** | `2b44cee…` (gate plan) | `test_d0_02_persisted_ready_empty_adapter_after_restart_fails` | Attestation orchestration continuity |

**Continuity @ `11cf7df00…`:**

| Path | Delta since evidence SHA |
|------|---------------------------|
| `production_process_serving_reconciliation.py` | **0** (`f2dd8ecb…` → HEAD) |
| `production_process_lifecycle.py` (reconcile-on-open) | **0** |
| `strict_production_serving_launcher.py` reconcile/handoff gate | **0** since `0dd51fe1…` (D4-R2 chain accepted) |
| `test_d4_stale_physical_unit_after_serving` body | **unchanged** (D4 module helper imports only — P3-A2 HTTP test helpers) |

## 7. Static continuity analysis

| Claim | Continuity basis |
|-------|------------------|
| Reconcile runs **before** production slot commit | `open_production_process_lifecycle` lines 388–411 — reconcile in loop, `_commit_production_active_slot` only after success |
| Caller cannot omit active traffic env | `resolve_mandatory_production_serving_reconciliation_targets` unions store `list_traffic_serving_records()` |
| Strict opener cannot skip reconcile | `open_strict_production_serving_session` always calls `open_production_process_lifecycle` with `serving_reconciliation_targets` |
| Handoff only after `RECONCILIATION_OUTCOME_READY` | `_bind_serving_handoff_for_environment` + `ready_for_traffic` gate raises if incomplete |
| No shadow reconcile owner | Single pipeline in `production_process_serving_reconciliation.py` — **no** new mechanism in P3-C |

## 8. P3-T-05 acceptance matrix

| # | Invariant | Classification | Evidence |
|---|-----------|----------------|----------|
| 1 | Durable **READY** discovered at process open | **PROVEN AND REUSABLE** | D3 `test_d3_open_fails_reconciliation_on_empty_adapter_after_serving`; D3-R1 `test_d3_r1_open_fails_stale_physical_without_caller_targets` |
| 2 | Missing / stale physical unit → reconciliation failure | **PROVEN AND REUSABLE** | D3 `test_d3_stale_physical_unit_fails_reconciliation`; D3 open-fail test; activation reconcile path |
| 3 | Strict serving session cannot complete successfully | **PROVEN AND REUSABLE** | D4 `test_d4_stale_physical_unit_after_serving`; **replay @ HEAD** (§9) |
| 4 | No trusted **ServingRuntimeHandoff** issued | **PROVEN BY STATIC CONTINUITY** | Strict opener fails before `_bind_serving_handoff_for_environment` returns session |
| 5 | No traffic-ready ASGI via strict launcher | **PROVEN BY STATIC CONTINUITY** | Session not returned → `create_governed_contractor_strict_production_asgi_app` requires successful `open_strict_production_serving_session` (D4-03 / P2-N01 family) |
| 6 | No execution admission on invalid session | **PROVEN BY STATIC CONTINUITY** | `ready_for_traffic` false → `StrictProductionServingLauncherError`; GCA admission bind not reached |
| 7 | Durable state not falsely promoted / silently rewritten | **PROVEN AND REUSABLE** | Reconcile failure aborts open; no commit_activation on this path |
| 8 | Consistent resources for subsequent recovery | **PROVEN AND REUSABLE** | D3-R1 `test_d3_r1_reconciliation_failure_releases_opening_slot` |

**Overall P3-T-05:** **PASS (scoped)** — historical D3/D4 negatives + static strict-launcher ordering; **no** new reconcile owner.

## 9. Integration-depth gaps

| Gap | Disposition |
|-----|-------------|
| Strict path: stale READY → **HTTP** request hits GCA after partial startup | **NOT PROVEN** — **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** (session never established; P3-P0 dim 14 “operator-visible” on live host) |
| Full strict launcher E2E roster replay | **NOT EXECUTED** (pytest budget / E.1 policy) |
| Physical E2E with production **VENV_BUNDLE** deployment table loss on real host | **DEFERRED — OPERATOR-SCHEDULED QUALIFICATION** |
| Global **FRZ-PRD-09/12** promotion | **OUT OF SCOPE** — contribution only |

## 10. Test execution (@ HEAD, strict budget)

**Permitted node IDs (only):**

```text
tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_stale_physical_unit_after_serving
```

| Metric | Value |
|--------|-------|
| Result | **1 passed** |
| Pytest wall time | **~3.25s** |
| Historical D3 module | **not rerun** (continuity: reconciliation code **0** delta since `f2dd8ecb…`) |

## 11. Physical / stub boundary

| Layer | Boundary |
|-------|----------|
| Adapter | `VenvBundleColocatedRuntimeDeploymentAdapter` with qualification locators (D3/D4 harness) — **real A1 attestation ports**, empty fresh adapter simulates post-restart unit loss |
| Stores | SQLite durable AD DB in D3/D4 fixtures |
| HTTP / ASGI | **not exercised** in P3-C replay (strict session failure before factory) |
| Reference / fake strict graph | Excluded by P1/P2 continuity (not re-audited here) |

## 12. Cleanup and recovery implications

Failed reconcile on `open_production_process_lifecycle`: opening slot aborted; continuation/PIS stores released on startup failure path. Durable **READY** row remains for operator-driven re-activation after physical restore — consistent with fail-closed semantics (no serving on stale unit). Strict opener closes lifecycle if partially constructed.

## 13. Enterprise architecture audit matrix

| Invariant | SO | CO | RA | Status @ P3-C |
|-----------|----|----|-----|----------------|
| Exactly-one READY commit authority | AD activation | Apps launcher | Serving store pointer | **Aligned** |
| Exactly-one physical attestation owner | AD `ActivationService` | — | Deployment instance + adapter | **Aligned** |
| Exactly-one reconcile pipeline | AD attest API | Apps `reconcile_production_serving_environment_on_process_open` | Mandatory traffic records | **Aligned** — **no** shadow path |
| Exactly-one strict composition owner | — | `open_strict_production_serving_session` | Handoff bind | **Aligned** |
| Durable ≠ physical | Stores vs adapter | Reconcile on open | DEC-08 | **Aligned** |
| Fail-closed | Attestation errors | Slot abort | — | **Aligned** |
| Governance ≠ execution | N/A in slice | Strict path uses reconciled state only w/o activation_intent | — | **Aligned** |
| No fake/reference fallback on strict graph | — | `_require_venv_production_adapter` | — | **Aligned** (P1 continuity) |
| Replaceability / traceability | Typed reconciliation evidence dataclasses | — | — | **Aligned** |

No **BLOCKED BY ARCHITECTURE** or **PROVEN DEFECT** @ HEAD.

## 14. FRZ mapping (contribution only — globals unchanged)

| FRZ ID | P3-C contribution |
|--------|-------------------|
| **FRZ-PRD-09** | Stale physical unit cannot re-establish traffic-ready strict session while durable READY persists |
| **FRZ-PRD-12** | Post-restart reconcile fail-closed + slot cleanup; D3-R1 recovery projection |
| **FRZ-PRD-04** | Reconcile ordered before production slot commit on process open |
| **FRZ-SEC-04** | No execution admission without successful reconcile + handoff bind |
| **FRZ-REG-02** | Reuses D3/D4 regression modules; **no** duplicate P3-named reconcile test |
| **FRZ-REG-03** | Exact-SHA record: this document @ commit on `development` |
| **FRZ-REG-09** | Continuity vs `f2dd8ecb…` / `0dd51fe1…` baselines |

Global checklist rows remain **OPEN** (P3-C is R7-Q wave-3 contribution only).

## 15. Remaining blockers / next slice

| Item | Disposition |
|------|-------------|
| **E.1** full D0–D4 replay | Operator-scheduled; blocks **R7-Q PASS** |
| **P3-F** (dim 20) | **BLOCKED** on parent **PROD-Q-R1** per P3-P0 |
| **P3-A2 ADR ratification** | Operator — **PROPOSED** |
| **Next mandatory P3 child (ordering)** | **PROD-Q-R1-R7-Q-P3-D** — drain/stop under strict session (P3-T-06) |

## 16. Cursor disposition

**READY FOR AUDIT** — independent GitHub SHA audit required for closure; this report is not final acceptance.

**Recommendation:** Scoped acceptance of **P3-T-05** via D3/D4 evidence reuse + one D4 stale strict replay @ HEAD; **no** new production code and **no** duplicate D3 reconciliation test under a P3 filename.
