# PROD-Q-R1-R7-D2 — Production lifecycle composition (Cursor evidence)

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** — recommends independent closure review; **not** CLOSED |
| **Branch** | `development` |
| **START_HEAD** | `f75af4a2bca63e9db6af30c0871d9f5ebe749c54` |
| **Alignment baseline** | prerequisite alignment commit (see session report) |
| **FINAL_COMMIT** | (session report) |
| **Parent** | **PROD-Q-R1-R7** — **NOT CLOSED** |

## Sanctioned production entry points (DEC-03)

| Symbol | Module | Role |
|--------|--------|------|
| `open_production_process_lifecycle` | `production_process_lifecycle.py` | Single sanctioned production process opener |
| `wire_production_lifecycle_services` | `production_lifecycle.py` | Durable SQLite bundle + required `RuntimeDeploymentAdapter` |
| `ProductionControlPlaneMutationAuthorizationBoundary` | `production_control_plane_governance_wiring.py` | Typed alias; explicit injection required |

Reference path (`open_reference_production_process_lifecycle`, `FakeInMemoryRuntimeDeploymentAdapter`) remains **REFERENCE-ONLY**.

## Regression (Cursor session)

**Log:** `.tmp/session/PROD-Q-R1-R7-D2/pytest-qualification.log`

**D2 module:** `tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py` — **10 passed**

**Pyright:** `.tmp/session/PROD-Q-R1-R7-D2/pyright-d2.log` — **0 errors** on D2 modules

## FRZ-PRD contributions (scoped evidence only — global rows OPEN)

| Criterion | Evidence |
|-----------|----------|
| FRZ-PRD-03 | Fail-closed on missing/fake adapter, governance, store mismatch |
| FRZ-PRD-04 | `open_production_process_lifecycle` startup + continuation ownership |
| FRZ-PRD-05 | `close_process_resources` / context manager cleanup |
| FRZ-PRD-07 | Strict production module vs reference lifecycle separation |
| FRZ-PRD-08 | No fake adapter reachability in `production_process_lifecycle.py` |

## Out of scope (unchanged)

D3 durable projection inputs, D4 external launcher/ASGI, R7-Q end-to-end certification.

---

## Current-HEAD independent acceptance audit (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE` (D2 slice) |
| **AUDITED_HEAD** | `a7e959d2d3171924c506ab9483e466543aef39b1` |
| **origin/development** | `a7e959d2d3171924c506ab9483e466543aef39b1` (clean) |
| **Implementation tip (D2 chain)** | `b631c18cf270bde29264d5e83d0ebb631173f0dc` (D2-R2); no commits on `production_process_lifecycle.py` / `production_lifecycle.py` since `f2dd8ecbb…` (D3-R1) |
| **Production delta** | **0** (audit-only) |

### Drift vs historical D2 evidence

Adjacent commits after `f2dd8ecbb…` touch **D4** strict launcher, **DIM13** recovery admission, **DIM16/F02** strict-session physical admission close, and P2/P3 HTTP — **not** the sanctioned D2 opener/wiring modules. **DIM16** extends shutdown at the D4 session boundary; `ProductionProcessLifecycle.close_process_resources` ownership unchanged.

### D2 criterion matrix (@ HEAD)

| Label | Map | Result | Evidence |
|-------|-----|--------|----------|
| D2-01 | DEC-03 single opener | **PASS** | `open_production_process_lifecycle` only in strict production module; slot tests in D2 module |
| D2-02 | governance before effects | **PASS** | `require_production_control_plane_governance`; deny/missing tests |
| D2-03 | typed deployment deps | **PASS** | `RuntimeDeploymentAdapter`; replaceable adapter test |
| D2-04 | prod/lab separation | **PASS** | no `FakeInMemory` / reference opener in `production_process_lifecycle.py` |
| D2-05 | cleanup / slot | **PASS** | startup failure, partial init, idempotent close tests |
| D2-06 | no shadow authority | **PASS** | launcher binds `mutation_authorization_boundary`; reconciliation does not open admission |

### Targeted regression (@ AUDITED_HEAD)

```text
uv run pytest -p no:xdist -q tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py
```

**18 passed**, exit **0**, ~7s — log: `.tmp/session/PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE/pytest.log`

### Risk review (D2-relevant)

| Risk | Disposition |
|------|-------------|
| Missing governance | **PASS** — fail-closed tests |
| Duplicate lifecycle | **PASS** — slot + concurrent tests |
| Failed init / partial cleanup | **PASS** — D2-R2 invocation-store paths |
| Fake adapter reachability | **PASS** — static + negative tests |
| DIM16 shutdown vs D2 cleanup | **PASS (compat)** — admission close is D4-layer; D2 `close_process_resources` contract preserved |

### FRZ contribution (scoped; global rows **OPEN**)

FRZ-PRD-03, FRZ-PRD-04, FRZ-PRD-05, FRZ-PRD-07, FRZ-PRD-08 — supporting evidence unchanged; **no new global FRZ PASS**.

### D2 acceptance recommendation

**READY FOR AUDIT** — recommends independent operator closure review on GitHub SHA `a7e959d2…`; **not** CLOSED. No further D2 implementation required unless independent audit found defect.

---

## PROD-Q-R1-R7-D2 GCA owned host store lifecycle closure (targeted recert)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D2` — delta `owned_governed_external_work_host_store_bundle` |
| **Branch** | `development` |
| **START_HEAD** | `ddd31385803175c6862767b1532c51d07208706c` |
| **FINAL_COMMIT** | (set at push) |
| **Parent** | `PROD-Q-R1-R7-D2` |
| **Status** | **READY FOR AUDIT** — `TECHNICALLY ACCEPTABLE — FORMAL ACCEPTANCE PENDING` |
| **Production delta** | `production_process_lifecycle.py` — uncommitted lifecycle cleanup includes GCA via canonical `close_process_resources` |

### Ownership graph (canonical)

| Phase | GCA bundle owner | Cleanup owner |
|-------|------------------|---------------|
| Before `ProductionProcessLifecycle(...)` | Caller / strict launcher injector | Caller (opener never took lifecycle root) |
| After lifecycle dataclass construction, before `_commit_production_active_slot` | `ProductionProcessLifecycle._owned_governed_external_work_host_store_bundle` | `_release_uncommitted_production_lifecycle_resources` → `close_process_resources()` |
| After active slot commit | Same field on active lifecycle | `ProductionProcessLifecycle.close_process_resources()` only |
| Strict session startup failure after successful open | Active lifecycle on session | `open_strict_production_serving_session` → `close_process_resources()` |

`release_lifecycle_owned_stores` remains continuation + invocation only (no GCA parameter); GCA never has a shadow cleanup path.

### D2-GCA acceptance matrix

| ID | Result | Rationale |
|----|--------|-----------|
| D2-GCA-01 | **PASS** | `close_process_resources` closes owned GCA exactly once; slot released via `_release_active_production_slot` |
| D2-GCA-02 | **PASS** | Pre-lifecycle: caller-owned; post-partial-root: `_release_uncommitted_production_lifecycle_resources`; D4 session path unchanged |
| D2-GCA-03 | **PASS** (post-fix) | Reconciliation / post-construction startup failures close partial lifecycle including GCA; primary reconciliation error preserved; `_process_closed` guard prevents double-close on outer handler |
| D2-GCA-04 | **PASS** | GCA close failure propagates; slot still released; strict session tests preserve primary + cleanup `ExceptionGroup` |

### Executed pytest (Cursor session)

```text
uv run pytest -p no:xdist \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_gca_owned_host_store_closed_after_close_process_resources \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_gca_owned_host_store_close_failure_propagates_and_releases_slot \
  tests/unit/applications/governed_contractor/test_prod_q_r1_r7_d4_strict_production_gca_launcher.py::test_d4_gca_owned_host_store_closed_after_reconciliation_failure \
  -q
```

**3 passed** — log: `.tmp/session/PROD-Q-R1-R7-D2-GCA/pytest.log`

### ECAP (scoped A1–A8, B1–B5; B6 PENDING)

Exactly-one semantic/composition owner for GCA host persistence; typed `OwnedGovernedExternalWorkHostStoreBundle`; fail-closed binding checks; no harness ownership transfer; error provenance via existing `raise_startup_failure_with_cleanup` / `raise_primary_with_cleanup_errors`; layer boundaries preserved.

### FRZ (scoped evidence only — global rows OPEN)

| FRZ | Contribution |
|-----|----------------|
| FRZ-PRD-04 | Acquired GCA bundle released on failed open after lifecycle construction |
| FRZ-PRD-05 | Canonical `close_process_resources` remains sole GCA close on committed lifecycle |
| FRZ-REL | No leaked durable GCA SQLite on reconciliation failure |
| FRZ-REG | Regression nodes above |

### Blockers

None in scope. Independent GitHub SHA audit required before formal closure.
