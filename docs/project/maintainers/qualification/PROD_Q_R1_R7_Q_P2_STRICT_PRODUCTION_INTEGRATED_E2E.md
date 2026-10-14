# PROD-Q-R1-R7-Q-P2 — Strict production integrated E2E (bounded P2-A)

| Field | Value |
|-------|-------|
| **Status** | **READY FOR AUDIT** (partial qualification — ENV-1 import corrective applied; P2-B chain **not executed**) |
| **Parent** | PROD-Q-R1-R7-Q (not PASS) |
| **Prior child** | PROD-Q-R1-R7-Q-P1 (accepted in bounded static scope) |
| **Branch** | `development` |
| **START_HEAD** | `e0b16ffdb6be6d615501b8ec94671ac5c0b0d459` |
| **ENV-1 FINAL_COMMIT** | *(pending push — see git `development` after PROD-Q-R1-R7-Q-P2-ENV-1)* |
| **Slice** | **P2-A** environment certification + declared D4/D1b reuse matrix; **P2-B** (happy-path + HTTP admission smoke) unblocked for in-process import only; default pytest entry still **ENV-2** |

## Tested production composition path (design intent)

Sanctioned strict chain (no parallel bootstrap):

`ProductionControlPlaneGovernance` → `open_production_process_lifecycle` (D2) → `_activate_bundle` / `ActivationService` (A1 `VenvBundleColocatedRuntimeDeploymentAdapter`) → D3 `ProductionServingReconciliation` → `open_strict_production_serving_session` (D4) → D1b `ServingRuntimeHandoff` → `create_governed_contractor_strict_production_asgi_app` → `colocated_execution_admission.admit_execution` (request-level accounting port).

**Fixture provenance:** Reuses D4 module helpers (`_storage_and_adapter`, `_qualified_canonical_bundle`, `_activate_bundle`, `_open_with_activation`, `_governance`, AC3 canonical authority seeds) — same as `test_prod_q_r1_r7_d4_strict_production_gca_launcher.py`. Physical units: real `prepare` / `check_readiness` on venv bundle under `tmp_path` trusted roots (`allow_qualification_locators=True`); no manual serving-pointer seeding in P2 tests.

## Real vs mocked

| Component | P2-A @ HEAD |
|-----------|-------------|
| Governance (`ProductionControlPlaneGovernance`) | **Not exercised** in P2-A (P2-B pending) |
| D2 lifecycle + launcher activate | **Not exercised** in P2-A |
| A1 venv bundle adapter + readiness | **Not exercised** in P2-A |
| D3 reconciliation / D1b handoff | **Not exercised** in P2-A |
| D4 strict session + GCA ASGI factory | **Not exercised** in P2-A |
| Request-level `admit_execution` | **Not exercised** in P2-A |
| Environment subprocess probes | **Executed** |
| Strict launcher fresh-process import (ENV-1) | **Executed** @ ENV-1 |
| D4/D1b negative replay (subprocess pytest) | **Runnable** when collected without ENV-2 conftest blocker |

No `ready_for_traffic=True` mocks in P2 deliverables; positive tests skip fail-closed when imports fail.

## Identity continuity (assertions when P2-B unblocked)

`assert_identity_continuity` (`_prod_q_r1_r7_q_p2_strict_integrated_e2e.py`):

- `session.environment` ↔ `handoff.identity` (`application_id`, `application_environment_id`, `runtime_revision_id`)
- `session.composition.serving_runtime_handoff is handoff`
- `handoff.runtime_agent_factory_resolver` present
- `colocated_execution_admission.admit_execution` / `release_execution` on `identity.serving_unit_ref`

## Pytest evidence

**Command (sequential, no xdist, partial CI path — `--noconftest` only):**

```text
uv run pytest -p no:xdist -q --noconftest \
  tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_strict_production_integrated_e2e.py
```

**Result @ START_HEAD workspace:** 4 passed, 10 skipped (~36 s). Log: `.tmp/session/prod-q-r1-r7-q-p2/p2-only-r3.log`.

| Test | Result @ HEAD |
|------|----------------|
| `test_p2_g00_qualification_baseline_ancestor` | PASS |
| `test_p2_g01_default_pytest_entry_blocked_by_conftest_cycle` | PASS (certifies INTERNALERROR) |
| `test_p2_g02_strict_launcher_import_chain_succeeds` | PASS @ ENV-1 (historical cycle below) |
| `test_env1_g01` … `test_env1_g04` (`test_prod_q_r1_r7_q_env1_strict_production_import_cycle.py`) | PASS @ ENV-1 |
| `test_p2_g03_bounded_d4_reuse_inventory_complete` | PASS |
| `test_p2_a01` … `test_p2_n03` | SKIP — ENVIRONMENT/TEST ISSUE |

**Default pytest entry (standard CI path, no `--noconftest`):**

```text
uv run pytest --collect-only -q
```

**Result:** INTERNALERROR in `tests/conftest.py` → `RuntimeCheckpoint` import cycle (`runtime_checkpoint.py` ↔ `agent_runtime_io.py` via `agent_distribution` / `runtime.architecture`). **Not** equivalent to qualification PASS.

## ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED

1. **Conftest / session start:** `pytest_sessionstart` imports `RuntimeCheckpoint` → circular import (P1 documented; still present @ `5a4bd283…`).
2. **Strict production import (historical @ START_HEAD — corrected ENV-1):** load-time cycle when `wiring.py` was partially initialized:

```text
wiring.py → agent_distribution/__init__.py → venv_bundle_colocated_runtime_deployment_adapter
→ activation_domain_coordinator → sqlite_stores (module load)
→ registry_projection_descriptor → registry_projection.py
→ wiring.py (BuilderMap) — ImportError: partially initialized module
```

**Root cause:** `sqlite_stores.py` imported `registry_projection_descriptor` at module import time. That pulled `registry_projection`, which imports `wiring` while `wiring` was still loading via `agent_distribution` package `__init__`. The dependency is required only when `SqliteRuntimeRegistryProjectionDescriptorStore.get_for_revision` decodes payloads — not at `sqlite_stores` import time.

**ENV-1 correction:** defer `registry_projection_descriptor` import to `get_for_revision` execution; keep `RuntimeRegistryProjectionDescriptor` under `TYPE_CHECKING` for `put` typing. No new composition/projection owner; same descriptor authority module.

**Fresh-process probe @ ENV-1:**

```text
uv run python -c "from intergrax.applications._shared.strict_production_serving_launcher import open_strict_production_serving_session"
```

→ exit 0.

**Remaining blocker — PROD-Q-R1-R7-Q-P2-ENV-2:** `pytest_sessionstart` / `tests/conftest.py` → `RuntimeCheckpoint` import cycle. Default `uv run pytest --collect-only` still INTERNALERROR until ENV-2.

After ENV-2: re-run P2 module **without** `--noconftest`, then execute P2-B and record HTTP smoke per gate plan §E.2.

## Qualification matrix

| Scenario | Mechanism @ HEAD | Disposition |
|----------|------------------|-------------|
| Happy path activate → session → handoff → GCA | `test_p2_a01`, `test_p2_a02` | **NOT RUN** (skip) |
| Request-level admission | `test_p2_a01` (`admit_execution`) | **NOT RUN** |
| Unready session → no GCA | `test_p2_n01` + D4 `test_d4_asgi_startup_only_after_preconditions` | **NOT RUN** |
| Stale / fresh adapter instance | D4 `test_d4_stale_physical_unit_after_serving` | **NOT RUN** |
| Governance DENY | D4 `test_d4_governance_deny_blocks_activation` | **NOT RUN** |
| Fake adapter | D4 `test_d4_fake_adapter_rejected` | **NOT RUN** |
| Missing serving pointer | D4 `test_d4_missing_serving_pointer_after_open` | **NOT RUN** |
| Wrong revision / scope | D4-21 + D1b `test_handoff_rejects_resolver_instance_mismatch` | **NOT RUN** |
| Invalid artifact / trusted root | Covered by D1 adapter suites (not replayed in P2-A) | **Reuse only** |
| Import / conftest certification | `test_p2_g01` (ENV-2), `test_p2_g02`, ENV-1 module | **PARTIAL** — strict import PASS; default pytest still blocked |

## Enterprise architecture audit (P2 deliverable scope)

| Criterion | Verdict |
|-----------|---------|
| Hard layer boundaries | PASS — ENV-1 defers existing cross-tier edge; no new `applications` import from `agent_distribution` modules |
| Contracts over implementations | PASS — P2 targets sanctioned ports (handoff, admission, launcher) |
| Exactly-one ownership | PASS — no new activation authority or composition owner |
| A1 strict pin | PASS — no alternate production adapter in P2 |
| Fail-closed | PASS — integration tests skip when import chain fails |
| Governance ≠ execution | PASS — reuse D4 governance wiring only (when runnable) |
| No fake/reference serving path | PASS — P2 does not construct reference lifecycle |
| Evidence continuity | PASS — regression module + subprocess probes |

## Tenant isolation audit (P2 scope)

No tenant-binding code changed. Cross-tenant adversarial matrix deferred to **P3**. **Not** a global tenant certification.

## FRZ contributions (evidence only — global OPEN unchanged)

| FRZ | P2-A contribution |
|-----|-------------------|
| FRZ-PRD-03 | Documents fail-closed default pytest + strict import prerequisites for integrated qualification |
| FRZ-PRD-04 | Declares D2→D4 integrated chain tests (execution pending ENV) |
| FRZ-PRD-07 | Negative matrix wired to existing D4/D1b tests |
| FRZ-PRD-08 | Admission port `admit_execution` named in positive test design |
| FRZ-PRD-09 | Identity continuity assertions specified |
| FRZ-PRD-12 | GCA strict factory handoff object identity test specified |
| FRZ-SEC-04 | Unready GCA + governance deny cases in matrix |
| FRZ-REG-02/03/09 | AC3 canonical authority fixture reuse (D4), no pointer seeding |

## Disposition vs parent R7-Q

| Gate | Status |
|------|--------|
| PROD-Q-R1-R7-Q-P2 | **PARTIAL** — P2-A certification only |
| PROD-Q-R1-R7-Q | **NOT QUALIFIED** |
| Local P2 qualification PASS | **No** — cannot assert strict E2E PASS without executing chain |

**Next bounded child:** **PROD-Q-R1-R7-Q-P2-ENV-2** (`RuntimeCheckpoint` / `conftest`), then **P2-B** execute `test_p2_a01`–`test_p2_n03` on default pytest entry.

---

Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.
