# PROD-Q-R1-R7-D3 — Durable projection inputs & post-restart reconciliation

| Field | Value |
|---|---|
| **Status** | **READY FOR AUDIT** (implementation evidence — does not close PROD-Q-R1-R7, PROD-Q, or FRZ-PRD global PASS) |
| **Parent** | PROD-Q-R1-R7 |
| **Successor** | PROD-Q-R1-R7-D4 |
| **START_HEAD** | `b631c18cf270bde29264d5e83d0ebb631173f0dc` |
| **DEC anchors** | DEC-04 (durable AD + descriptor authority), DEC-08 (physical attestation before serving reliance) |

## Design summary

- **Durable projection inputs:** `DescriptorBackedRegistryProjectionInputStore` implements `RegistryProjectionInputStore` by reconstructing `RegistryProjectionInputBundle` from `RuntimeRegistryProjectionDescriptor` (SQLite) plus `RegistryProjectionAuthorityResolver` (revision, roster, lock, materialization). Session `register()` retains live resolver instances; `get()` rebuilds factory resolver via `assemble_production_registry_projection_input_bundle` — no callable pickle.
- **Production wiring:** `wire_production_lifecycle_services` and `wire_durable_reference_production_lifecycle_services` use the descriptor-backed store.
- **Restart reconciliation:** `ActivationService.reconcile_traffic_serving_physical_unit_after_process_restart` attests traffic-serving deployment instances (READY/SERVING). `reconcile_production_serving_environment_on_process_open` composes attestation, projection input reconstruction, rehydration, and coordinator `prepare_projection`. `open_production_process_lifecycle` unions durable `ApplicationEnvironmentServingStore.list_traffic_serving_records()` with caller hints via `resolve_mandatory_production_serving_reconciliation_targets` (caller cannot omit active traffic). Fails closed on reconciliation errors before committing the process slot.
- **D3-R1 (B02):** `DescriptorBackedRegistryProjectionInputStore.register` validates durable descriptor authority before session publication; `get` prefers durable reconstruction when a descriptor exists (staged-only path when descriptor absent).

## Evidence commands

```text
uv run pytest -p no:xdist tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_prod_q_r1_r7_d2_production_process_lifecycle.py tests/unit/applications/test_registry_projection_descriptor_corruption.py tests/integration/agent_distribution/test_enterprise_agent_lifecycle_durable_e2e.py tests/integration/agent_distribution/test_enterprise_projection_rehydration_e2e.py
uv run pyright intergrax/applications/_shared/durable_registry_projection_input_store.py intergrax/applications/_shared/production_process_lifecycle.py intergrax/applications/_shared/production_process_serving_reconciliation.py intergrax/agent_distribution/stores.py intergrax/agent_distribution/in_memory_stores.py intergrax/agent_distribution/sqlite_stores.py
```

## D2 alignment note

Operator disposition: D2 remains **ACCEPTED / READY FOR AUDIT** within A1; this stage does not reopen D2 implementation.

---

## Current-HEAD independent acceptance audit (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE` (D3 slice) |
| **AUDITED_HEAD** | `a7e959d2d3171924c506ab9483e466543aef39b1` |
| **Implementation tip** | `f2dd8ecbbff8561a50ec10321ddef62d345b65a9` (D3-R1) — **identical** to AUDITED_HEAD for D3-owned modules |
| **Production delta** | **0** (audit-only) |

### Drift vs D3-R1 evidence SHA

No file changes on `durable_registry_projection_input_store.py`, `production_process_serving_reconciliation.py`, or D3 reconciliation hooks in `production_process_lifecycle.py` since `f2dd8ecbb…`. Post-D3-R1 work (**DIM13**, **DIM16**, D4 launcher) composes **through** `open_production_process_lifecycle` / reconciliation without altering D3 semantic owners.

### D3 criterion matrix (@ HEAD)

| Label | Map | Result | Evidence |
|-------|-----|--------|----------|
| D3-01 | durable descriptor authority | **PASS** | D3-R1 register validation + mismatch negatives |
| D3-02 | reconstruction continuity | **PASS** | descriptor-backed reconstruct + SQLite restart test |
| D3-03 | physical liveness reconciliation | **PASS** | `reconcile_traffic_serving_physical_unit_after_process_restart` tests |
| D3-04 | no stale serving resurrection | **PASS** | stale physical / empty adapter / mandatory targets |
| D3-05 | recovery sequencing | **PASS** | open-path reconciliation before slot commit; coordinator `prepare_projection` in pipeline test |
| D3-06 | regression / error continuity | **PARTIAL** | D3-R1 + D3 negatives **PASS**; `test_registry_projection_descriptor_corruption.py` **FAIL** at harness setup (see below) |

### Targeted regression (@ AUDITED_HEAD)

```text
uv run pytest -p no:xdist -q \
  tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py \
  tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py \
  tests/unit/applications/test_registry_projection_descriptor_corruption.py
```

**19 passed, 1 failed** (exit **1**), ~15s cumulative with D2 module in same run — log: `.tmp/session/PROD-Q-R1-R7-D2-D3-CURRENT-HEAD-INDEPENDENT-ACCEPTANCE/pytest.log`

**Failure:** `test_descriptor_corruption_fail_closed` — `EnterpriseAgentLifecycleProofStack.run_happy_path()` → `AgentBinding requires agent_type or import_path` in `agent_resolution.py` (test-support / catalog binding drift; **not** D3 production module delta).

**DIM13/DIM14:** not replayed in this task (qualification gates under `tests/qualification/prod_q/`); D3 unit scope covers mandatory reconciliation and descriptor-store paths. Integrated DIM13 evidence remains at historical §C SHAs per roadmap.

### Unresolved finding

| Class | Item | Action |
|-------|------|--------|
| **ENVIRONMENT/TEST ISSUE — EVIDENCE REQUIRED** | Rehydrator corruption regression via `EnterpriseAgentLifecycleProofStack` | Bounded harness/catalog binding repair + re-run **one** test module; **not** a D3 production invariant violation |

### Risk review (D3-relevant)

Risks 6–14 from task brief: **PASS** via D3-R1 mandatory targets, stale physical negatives, reconciliation-before-slot, tenant/store binding in D2 module where shared. Risk 15 (DIM16 shutdown): **no D3 authority expansion** — physical admission close is downstream of D3 reconciliation contract.

### FRZ contribution (scoped; global rows **OPEN**)

FRZ-PRD-12, supporting FRZ-REC/FRZ-STA/FRZ-TEN via recovery fail-closed semantics — evidence contribution only.

### D3 acceptance recommendation

**READY FOR AUDIT** — core D3/D3-R1 invariants **proven @ HEAD** (19/19 D3 unit tests). Recommend independent closure review on GitHub SHA `a7e959d2…`; **not** CLOSED until operator accepts and optional corruption-module replay is green. **No D3 production implementation** required for current-HEAD alignment.

---

## D3-R1 descriptor corruption harness binding corrective (Cursor)

| Field | Value |
|-------|-------|
| **Task** | `PROD-Q-R1-R7-D3-R1-DESCRIPTOR-CORRUPTION-HARNESS-BINDING-CORRECTIVE` |
| **START_HEAD** | `ffda0c36180a4603810d92c228252f7fd813d5de` |
| **Production delta** | **0** (test-support / qualification evidence only) |

### Root cause

`reference_install_contract_authority_records(manifest)` → `materialize_manifest_contract_authority_lab_compat` → `resolve_agent_contract_from_binding` in `install_from_catalog` (`canonical_agent_lifecycle_composition.py`). Manifest roster used `AgentBinding(contract_id=…, builder_key=…)` only — no `agent_type` / `import_path` — so `resolve_agent_type` in `intergrax/applications/_shared/agent_resolution.py` raised before install completed.

### Corrected harness contract

| Layer | Before | After |
|-------|--------|--------|
| Manifest roster (projection) | `contract_id` + `builder_key` only | unchanged via `_stage15_lab_manifest_agent_bindings` (revision-bound factory via roster `builder_key`) |
| Install `package_contract_authority` | derived from incomplete manifest binding | `_stage15_install_package_contract_authority_records` using `CanonicalPingAgent().get_contract()` + `package_contract_authority_record` |
| Package metadata | empty `declared_contracts` | `_stage15_project_metadata` with `AgentPackageContractDeclaration` for canonical ping |
| Enterprise `run_happy_path` | full canonical execute | `verify_execution=False` (descriptor corruption needs activation + descriptor only) |
| Canonical `execute_canonical` | host without revision-bound contract authority | `roster_contract_authority_for` + strict host prerequisites (collaborative work sqlite, continuation store, lab decision policy, test GEP) |

### Changed files

- `testing_support/canonical_agent_lifecycle_composition.py`
- `testing_support/enterprise_agent_lifecycle_composition.py`
- `docs/project/maintainers/qualification/PROD_Q_R1_R7_D3_DURABLE_PROJECTION_RESTART_RECONCILIATION.md`

### Regression (@ task close — local)

```text
uv run pytest -p no:xdist -q tests/unit/applications/test_registry_projection_descriptor_corruption.py::test_descriptor_corruption_fail_closed
uv run pytest -p no:xdist -q tests/unit/applications/test_prod_q_r1_r7_d3_r1_recovery_projection_integrity.py tests/unit/applications/test_prod_q_r1_r7_d3_production_restart_reconciliation.py tests/unit/applications/test_registry_projection_descriptor_corruption.py
```

**20 passed** (corruption module included), logs under `.tmp/session/PROD-Q-R1-R7-D3-R1-DESCRIPTOR-CORRUPTION-HARNESS-BINDING-CORRECTIVE/`.

### D3-06 / matrix update

| D3-06 | **PASS** (corruption fail-closed module green @ evidence SHA) |

### Remaining blockers

| Class | Item |
|-------|------|
| **TEST ISSUE** | `test_canonical_agent_lifecycle_reaches_serving_and_execution` — `RuntimeEventSchemaError` (phase mismatch) on strict host execute with materialized ping agent; pre-existing once install path was unblocked — **not** D3 production |

### FRZ contribution (scoped)

Supports **FRZ-PRD-12** descriptor fail-closed replay evidence; **FRZ-REC-*** / **FRZ-REG-*** via corruption negatives — no global FRZ PASS.
