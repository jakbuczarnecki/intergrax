# EBH-7 — L3 skipped-test disposition register

**Status:** individual dispositions READY for independent B6 review · **not** a B6 PASS.

Source of truth: [`tests/qualification/ebh_7/_ebh7_skip_disposition_register.py`](../../../../../tests/qualification/ebh_7/_ebh7_skip_disposition_register.py). Verified against the real L3 `junit.xml` by `scripts/qualification/ebh7_junit_evidence.py skip-disposition` (artifact `skip-disposition.json`): every skipped node must be registered, every evidence node must be PASS in the same JUnit, unregistered or stale skips fail the replay. No JUnit total is hardcoded.

Disposition vocabulary: `ACCEPTED_NA` = approved `N/A — WITH EVIDENCE` (Path B; no live-provider test implied).

| # | NODE_ID | MANDATORY | DISPOSITION | AUDIT_STATUS | SOURCE_SHA |
| --- | --- | --- | --- | --- | --- |
| 1 | `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-04]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 2 | `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-07]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 3 | `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-11]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 4 | `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-14]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 5 | `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-010]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 6 | `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-016]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 7 | `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-017]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 8 | `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-029]` | YES | ACCEPTED_NA | READY | `8bbda35dd168` |
| 9 | `tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py::test_ebh7_w9_proof_staging_dir_exists` | NO | ACCEPTED_NA | READY | `496016c0cf6c` |

## 1. `EBH7-TEN-E2E-04`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-04]`
- **SKIP_REASON:** async/governed_execution proofs run in tenant_platform_e2e module
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:27 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-TEN-E2E-04 (Exec→Tool, DENY cross-tenant tool scope) — tenant E2E matrix 26/26 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_platform_tenant_boundary_proofs.py::test_ebh7_ten_e2e_04_exec_tool_tenant_boundary_tool_runtime`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Registered proof `test_ebh7_ten_e2e_04_exec_tool_tenant_boundary_tool_runtime` is `async` and cannot run via synchronous `execute_proof`; it is executed natively (pytest-asyncio) in the tenant_platform_e2e module. The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 2. `EBH7-TEN-E2E-07`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-07]`
- **SKIP_REASON:** proof requires tmp_path — run in dedicated fixture test
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:29 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-TEN-E2E-07 (Persistence→Cache, DENY cross-tenant durable load) — tenant E2E matrix 26/26 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_scenario_07_durable_cross_tenant`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof `test_tx_r3_q07_adversarial_durable_load_cross_tenant` needs `tmp_path`. The dedicated fixture test `test_ebh7_w6_tenant_e2e_scenario_07_durable_cross_tenant` runs the same registered callable with a real `tmp_path` (also recorded as `executed_equivalent_node_id` in the semantic disposition SSOT). The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 3. `EBH7-TEN-E2E-11`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-11]`
- **SKIP_REASON:** async/governed_execution proofs run in tenant_platform_e2e module
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:27 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-TEN-E2E-11 (Trace→Evidence, REJECT cross export) — tenant E2E matrix 26/26 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_platform_tenant_boundary_proofs.py::test_ebh7_ten_e2e_11_trace_attribution_cross_tenant_rejected`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof requires the `governed_execution` pytest fixture; it is executed natively with that fixture in the tenant_platform_e2e module (also recorded as `executed_equivalent_node_id`). The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 4. `EBH7-TEN-E2E-14`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py::test_ebh7_w6_tenant_e2e_executes_registered_proof[EBH7-TEN-E2E-14]`
- **SKIP_REASON:** proof requires tmp_path — run in dedicated fixture test
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w6_tenant_e2e_behavioral_gates.py:29 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-TEN-E2E-14 (Provider→Persistence, PRESERVE tenant on backend switch) — tenant E2E matrix 26/26 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/tenant_platform_e2e/test_ebh7_platform_tenant_boundary_proofs.py::test_ebh7_ten_e2e_14_backend_switch_preserves_tenant`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof `test_ebh7_ten_e2e_14_backend_switch_preserves_tenant(tmp_path)` is executed natively by pytest with a real `tmp_path` (sqlite/duckdb dual slug, distinct tenant binding). The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 5. `EBH7-XDM-010`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-010]`
- **SKIP_REASON:** proof requires tmp_path
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py:25 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-XDM-010 (Backend x Persistence, REJECT sqlite bleed) — cross-domain matrix 34/34 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_xdm_010_sqlite_behavioral`
  - `tests/qualification/ebh_7/cross_domain_e2e/test_ebh7_cross_domain_platform_proofs.py::test_ebh7_xdm_010_backend_persistence_sqlite_behavioral`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof needs `tmp_path`; executed natively in the cross-domain platform module and again by the dedicated W7 fixture test (also `executed_equivalent_node_id`). The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 6. `EBH7-XDM-016`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-016]`
- **SKIP_REASON:** proof requires tmp_path
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py:25 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-XDM-016 (VBN x Composition, REJECT hardcoded slug) — cross-domain matrix 34/34 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/cross_domain_e2e/test_ebh7_cross_domain_platform_proofs.py::test_ebh7_xdm_016_vbn_composition_factory_behavioral`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof `test_ebh7_xdm_016_vbn_composition_factory_behavioral(tmp_path)` is executed natively by pytest with a real `tmp_path` (Package C v2 relational dual slug + unregistered fail-closed). The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 7. `EBH7-XDM-017`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-017]`
- **SKIP_REASON:** proof requires tmp_path
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py:25 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-XDM-017 (Failure x Diagnostics, REJECT tenant leak) — cross-domain matrix 34/34 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/cross_domain_e2e/test_ebh7_cross_domain_platform_proofs.py::test_ebh7_xdm_017_failure_diagnostics_tenant_leak`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof `test_ebh7_xdm_017_failure_diagnostics_tenant_leak` (Trace P5 configuration/profile tenant mismatch fail-closed) is executed natively by pytest in the cross-domain platform module. The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 8. `EBH7-XDM-029`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_executes_registered_proof[EBH7-XDM-029]`
- **SKIP_REASON:** proof requires tmp_path
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py:25 pytest.skip (SKIP_ROUTED_TO_EQUIVALENT)`
- **REQUIREMENT:** EBH7-XDM-029 (Integration x External, DENY DI bypass) — cross-domain matrix 34/34 EXACT
- **MANDATORY:** YES
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/test_ebh_7_w7_cross_domain_behavioral_gates.py::test_ebh7_w7_cross_domain_xdm_029_integration_external_di_boundary`
  - `tests/qualification/ebh_7/cross_domain_e2e/test_ebh7_cross_domain_platform_proofs.py::test_ebh7_xdm_029_integration_external_di_boundary`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md`
- **SOURCE_SHA:** `8bbda35dd1686b15bd6959b732ade3ba0f268632` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** Proof needs `tmp_path`; executed natively in the cross-domain platform module and again by the dedicated W7 fixture test. The skipped parametrization would call the registered proof callable through `execute_proof()` (plain function call, no pytest fixture/async support). The identical callable is collected and executed natively by pytest in the evidence node(s), which must be PASS in the same JUnit. No obligation is dropped; the skipped parameter is a duplicate, fixture-incapable route.
- **AUDIT_STATUS:** READY

## 9. `EBH-7 W9`

- **NODE_ID:** `tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py::test_ebh7_w9_proof_staging_dir_exists`
- **SKIP_REASON:** gitignored local proof staging (.tmp/) — absent in a clean checkout; not replayable evidence
- **Skip source:** `tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py:26 pytest.skip (SKIP_LOCAL_STAGING_ONLY)`
- **REQUIREMENT:** EBH-7 W9 evidence dossier index lists `.tmp/proof/ebh-7-materialization/MANIFEST.md` as a staged, gitignored local artifact (EBH_7_EVIDENCE_DOSSIER_INDEX.md) — not a replay-time obligation
- **MANDATORY:** NO
- **DISPOSITION:** ACCEPTED_NA (`N/A — WITH EVIDENCE`)
- **EVIDENCE (must be PASS in the same JUnit):**
  - `tests/qualification/ebh_7/test_ebh_7_w9_dossier_gates.py::test_ebh7_w9_dossier_index_present`
- **Evidence documents:**
  - `docs/project/maintainers/qualification/EBH_7_EVIDENCE_DOSSIER_INDEX.md`
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json`
  - `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json`
- **SOURCE_SHA:** `496016c0cf6cc5cf1f9a29d885677605dd0415d3` (last commit touching the skip-bearing test file @ pre-reconciliation TEST `496016c0c`)
- **JUSTIFICATION:** The gate inspects an operator-local, gitignored staging directory that by definition does not exist in a clean checkout and is not replayable evidence. Replayable W9 evidence is the committed dossier index (node `test_ebh7_w9_dossier_index_present`, PASS) plus B5 immutable digest verification of the committed dossier artifacts and MANIFEST (git objects @ replay HEAD). Skipping does not hide an environment or dependency defect: it is an explicit, message-bearing `pytest.skip` on a path-existence check, and the staging directory is never required by any other gate. Path B: no live-provider test is implied.
- **AUDIT_STATUS:** READY

## Non-claims

B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`, OD-EBH7-08, EBH-7 CLOSED, EVOL-X, global FRZ PASS.
