# HIST-02 Current-HEAD Qualification Report

Target: pinned checkout `D:/Projekty/_wt/intergrax/pin-ba07eba`, HEAD `ba07ebaa34d04633bd60510f68c18346b1ef160d` (verified before and after; `git status --short` empty, only ignored .venv/__pycache__ artefacts). Heavy lock held for each run and released.

## Runs
| Run | Command (abridged) | Result |
|---|---|---|
| attempt0 | pytest ... -p no:cacheprovider | FAILED-TO-RUN: `Unknown config option: cache_dir`, no tests ran. The logged `EXIT=0` was tee's status, not pytest's. Log: replay_attempt0_cacheprovider_error.log |
| run1 | one `uv run pytest` over 27 paths, `--basetemp D:/Projekty/_wt/_scratch/hist02-replay -o cache_dir=<scratch>` | pytest exit **1**: 818 collected, 813 passed, 5 failed, 0 errors, 0 skipped, 431.09s. Log: replay_ba07eba.log |
| collect | `--collect-only -q` (no tests executed) | 818 collected, exit 0; used only for per-path counts |
| rerun4 | the 4 environment/harness-affected ids, `--extra dev-ci`, no --basetemp | pytest exit 0, 4 passed. pyproject extra `dev-ci` (contains duckdb>=1.0,<2); the pin .venv was extended. Log: replay_rerun4_ba07eba.log |

## Lock protocol incident (HIST-02 fault)
The collect-only pass ran WITHOUT the heavy lock (my retry loop never acquired it and had no abort guard), and its cleanup then deleted HIST-03's live lock (holder.txt and directory). The 4-id rerun's `mkdir` therefore succeeded because the directory was absent, not because HIST-03 had finished. That rerun overlapped HIST-03's still-running 27-path pytest in the same pin venv, and I extended that venv with `--extra dev-ci` mid-run. Run1 itself held a valid lock and was released correctly. HIST-03's results may need checking for interference. Details in WORKLOG.md (P10).

## Failure triage
1. 3x EBH-5 package_c_v2 dual-slug relational_store: `duckdb` not installed in default env. Environment gap; pass with `--extra dev-ci` (F-014).
2. test_ue_9br1_synthetic_mint_execution_id_call_detected: harness artefact of `--basetemp` outside the repo root; passes without it (F-014).
3. **test_ebh_4_r1_production_nexus_imports_outside_ee_are_zero: REAL product failure** (F-013): `intergrax/runtime/tools/tool_operation_recovery.py:22` imports `intergrax.runtime.nexus.errors.error_codes`. Introduced by TOOL-LIFE-X Batch C `a5f813a2c` (2026-10-24). The gate has been unchanged since 2026-09-30. It is a QUAL-X P5 F01 mandatory module. Not fixed, not explained away.

## Per-path results (run1)
| Path | Collected | Passed | Failed | Disposition |
|---|---|---|---|---|
| qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py | 86 | 83 | 3 | FAIL_ENV_OR_HARNESS_THEN_PASS_ON_RERUN |
| qualification/ebh_5/test_ebh_5_behavioral_proofs.py | 17 | 17 | 0 | PASS |
| qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py | 6 | 6 | 0 | PASS |
| qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py | 5 | 5 | 0 | PASS |
| qualification/ebh_6/test_ebh_6_contracts_typing_qualification.py | 11 | 11 | 0 | PASS |
| qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py | 8 | 8 | 0 | PASS |
| qualification/ebh_6/test_ebh_6_q2d2_dual_path.py | 7 | 7 | 0 | PASS |
| qualification/ebh_7/test_ebh_7_rpl_final_convergence_gates.py | 1 | 1 | 0 | PASS |
| qualification/ebh_7/test_ebh_7_w5_vbn_gates.py | 6 | 6 | 0 | PASS |
| qualification/governance/gr11/test_gr11_plugin_enterprise_certification.py | 29 | 29 | 0 | PASS |
| unit/applications/test_catalog_hot_reload_governance.py | 20 | 20 | 0 | PASS |
| unit/architecture/test_ebh_2a_public_contract_boundary_gate.py | 17 | 17 | 0 | PASS |
| unit/architecture/test_ebh_2i_final_rescan_gate.py | 12 | 12 | 0 | PASS |
| unit/architecture/test_ebh_3_dependency_ownership_gate.py | 13 | 13 | 0 | PASS |
| unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py | 8 | 7 | 1 | FAIL_REAL (product) |
| unit/context/test_context_plugin_registry.py | 5 | 5 | 0 | PASS |
| unit/integrations/registry/test_catalog_revision_and_mutation.py | 12 | 12 | 0 | PASS |
| unit/integrations/test_external_plugin.py | 2 | 2 | 0 | PASS |
| unit/integrations/test_p2_003_explicit_contract_specs.py | 457 | 457 | 0 | PASS |
| unit/runtime/architecture/test_ebh_2f_r2_integration_plugin_contract_typing_gate.py | 24 | 24 | 0 | PASS |
| unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py | 2 | 2 | 0 | PASS |
| unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py | 8 | 8 | 0 | PASS |
| unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py | 9 | 9 | 0 | PASS |
| unit/runtime/architecture/test_harness_w6_runtime_intelligence_boundary_gate.py | 10 | 10 | 0 | PASS |
| unit/runtime/architecture/test_ue_9br1_runtime_event_legacy_retirement_gate.py | 8 | 7 | 1 | FAIL_ENV_OR_HARNESS_THEN_PASS_ON_RERUN |
| unit/runtime/events/test_evidence_persistence_boundary.py | 23 | 23 | 0 | PASS |
| unit/runtime/nexus/test_authoritative_decision_exposure_propagation.py | 12 | 12 | 0 | PASS |

## What this does and does not show
- The 813 passes show the existing scoped gates still hold at the pin. They promote nothing; all 35 rows stay OPEN.
- Effect on rows: BND-04 -> IMPLEMENTATION_GAP, BND-01 downgraded to QUALIFICATION_GAP (its P5 F01 selection is not fully green); BND-02/03 annotated. Counts now: {'QUALIFICATION_GAP': 8, 'EVIDENCE_GAP': 14, 'AUTHORITY_GAP': 1, 'IMPLEMENTATION_GAP': 1, 'VALID_SCOPED_PASS_CANDIDATE': 7, 'SSOT_DRIFT': 1, 'INDEPENDENT_AUDIT_GAP': 3}.
- Green dual-slug tests (with dev-ci) do not discharge T-B7-VBN-INDEPENDENT-ACCEPT for RPL-01..04.
- Not shown: F-001, F-005, F-007, F-008, F-009 (authority/auditor/qualification gaps) are not testable by this replay.
- This is evidence at the pin by HIST-02 (implementer-side). It is not an independent audit.
