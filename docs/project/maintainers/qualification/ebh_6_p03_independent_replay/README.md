# EBH-6 — Protocol 03 independent replay bundle

Auditor-accessible machine exports @ **`TEST_EXECUTION_SHA`** in [`MANIFEST.json`](MANIFEST.json).

| Artifact | Contents |
| --- | --- |
| [`p03-handoff.json`](p03-handoff.json) | 133 A–J groups, 23 families, 8 L9 blockers, tier census, certification banner |
| [`pre-cert-export.json`](pre-cert-export.json) | 133 pair rows + 19 T2 bundle rows with full pytest nodeids |
| [`pytest-transcript.txt`](pytest-transcript.txt) | Full `tests/qualification/ebh_6/` verbose transcript (**272 passed** — do not overwrite) |
| [`l9-pytest-transcript.txt`](l9-pytest-transcript.txt) | Mandatory **8 L9 BLK** gate nodeids (10 unique tests) @ `executions.l9_blocker_gates_b3_b4_b5` |
| [`l9-pytest-transcript-fail-historical-653cdd93f.txt`](l9-pytest-transcript-fail-historical-653cdd93f.txt) | **Historical** GOV gate FAIL before L9-01 remediation (do not delete) |
| [`l9-nodeid-attestation.json`](l9-nodeid-attestation.json) | Per-nodeid PASS / MISSING_OR_FAIL vs transcript |
| [`t2-primary-gates-gap-pytest-transcript.txt`](t2-primary-gates-gap-pytest-transcript.txt) | **11** missing T2 primary gates @ `executions.t2_primary_gates_gap` |
| [`t2-primary-gates-attestation.json`](t2-primary-gates-attestation.json) | **25** unique T2 primary gates + **19** bundle rollup (gap + qual + L9 cross-ref) |

**Triple execution** — see `executions` in [`MANIFEST.json`](MANIFEST.json): `ebh6_qualification_suite` + `l9_blocker_gates_b3_b4_b5` + `t2_primary_gates_gap`.

**Regenerate L9 execution** (UTF-8 transcript required):

```powershell
$nodes = @(
  "tests/unit/runtime/architecture/test_ee_final_arch_owner_uniqueness.py::test_ee_final_arch_owner_matrix_documented",
  "tests/unit/runtime/architecture/test_ee_a1_execution_engine_ownership_certification_gate.py::test_ee_a1_p0_inventory_reports_zero_production_bypass",
  "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_execution_does_not_host_policy_engine",
  "tests/unit/runtime/architecture/test_npsc42_h1_governance_boundary_freeze.py::test_npsc42_h1_governance_planes_do_not_mint_execution_identity",
  "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py::test_no_recovery_god_object_in_production",
  "tests/unit/runtime/architecture/test_w3_c4_decision_durable_recovery_canonical_wiring.py::test_decision_durable_recovery_has_no_production_bypass",
  "tests/qualification/control_planes/ctrl_x/test_ctrl_x_r1_security_composition.py::test_r1_sec_01_canonical_composition_rejects_external_fail_open",
  "tests/unit/runtime/architecture/test_ee_b1_2_capacity_architecture_gate.py::test_ee_b1_2_capacity_contract_no_identity_mint",
  "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_published_event_reaches_bounded_sink",
  "tests/unit/runtime/security/test_tenant_scope.py::test_tenant_scope_adversarial_matrix"
)
uv run pytest @nodes -v --tb=short 2>&1 | Out-File -FilePath docs/project/maintainers/qualification/ebh_6_p03_independent_replay/l9-pytest-transcript.txt -Encoding utf8
uv run python scripts/qualification/attest_ebh6_p03_l9_execution.py
```

**Regenerate T2 primary gate gap batch + rollup** (UTF-8 gap transcript required):

```powershell
$gap = @(
  "tests/integration/agent_distribution/test_canonical_agent_lifecycle_architecture_gate.py::test_stage15_proof_files_avoid_forbidden_lifecycle_bypass_fragments",
  "tests/qualification/ebh_5/test_ebh_5_mechanical_exit_gates.py::test_ebh5_exit_inventory_complete",
  "tests/unit/architecture/test_ebh_2i_final_rescan_gate.py::test_ebh_2i_contract_surface_purity_gate_passes",
  "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_factory_module_is_sanctioned_composition_owner",
  "tests/qualification/ebh_5/test_ebh_5_rpl_g_regression_gates.py::test_ebh5_rpl_g_config_x_active_blockers_remain_empty",
  "tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py::test_hardening_3_contracts_do_not_import_runtime_except_allowlist",
  "tests/unit/runtime/events/test_enterprise_scale_resilience_w5_b2_composition_wiring.py::test_production_composition_creates_bounded_sink",
  "tests/qualification/ebh_5/test_ebh_5_proof_integrity_gates.py::test_ebh5_proof_sqlite_closed_world_accountability",
  "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_integration_unknown_slug_fail_closed",
  "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_observability_fail_closed_without_role",
  "tests/qualification/ebh_5/test_ebh_5_reachability_gates.py::test_ebh5_reach_no_second_global_provider_registry_in_rag_graph"
)
uv run pytest @gap -v --tb=short 2>&1 | Out-File -FilePath docs/project/maintainers/qualification/ebh_6_p03_independent_replay/t2-primary-gates-gap-pytest-transcript.txt -Encoding utf8
uv run python scripts/qualification/attest_ebh6_p03_t2_primary_gates.py
```

**Regenerate full EBH-6 qual export** (does not re-run L9 batch):

```powershell
git checkout development
uv run pytest tests/qualification/ebh_6/ -v --tb=no 2>&1 | Tee-Object -FilePath docs/project/maintainers/qualification/ebh_6_p03_independent_replay/pytest-transcript.txt
uv run python scripts/qualification/export_ebh6_p03_independent_replay_bundle.py
```

Reconciliation narrative: [`EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md`](../EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md).

**Agent forbidden:** Protocol 03 PASS · EBH-6 CLOSED · E02-AUD-01 ACCEPT.
