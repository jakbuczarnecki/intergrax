# EBH-6 — Protocol 03 independent replay bundle

Auditor-accessible machine exports @ **`TEST_EXECUTION_SHA`** in [`MANIFEST.json`](MANIFEST.json).

| Artifact | Contents |
| --- | --- |
| [`p03-handoff.json`](p03-handoff.json) | 133 A–J groups, 23 families, 8 L9 blockers, tier census, certification banner |
| [`pre-cert-export.json`](pre-cert-export.json) | 133 pair rows + 19 T2 bundle rows with full pytest nodeids |
| [`pytest-transcript.txt`](pytest-transcript.txt) | Full `tests/qualification/ebh_6/` verbose transcript (**272 passed** — do not overwrite) |
| [`l9-pytest-transcript.txt`](l9-pytest-transcript.txt) | Mandatory **8 L9 BLK** gate nodeids (10 unique tests) @ `executions.l9_blocker_gates_b3_b4_b5` |
| [`l9-nodeid-attestation.json`](l9-nodeid-attestation.json) | Per-nodeid PASS / MISSING_OR_FAIL vs transcript |

**Dual execution** — see `executions` in [`MANIFEST.json`](MANIFEST.json): `ebh6_qualification_suite` + `l9_blocker_gates_b3_b4_b5`.

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

**Regenerate full EBH-6 qual export** (does not re-run L9 batch):

```powershell
git checkout development
uv run pytest tests/qualification/ebh_6/ -v --tb=no 2>&1 | Tee-Object -FilePath docs/project/maintainers/qualification/ebh_6_p03_independent_replay/pytest-transcript.txt
uv run python scripts/qualification/export_ebh6_p03_independent_replay_bundle.py
```

Reconciliation narrative: [`EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md`](../EBH_6_PROTOCOL_03_EXACT_FINDINGS_RECONCILIATION.md).

**Agent forbidden:** Protocol 03 PASS · EBH-6 CLOSED · E02-AUD-01 ACCEPT.
