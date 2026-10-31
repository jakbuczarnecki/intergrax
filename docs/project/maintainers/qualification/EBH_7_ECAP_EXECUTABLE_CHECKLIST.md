# EBH-7 — ECAP executable checklist (C1–C5)

**Purpose:** Map ECAP-§2.C steps to **executable** proof — normative detail in [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md) and §14 [`EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md`](EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md).

| ECAP | Normative scope (ECAP-§2.C) | Executable gate |
| --- | --- | --- |
| C1 | Closed-world Tier-0..3 + architecture families | `test_ebh6_family_row_count_closed_world` + `test_ebh7_structural_exit_traceability_and_mechanisms` + enterprise family matrix |
| C2 | Duplicate authority / cross-domain scan | `test_ebh_7_w4_gov_dup_gates.py` (runs DUP-X subset) |
| C3 | VBN §3.0.8 + discoveries | `test_ebh6_prohibited_unresolved_zero` + `test_ebh6_findings_unclassified_zero` + CONFIG-X + INT2E |
| C4 | Drift vs accepted baseline | `test_ebh6_config_x_head_reconciliation` + P03 manifest roles + B5 GHA `41214574614` |
| C5 | ECAP §14 report + platform banner | `EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md` |

**Deterministic L3 profile (MANIFEST — includes F07 operational gates):**

```text
uv run pytest tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py tests/qualification/ebh_7/test_ebh_7_rpl_dual_slug_catalog_coverage.py tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py tests/qualification/ebh_7/test_ebh_7_certification_mutation_fail_closed.py tests/qualification/ebh_7/ -q --tb=line
```

**F10/F11:** `test_ebh_7_b6_f10_f11_corrective_gates.py` @ replay HEAD (qualification infra — outside TEST_EXECUTION_SHA).

**Non-claims:** B6 PASS · OD-EBH7-08 · EBH-7 CLOSED · global FRZ PASS.
