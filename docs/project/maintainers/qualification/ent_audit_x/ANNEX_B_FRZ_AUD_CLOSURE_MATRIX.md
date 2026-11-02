# ANNEX B — FRZ-AUD-01..10 closure evidence matrix

**START_HEAD:** `8a41d32043b89e555d7c18e052cf7022b8c94681`

| Criterion | Code deliverable | Positive test | Negative test | Execution evidence | Impl SHA | Independent auditor | FRZ promotion gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FRZ-AUD-01 | `EnterpriseAuditRunner` + CLI | `tests/qualification/ent_audit_x/integration/test_ent_audit_runner_exists.py` | `test_ent_audit_runner_missing_neg` | L2 log + manifest | TBD @ B2 | Protocol 03 ENT-AUDIT-X | W12 after B6 PASS |
| FRZ-AUD-02 | Modular `AuditCheck` registry | `test_ent_audit_registry_contract` | `test_ent_audit_monolith_script_neg` | registry unit output | TBD @ B1/B3 | same | checklist row update post B6 |
| FRZ-AUD-03 | FRZ/invariant coverage matrix | `test_ent_audit_frz_coverage_complete` | `test_ent_audit_unmapped_frz_neg` | `coverage_matrix.v1.json` | TBD @ B4/B8 | same | W12 |
| FRZ-AUD-04 | JSON + Markdown reports | `test_ent_audit_report_schemas` | `test_ent_audit_report_schema_drift_neg` | report artifacts | TBD @ B8 | same | W12 |
| FRZ-AUD-05 | Exact SHA attribution | `test_ent_audit_manifest_sha_fields` | `test_ent_audit_stale_sha_neg` | manifest @ SHA | TBD @ B7/B8 | same | W12 |
| FRZ-AUD-06 | Baseline vs current | `test_ent_audit_baseline_compare` | `test_ent_audit_non_ancestor_baseline_neg` | `baseline_comparison.v1.json` | TBD @ B7 | same | W12 |
| FRZ-AUD-07 | NEW/RESOLVED/REGRESSED/UNCHANGED | `test_ent_audit_drift_semantics` | `test_ent_audit_invalid_drift_class_neg` | drift section in JSON | TBD @ B7 | same | W12 |
| FRZ-AUD-08 | Exit codes 0–3 | `test_ent_audit_exit_code_contract` | `test_ent_audit_false_pass_neg` | CLI transcript | TBD @ B9 | same | W12 |
| FRZ-AUD-09 | FAST/FULL/FREEZE/CHANGED-SINCE | `test_ent_audit_mode_policy_matrix` | `test_ent_audit_partial_labeled_full_neg` | mode manifests | TBD @ B9 | same | W12 |
| FRZ-AUD-10 | Post-freeze scenario-wave enforcement | `test_ent_audit_ci_policy_documented` + workflow run | `test_ent_audit_branch_protection_required_neg` | GitHub settings screenshot + required check IDs | TBD @ B10 | same + **OD-ENT-02** | W12 |

## FRZ 26-family reconciliation @ checklist HEAD

| Family | ENT-AUDIT-X role @ HEAD | Status handling |
| --- | --- | --- |
| AUD | **Primary closer** FRZ-AUD-01..10 | OPEN until B12 |
| TEN | Executable TEN-* checks reuse TENANT-X | Checklist PASS **not** reissued by ENT; continuous re-orchestration |
| EVOL | CHANGED-SINCE selection reuse | FRZ-EVOL **OPEN** — EVOL B6 pending |
| PRD | Dimension B live slice | **OPEN** per Path B — `QUALIFICATION_LIMITATIONS` |
| RPL/CFG/BND/… | Orchestrated gates | Historical stage PASS = scoped evidence only (ECAP-P2) |
| FRZ (freeze mechanics) | FRZ-FRZ-11..13 depend on ENT closure | OPEN until ENT independently CLOSED |

**Count @ HEAD:** 225 criteria / 26 families per ECAP-§3.0 (recompute before audit if checklist diff).
