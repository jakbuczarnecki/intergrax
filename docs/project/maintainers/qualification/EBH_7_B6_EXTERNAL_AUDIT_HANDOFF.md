# EBH-7 — External independent B6 audit handoff

**Agent status:** G01–G09 **PASS** (0 OPEN_FINDING). L3 profile binds F07 operational gates (8). **Not** B6 PASS · **Not** OD-EBH7-08.

## Checkout and replay

```powershell
git fetch origin development
git checkout 4eb7eb1a89f82215d5356aa7d000198e432777b0
uv sync --frozen --extra dev-ci
uv run pytest tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py tests/qualification/ebh_7/test_ebh_7_rpl_dual_slug_catalog_coverage.py tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py tests/qualification/ebh_7/test_ebh_7_certification_mutation_fail_closed.py tests/qualification/ebh_7/ -q --tb=line
```

After evidence publication @ HEAD, verify `git merge-base --is-ancestor 4eb7eb1a89f82215d5356aa7d000198e432777b0 HEAD`.

## SHA roles (Protocol 03)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA (ancestor) | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| EBH7_CLOSURE_IMPLEMENTATION_SHA (lineage) | `d4937889fc0b07515649dbdddf602df2a4415489` |
| EVIDENCE_PUBLICATION_SHA | see [`MANIFEST.json`](ebh_7_p03_independent_replay/MANIFEST.json) |

## Primary artifacts

| Artifact | Purpose |
| --- | --- |
| [`EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md`](EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) | Protocol 03 B2 checklist |
| [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](ebh_7_p03_independent_replay/EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) | G01–G09 PASS / N/A / OPEN (SSOT) |
| [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](ebh_7_p03_independent_replay/B6_EVIDENCE_CONSISTENCY_MATRIX.md) | Counts + non-claims |
| [`EBH_7_INDEPENDENT_B6_AUDIT_DOSSIER.md`](EBH_7_INDEPENDENT_B6_AUDIT_DOSSIER.md) | Dossier scope |
| [`ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json`](ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json) | SHA256 ladder |

## Executable gates (sample)

```powershell
uv run pytest tests/qualification/ebh_7/test_ebh_7_closure_evidence_audit_gates.py tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py -q
```

## B6 outcome record

- **PASS:** create [`EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md); set `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` in MANIFEST.
- **FAIL:** register findings in `tests/qualification/ebh_7/_gate_findings_register.py` — do **not** agent-self-issue PASS.

## OD-EBH7-08

Operator-only after B6 PASS: [`EBH_7_OD_EBH7_08_CLOSURE_PROCEDURE.md`](EBH_7_OD_EBH7_08_CLOSURE_PROCEDURE.md).
