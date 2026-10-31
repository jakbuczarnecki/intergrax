# EBH-7 — Independent B6 audit dossier (pre-review)

| Field | Value |
| --- | --- |
| **Agent status** | G01–G09 closure audit PASS · **F10/F11 remediated — READY FOR INDEPENDENT B6 RE-AUDIT** |
| **Historical audited HEAD (superseded)** | see `EBH_7_SHA_LADDER.md` pre-B5 dossier generation |
| **TEST_EXECUTION_SHA (certified L3 code)** | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| **B5 canonical GHA replay** | run `41214574614` @ checkout `210accb224aad276a472307c8fefe3798bd44f10` |
| **B6 PASS** | **NOT ISSUED** — external auditor only |
| **OD-EBH7-08** | **OPEN** — after independent B6 PASS |

## Evidence ladder

| Artifact | Role |
| --- | --- |
| [`ebh_7_p03_independent_replay/MANIFEST.json`](ebh_7_p03_independent_replay/MANIFEST.json) | TEST_EXECUTION_SHA, profile, digests |
| [`EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md`](EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) | Protocol 03 B2 checklist |
| [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md) | ECAP C1–C5 normative proof bindings (B6-F10) |
| [`EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md`](EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md) | ECAP §14 platform report (C5) |
| [`EBH_7_ECAP_EXECUTABLE_CHECKLIST.md`](EBH_7_ECAP_EXECUTABLE_CHECKLIST.md) | L3 + ECAP executable map |
| [`EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md`](EBH_7_FIXED_SUBSTRATE_RATIFICATIONS.md) | OD-RPL-G2 ratifications |
| ADR-EBH-7-INTERNAL-* | INTERNAL composition + PROD-Q gate |
| [`EBH_7_B6_EXTERNAL_AUDIT_HANDOFF.md`](EBH_7_B6_EXTERNAL_AUDIT_HANDOFF.md) | External B6 replay + gates |
| [`EBH_7_B5_INTEGRITY_RECORD.md`](EBH_7_B5_INTEGRITY_RECORD.md) | Immutable P03 verification |
| [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](ebh_7_p03_independent_replay/EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) | G01–G09 SSOT |

## RPL @ terminal closure

- **56 / 56** normative QUALIFIED (0 B6-open).
- **11** terminal dispositions executed: 2 INTERNAL, 6 FIXED substrate, 3 behavioral (agent, marketplace, external_work).

## Replay (MANIFEST L3 profile — authoritative)

```text
uv run pytest tests/qualification/ebh_6/test_ebh_6_hardcoding_register_integrity.py::test_ebh6_prohibited_unresolved_zero tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py tests/qualification/ebh_5/test_ebh_5_dual_slug_rpl_e_operational_gates.py tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py tests/qualification/ebh_7/test_ebh_7_int2e_bundle_regression.py tests/qualification/ebh_7/test_ebh_7_rpl_dual_slug_catalog_coverage.py tests/qualification/ebh_7/test_ebh_7_certification_ssot_gates.py tests/qualification/ebh_7/test_ebh_7_certification_mutation_fail_closed.py tests/qualification/ebh_7/ -q --tb=line
```

Independent CI: [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](ebh_7_p03_independent_replay/EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) — **41214574614** (supersedes **41212701986**).

F10/F11 gates (`test_ebh_7_b6_f10_f11_corrective_gates.py`) execute @ replay HEAD — not retroactively part of TEST_EXECUTION_SHA tree.

## B6 corrective scope (agent)

- **B6-F01–F09:** prior wave — see findings register resolutions.
- **B6-F10:** ECAP C1–C5 normative matrix + §14 report.
- **B6-F11:** dossier / traceability / handoff sync.

## Explicit non-claims

Independent B6 PASS, OD-RPL-G2 global approval, OD-EBH7-08, parent CLOSED, EVOL-X, global FRZ PASS.
