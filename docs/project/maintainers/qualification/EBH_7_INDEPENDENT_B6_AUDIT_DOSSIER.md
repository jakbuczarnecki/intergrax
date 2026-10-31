# EBH-7 — Independent B6 audit dossier (pre-review)

| Field | Value |
| --- | --- |
| **Agent status** | B6-F10 rebuilt as computed certification (closed normative sets C1–C5) · **READY FOR INDEPENDENT B6 RE-AUDIT** — F10 execution attestation of GHA run 41231430749 clean |
| **Historical audited HEAD (superseded)** | see `EBH_7_SHA_LADDER.md` pre-B5 dossier generation |
| **TEST_EXECUTION_SHA (certified L3 code)** | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` |
| **B5 TEST_EXECUTION_SHA (historical)** | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| **B5 canonical GHA replay (historical)** | run `41214574614` @ checkout `210accb224aad276a472307c8fefe3798bd44f10` |
| **B6-F10 GHA replay** | run `41231430749` @ checkout `e6ff57b13848f70c964ab141cba5bd89b50e8315` — 581 passed / 9 skipped / 0 failed; F10 attestation PASS (56 nodes) — [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](ebh_7_p03_independent_replay/EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) |
| **B6 PASS** | **NOT ISSUED** — external auditor only |
| **OD-EBH7-08** | **OPEN** — after independent B6 PASS |

## Evidence ladder

| Artifact | Role |
| --- | --- |
| [`ebh_7_p03_independent_replay/MANIFEST.json`](ebh_7_p03_independent_replay/MANIFEST.json) | TEST_EXECUTION_SHA, profile, digests |
| [`EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md`](EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md) | Protocol 03 B2 checklist |
| [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md) | ECAP C1–C5 computed certification (generated; read-only gate) |
| [`EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md`](EBH_7_ECAP_CURRENT_HEAD_AUDIT_REPORT.md) | ECAP §14 platform report (C5; computed block) |
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

The authoritative command is `pytest_profile.command` in [`MANIFEST.json`](ebh_7_p03_independent_replay/MANIFEST.json). It now executes the whole `tests/qualification/ebh_7/` directory: the earlier profile listed four `ebh_7` files before the directory argument, and pytest 8 de-duplicated the directory down to those four files (195-node runs, including run **41214574614**; superseded run **41212701986**). The profile also binds the F10 regression nodes (DUP-X, CONFIG-X, EBH-5/6 exit gates, tenant/invoker/state regressions; e.g. `test_ebh_5_dual_slug_rpl_e_operational_gates.py`, `test_dup_x_closed_world_integrity.py`).

After pytest, `scripts/qualification/attest_ebh7_f10_execution.py` checks the JUnit report: every PROFILE_BOUND F10 evidence node must have been executed and passed at TEST_EXECUTION_SHA; a missing or failed node fails the workflow.

## B6 corrective scope (agent)

- **B6-F01–F09:** prior wave — see findings register resolutions (not reopened).
- **B6-F10:** computed certification — `_ebh7_f10_canonical_inputs.py` (closed sets from canonical sources) · `_ebh7_f10_ecap_normative_matrix.py` (evaluation) · 15 adversarial gates.
- **B6-F11:** dossier / traceability / handoff sync (static acceptance preserved).
- **F10-derived findings:** F10-C3-REG-01, F10-C2-OWNER-DELTA, F10-C1-TIER-EXC-01, F10-C1-STALE-GATE-REG, F10-C4-STATE-REG-01, F10-P03-PROFILE-SHADOW — see §D of the ECAP report.

## Explicit non-claims

Independent B6 PASS, OD-RPL-G2 global approval, OD-EBH7-08, parent CLOSED, EVOL-X, global FRZ PASS.
