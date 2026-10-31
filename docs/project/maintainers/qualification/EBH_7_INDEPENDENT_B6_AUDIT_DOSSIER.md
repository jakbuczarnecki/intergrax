# EBH-7 — Independent B6 audit dossier (pre-review)

| Field | Value |
| --- | --- |
| **Agent status** | Final B6 evidence reconciliation complete: ECAP C1–C5 PASS (computed, attested), strict JUnit metadata, nine individual skip dispositions, P03 records reconciled, B5 integrity unchanged · **READY FOR INDEPENDENT B6 ACCEPTANCE** — final GHA L3 replay `41238620036` clean |
| **Historical audited HEAD (superseded)** | see `EBH_7_SHA_LADDER.md` pre-B5 dossier generation |
| **TEST_EXECUTION_SHA (certified L3 code)** | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (final reconciliation; = `EBH7_B6_FINAL_RECONCILIATION_IMPLEMENTATION_SHA`) |
| **B6-F10 TEST_EXECUTION_SHA (historical)** | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` |
| **B5 TEST_EXECUTION_SHA (historical)** | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| **B5 canonical GHA replay (historical)** | run `41214574614` @ checkout `210accb224aad276a472307c8fefe3798bd44f10` |
| **Final reconciliation GHA replay (authoritative)** | run `41238620036` @ checkout `73965c2eb3533dfb11ba5c57697316377f956bdd` — JUnit 634 tests: 625 passed / 9 skipped / 0 failed; F10 attestation PASS (56 nodes); skip disposition PASS (9/9); EVIDENCE `f30a90a14910afe8583081cab3b2e39480988c17` — [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](ebh_7_p03_independent_replay/EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) |
| **B6-F10 GHA replay (historical)** | run `41231430749` @ checkout `e6ff57b13848f70c964ab141cba5bd89b50e8315` — 581 passed / 9 skipped / 0 failed; F10 attestation PASS (56 nodes) — [`EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`](ebh_7_p03_independent_replay/EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md) |
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

## Final reconciliation summary (agent-attested; independent B6 decides)

| # | Item | Result | Evidence |
| --- | --- | --- | --- |
| 1 | ECAP C1–C5 | **PASS** in the verified scope (C1 32 · C2 43 · C3 105 · C4 9 · C5 3 obligations; 0 failed) | [`EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md`](EBH_7_B6_F10_ECAP_NORMATIVE_MATRIX.md), `f10-attestation.json` |
| 2 | L3 | exit 0 · 634 tests = 625 passed / 9 skipped / 0 failed / 0 errors / 0 xfailed · values read from `junit.xml` | run `41238620036` |
| 3 | F10 execution attestation | `PASS nodes=56 problems=0` | `f10-attestation.json` |
| 4 | Skipped tests | **9** — nine individual `ACCEPTED_NA` (`N/A — WITH EVIDENCE`) dispositions; 8 mandatory-scenario routes with executed-equivalent evidence PASS in the same JUnit, 1 non-mandatory local-staging gate; 0 BLOCKED / 0 unregistered / 0 stale | [`EBH_7_SKIP_DISPOSITION_REGISTER.md`](ebh_7_p03_independent_replay/EBH_7_SKIP_DISPOSITION_REGISTER.md), `skip-disposition.json` |
| 5 | JUnit metadata | `meta.txt` now `junit_parse=PASS junit_tests=634 junit_passed=625 junit_failed=0 junit_errors=0 junit_skipped=9 junit_xfailed=0` (was `junit_tests=0`); invalid/missing/inconsistent JUnit ⇒ `unknown` + FAIL | `scripts/qualification/ebh7_junit_evidence.py`, `test_ebh_7_b6_junit_skip_disposition_gates.py` (44 tests) |
| 6 | B5 integrity | immutable git-object digests PASS · executable tree identity PASS (17 allowlisted paths) · MANIFEST pins PASS · clean tracked tree PASS; verifier/allowlist unchanged | [`EBH_7_B5_INTEGRITY_RECORD.md`](EBH_7_B5_INTEGRITY_RECORD.md) |
| 7 | SHA ladder | TEST/IMPLEMENTATION `dc18e36d9…` · EVIDENCE `f30a90a14…` · replay `73965c2ea…` · historical roles preserved | [`EBH_7_SHA_LADDER.md`](EBH_7_SHA_LADDER.md) |
| 8 | Findings | F01–F11 and F10-derived findings: resolutions unchanged (register `_gate_findings_register.py`). This wave: **JUNIT-META-01** (`junit_tests=0`), **META-F07-02** (`f07_log_lines` printed twice, meaningless under `-q`; replaced by `junit_f07_operational_gate_nodes`), **SKIP-REG-03** (skip-equivalent SSOT covered 3 of 8 routes; now 8 + register), **P03-STALE-04** (stale replay-record status/default checkout) — all resolved and verified in run `41238620036`. **0 unresolved.** | this dossier, replay record |

## Instruction for the independent auditor

1. `git fetch origin development`; `git merge-base --is-ancestor dc18e36d971c93c8f631fbc86f73149d1b81b4d7 HEAD` and `git diff --name-only dc18e36d971c93c8f631fbc86f73149d1b81b4d7 HEAD` (documentation / `.github/` / B5 allowlist only).
2. Download artifact `ebh7-p03-l3-dc18e36d971c93c8f631fbc86f73149d1b81b4d7` of run `41238620036`; compare SHA-256 values in the replay record; confirm `meta.txt` totals equal `junit.xml` and `junit_evidence_status=PASS`.
3. `python scripts/qualification/ebh7_junit_evidence.py skip-disposition --junit junit.xml` → `PASS` (9 skipped, 9 registered); review each record's justification in the skip register.
4. Re-run the MANIFEST profile or dispatch `profile=ebh7_p3_l3`; verify ECAP C1–C5 via the F10 matrix and `f10-attestation.json`.
5. Issue or withhold B6 PASS, `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` and OD-EBH7-08 — **not** the agent.
## Explicit non-claims

Independent B6 PASS, OD-RPL-G2 global approval, OD-EBH7-08, parent CLOSED, EVOL-X, global FRZ PASS.
