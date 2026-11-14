# EBH-7 — B6 evidence consistency matrix (P03)

**Generated for independent reconciliation @ HEAD ancestry of TEST_EXECUTION_SHA.**  
**Authority:** executable gates cited below; this table is checked by `_ebh7_p03_evidence_integrity.py`.

## SHA roles (Protocol 03)

| Role | SHA |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA (pre-B6 terminal closure) | `d4937889fc0b07515649dbdddf602df2a4415489` |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| TEST_EXECUTION_SHA (= replay tree; PDC-X P6 re-pin, T-IGX-036-W) | `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` |
| Previous TEST_EXECUTION_SHA (historical, run `41238620036`) | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` |
| EBH7_B6_F10_TEST_EXECUTION_SHA (historical, run `41231430749`) | `496016c0cf6cc5cf1f9a29d885677605dd0415d3` |
| EBH7_B5_TEST_EXECUTION_SHA (historical) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA | see `MANIFEST.json` |
| INDEPENDENT_ACCEPTANCE_BASELINE_SHA (independently accepted tree; not re-accepted at the re-pinned TEST) | `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` |
| B6 acceptance publication | see pass record + `MANIFEST.json` → `B6_ACCEPTANCE_PUBLICATION_SHA` |

## Normative outcomes (effective)

| Domain | Count | Gate / SSOT |
| --- | --- | --- |
| RPL-H QUALIFIED | **56 / 56**; B6-open **0** | `_rpl_h_normative_evidence.py`, `test_ebh7_batch_c_rpl_semantic_verdict_counts_recorded` |
| TEN-E2E EXACT / OPEN | **26 EXACT / 0 OPEN** | `_semantic_disposition_registry.py`, `semantic_exact_counts()` |
| XDM EXACT / OPEN | **34 EXACT / 0 OPEN** | `_semantic_proof_binding_audit.py`, `test_ebh7_semantic_exact_blocker_counts_registered` |
| XDM COMPOSED_PROXY | **0** | `assert_ebh7_ten_xdm_evidence_ssot` |
| PROHIBITED unresolved | **0** | `test_ebh6_prohibited_unresolved_zero` (L3 profile) |

## L3 replay (MANIFEST `pytest_profile.last_replay`)

**Re-pin replay: PENDING.** TEST_EXECUTION_SHA was re-pinned to `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec` (PDC-X P6, T-IGX-036-W) and the fresh L3 replay at this pin has not run yet; `last_replay` carries no counts. Counts, run ID and artifact digests are recorded after the replay (W2). This record is not a B6 PASS for the re-pinned tree; the independent acceptance (scoped) stays bound to `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` `dc18e36d9…`.

Previous authoritative replay (historical, kept in `MANIFEST.json` → `pytest_profile.historical_replays`): final reconciliation GitHub Actions run `41238620036` @ `73965c2ea…` (TEST `dc18e36d9…`; JUnit 634 tests; recorded in `EBH_7_P03_GITHUB_ACTIONS_REPLAY_RECORD.md`). Older historical (superseded): B6-F10 run `41231430749` @ `e6ff57b13…` (TEST `496016c0c…`, JUnit 590 tests: 581 passed / 9 skipped). Skipped tests: nine individual dispositions in [`EBH_7_SKIP_DISPOSITION_REGISTER.md`](EBH_7_SKIP_DISPOSITION_REGISTER.md). The historical B5 run (`4eb7eb1a…`, 195 passed) shadowed the `ebh_7` directory (F10-P03-PROFILE-SHADOW).

| Field | Previous replay `41238620036` (historical, TEST `dc18e36d9`) | Re-pin replay (TEST `146e7acf1`) |
| --- | --- | --- |
| JUnit tests | **634** | PENDING |
| Passed | **625** | PENDING |
| Skipped | **9** (nine individual `ACCEPTED_NA` dispositions; 0 unregistered / 0 stale) | PENDING |
| Failed / errors | **0 / 0** | PENDING |
| Xfailed | **0** | PENDING |
| Profile | `MANIFEST.json` → `pytest_profile.command` | same profile |
| Log | GHA run `41238620036` artifact `ebh7-p03-l3-dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (pytest.log, junit.xml, meta.txt, junit-summary.json, skip-disposition.json, f10-attestation.json) | PENDING |

## INT2E / ECAP

| Scope | Status |
| --- | --- |
| INT2E bundle regression | In L3 profile (`test_ebh_7_int2e_bundle_regression.py`) |
| ECAP C1–C5 + mutation | `test_b6_f10_normative_certification_computed` (computed), `test_ebh_7_b6_f10_certification_adversarial.py`, `test_ebh_7_certification_ssot_gates.py`, `test_ebh_7_certification_mutation_fail_closed.py` |

## G01–G09 closure audit

| Item | Status |
| --- | --- |
| Closure evidence matrix | [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) — **0 OPEN_FINDING** @ SSOT |
| Gate | `test_ebh_7_closure_evidence_audit_gates.py` |

## B6 readiness

| Claim | Status |
| --- | --- |
| Independent B6 PASS | **ISSUED — PASS (SCOPED)** @ [`EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md`](../EBH_7_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md) (auditor **ChatGPT**, 2026-11-01) |
| OD-EBH7-08 | **APPROVED** (2026-11-01) |
| Parent EBH-7 | **CLOSED — INDEPENDENTLY ACCEPTED (SCOPED)** |
| Banner | `MANIFEST.json` → `certification_banner` |

## Historical superseded (provenance only)

| Item | Superseded state | Effective state |
| --- | --- | --- |
| TEN semantic | 14 EXACT / 12 OPEN @ W1W2 recon | 26 EXACT / 0 OPEN |
| XDM semantic | 16 EXACT / 18 OPEN @ correction wave | 34 EXACT / 0 OPEN |
| RPL Package C review | 51× PARTIAL agent inventory | 56× QUALIFIED normative |

See [`EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md`](../EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md) § B6-F03 / F04.
