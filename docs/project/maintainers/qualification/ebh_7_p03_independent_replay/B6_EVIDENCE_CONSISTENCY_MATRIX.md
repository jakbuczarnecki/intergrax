# EBH-7 — B6 evidence consistency matrix (P03)

**Generated for independent reconciliation @ HEAD ancestry of TEST_EXECUTION_SHA.**  
**Authority:** executable gates cited below; this table is checked by `_ebh7_p03_evidence_integrity.py`.

## SHA roles (Protocol 03)

| Role | SHA |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA (pre-B6 terminal closure) | `d4937889fc0b07515649dbdddf602df2a4415489` |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| TEST_EXECUTION_SHA (= replay tree) | `74e2191dd4eaad544b2146f505fe691d459997b3` |
| EBH7_B5_TEST_EXECUTION_SHA (historical) | `4eb7eb1a89f82215d5356aa7d000198e432777b0` |
| EVIDENCE_PUBLICATION_SHA | see `MANIFEST.json` |
| Repository HEAD (doc pin chain) | see `ARTIFACT_DIGESTS.json` `HEAD` |

## Normative outcomes (effective)

| Domain | Count | Gate / SSOT |
| --- | --- | --- |
| RPL-H QUALIFIED | **56 / 56**; B6-open **0** | `_rpl_h_normative_evidence.py`, `test_ebh7_batch_c_rpl_semantic_verdict_counts_recorded` |
| TEN-E2E EXACT / OPEN | **26 EXACT / 0 OPEN** | `_semantic_disposition_registry.py`, `semantic_exact_counts()` |
| XDM EXACT / OPEN | **34 EXACT / 0 OPEN** | `_semantic_proof_binding_audit.py`, `test_ebh7_semantic_exact_blocker_counts_registered` |
| XDM COMPOSED_PROXY | **0** | `assert_ebh7_ten_xdm_evidence_ssot` |
| PROHIBITED unresolved | **0** | `test_ebh6_prohibited_unresolved_zero` (L3 profile) |

## L3 replay (MANIFEST `pytest_profile.last_replay`)

Until the B6-F10 GitHub Actions replay is recorded, `last_replay` is the historical B5 run (`4eb7eb1a…`, `ebh_7` directory shadowed — F10-P03-PROFILE-SHADOW).

| Field | Value |
| --- | --- |
| Passed | **195** |
| Skipped | **0** |
| Xfailed | **0** |
| Profile | `MANIFEST.json` → `pytest_profile.command` |
| Log | `.tmp/session/ebh-7-final-cert/l3-final.log` |

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
| Independent B6 PASS | **NOT ISSUED** (agent / repo non-claim) |
| OD-EBH7-08 | **OPEN** |
| Banner | `MANIFEST.json` → `certification_banner` (READY FOR INDEPENDENT B6) |

## Historical superseded (provenance only)

| Item | Superseded state | Effective state |
| --- | --- | --- |
| TEN semantic | 14 EXACT / 12 OPEN @ W1W2 recon | 26 EXACT / 0 OPEN |
| XDM semantic | 16 EXACT / 18 OPEN @ correction wave | 34 EXACT / 0 OPEN |
| RPL Package C review | 51× PARTIAL agent inventory | 56× QUALIFIED normative |

See [`EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md`](../EBH_7_INDEPENDENT_AUDIT_CORRECTIVE_FINDINGS.md) § B6-F03 / F04.
