# EBH-7 — Independent audit corrective findings

**Trigger:** B6 BLOCKED — automatic `proof_module` → EXACT_PROOF promotion @ closure wave (`f913944ff`).

## B6-F01 / F02 / sweep (@ `d1bdcfa46` audit → fixed @ `8afff322`)

| ID | Defect | Corrective evidence |
| --- | --- | --- |
| **B6-F01** | External-work dual instance (same class) | `QualExternalWorkAlphaIntegration` / `QualExternalWorkBetaIntegration` + `IntegrationProfile` |
| **B6-F02** | PROD-Q negative = source-string check | `_internal_rpl_behavioral_proofs` + `_require_venv_production_adapter` |
| **Sweep** | SQLite conversational FIXED proof / store API | `SQLiteConversationalMemoryStore.save_memory` + async roundtrip proof |

## B6-F03 — TEN/XDM vs corrective doc (reconciled)

**Finding:** This file previously listed **12 TEN OPEN** and **18 XDM OPEN** under “Correction @ semantic disposition SSOT” without marking that state as **superseded**, while terminal artifacts and `_semantic_disposition_registry.py` already reflect **26/26** and **34/34 EXACT**.

| Layer | TEN-E2E | XDM | Authority |
| --- | --- | --- | --- |
| **Historical** (fail-closed restore @ `edebda9d`) | 14 EXACT / **12 OPEN** | 16 EXACT / **18 OPEN** | Wave after auto-EXACT removal — **superseded** |
| **Effective @ TEST_EXECUTION_SHA** | **26 EXACT / 0 OPEN** | **34 EXACT / 0 OPEN** | `_semantic_disposition_registry.py` + `semantic_exact_counts()`; gates `test_ebh7_semantic_exact_blocker_counts_registered`, `test_ebh7_b6_readiness_mandatory_exact_coverage` |

**RPL 51× PARTIAL:** historical `_rpl_h_semantic_review.py` Package C inventory — **superseded** by normative layer **56/56 QUALIFIED** (`_rpl_h_normative_evidence.py`, `normative_rpl_b6_open_concern_ids()` → 0).

Provenance preserved in [`EBH_7_W1W2_QUALIFICATION_RECONCILIATION_RECORD.md`](EBH_7_W1W2_QUALIFICATION_RECONCILIATION_RECORD.md) and gap-closure records; **do not** treat OPEN counts in this section as current certification state.

<a id="TENANT-X"></a>

### W6-TENANT-X-FINAL-READY — historical supersession

Single TENANT-X readiness test is **not** platform E2E. Effective proof map: `_semantic_disposition_registry.py` + gates cited under **Effective @ TEST_EXECUTION_SHA** above.

<a id="readiness-reconcile"></a>

### READINESS-TABLE-PASS — historical supersession

Readiness tables distinguish materialization vs qualification columns; see B6-F03 effective layer and closure audit G04–G05 @ `TEST_EXECUTION_SHA`.

## B6-F04 — P03 TEST vs closure SHA (reconciled)

| Role | SHA | Meaning |
| --- | --- | --- |
| `EBH7_CLOSURE_IMPLEMENTATION_SHA` | `d4937889fc0b07515649dbdddf602df2a4415489` | Terminal operator-disposition closure (pre-B6 code tree) — **lineage only** |
| `EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA` | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` | B6 F01/F02 + INTERNAL/FIXED proofs + SQLite store fix |
| `TEST_EXECUTION_SHA` | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` | **Must equal B6 corrective** — L3 was **not** executed on `d4937889` alone after corrective landed |

Enforced by `tests/qualification/ebh_7/_ebh7_p03_evidence_integrity.py`.

## Historical semantic-correction table (superseded — do not use for B6 readiness)

| Finding | Resolution @ correction wave | Superseded by |
| --- | --- | --- |
| Auto-EXACT in `_semantic_proof_binding_audit.py` | Removed; registry authoritative | Batch D + gap closure @ `8bbda35d` |
| ECAP false 26/26 + 34/34 | Restored 14+16 EXACT, 12+18 OPEN | **26/0 + 34/0** @ registry |
| W6/W7 pending==0 gates | Open-register + EXACT coverage gates | `test_ebh7_semantic_proof_binding_gates.py` |
| RPL 56/56 without review | 51× PARTIAL Package C | **56× QUALIFIED** normative SSOT |

**B6 readiness:** corrective package submitted; **independent B6 PASS not issued**. See [`ebh_7_p03_independent_replay/B6_EVIDENCE_CONSISTENCY_MATRIX.md`](ebh_7_p03_independent_replay/B6_EVIDENCE_CONSISTENCY_MATRIX.md).

**Not claimed:** B6 PASS, EBH-7 CLOSED, global FRZ PASS.
