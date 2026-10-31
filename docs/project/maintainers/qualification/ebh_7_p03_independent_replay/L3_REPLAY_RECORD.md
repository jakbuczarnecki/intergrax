# EBH-7 — L3 replay record

| Field | Value |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA | `d4937889fc0b07515649dbdddf602df2a4415489` (pre-B6 terminal closure — lineage) |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| TEST_EXECUTION_SHA | `4eb7eb1a89f82215d5356aa7d000198e432777b0` (F07 fail-closed + L3-bound operational gates) |
| Profile | ECAP L3 incl. `test_ebh_5_dual_slug_rpl_e_operational_gates.py` (8 nodes) |
| Result | **195 passed**, **0 skipped**, **0 xfailed** |
| Log | `.tmp/session/ebh-7-final-cert/l3-final.log` |

**Semantic @ replay tree:** RPL **56/56 QUALIFIED**; dual-slug **28/28**; TEN **26 EXACT / 0 OPEN**; XDM **34 EXACT / 0 OPEN**; G01–G09 **0 OPEN_FINDING**.

Matrix: [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) · [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](B6_EVIDENCE_CONSISTENCY_MATRIX.md).
