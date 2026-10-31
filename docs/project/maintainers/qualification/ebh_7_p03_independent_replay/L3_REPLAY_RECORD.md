# EBH-7 — L3 replay record

| Field | Value |
| --- | --- |
| EBH7_CLOSURE_IMPLEMENTATION_SHA | `d4937889fc0b07515649dbdddf602df2a4415489` (pre-B6 terminal closure — lineage) |
| EBH7_B6_CORRECTIVE_IMPLEMENTATION_SHA | `8afff322cb2525f0106c2bbccf2d44f682d3e0df` |
| TEST_EXECUTION_SHA | `83136ef7be70ff33d0a81b978c154ac8528930a2` (F07/F08/F09 proof-strength + prior G01–G09) |
| Profile | ECAP L3 + terminal RPL gates + full `tests/qualification/ebh_7/` |
| Result | **187 passed**, **0 skipped**, **0 xfailed** |
| Log | `.tmp/session/ebh-7-b6-f07-f09/l3-final.log` |

**Semantic @ replay tree:** RPL **56/56 QUALIFIED** (0 B6-open); TEN-E2E **26 EXACT / 0 OPEN**; XDM **34 EXACT / 0 OPEN**; G01–G09 **0 OPEN_FINDING**.

Matrix: [`EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md`](EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md) · [`B6_EVIDENCE_CONSISTENCY_MATRIX.md`](B6_EVIDENCE_CONSISTENCY_MATRIX.md).
