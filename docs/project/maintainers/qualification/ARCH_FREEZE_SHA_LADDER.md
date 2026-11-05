# ARCH-FREEZE SHA ladder

| Field | SHA / value | Status @ CLOSURE_DOC_HEAD `950097ea4bcfd5bba828f663bcd993b8bd1e1d63` |
| --- | --- | --- |
| AUDIT_BASELINE_SHA | a0624e28ff77b061e83f3e10aab4c2a9083ee1b4 | **LOCKED** (immutable planning lock) |
| B1_EXECUTION_SHA | 6bdf78c626bfcb347ad3ad9e3e18b65222f1b169 | **LOCKED** (wave-1 B1) |
| B1_INVENTORY_DIGEST | 0ffa2676… (full digest in B1 handoff) | reproducible @ B1 EXEC |
| B2_B4_EXECUTION_SHA | f7ce9daebad35537da974de0dda3962126c6030e | **LOCKED** (wave-1 B2–B4 + L3 FULL 85/85) |
| L3_FULL_G6_DIGEST | 0fe966e4… (G6 L3 bundle) | immutable G6 replay |
| G6_INDEPENDENT_AUDIT_SHA | 3d96691f1b02b393a4de189139a6e1347db99382 | **LOCKED** (wave-1 G6 @ AUDIT HEAD) |
| CLOSURE_DOC_HEAD | 950097ea4bcfd5bba828f663bcd993b8bd1e1d63 | current planning / R0 SSOT (mutable only via authorized R0 batches) |
| IMPLEMENTATION_SHA | TBD at **T-B7-LOCK-FREEZE-SHA** | OPEN |
| FREEZE_CERTIFICATION_SHA | TBD (= IMPLEMENTATION_SHA at admission) | OPEN |
| TEST_EXECUTION_SHA | TBD per certifying run | OPEN |
| EVIDENCE_PUBLICATION_SHA | TBD | OPEN |
| EVOL_B6_ACCEPTANCE_PUBLICATION_SHA | unset | PENDING |
| ENT_B6_ACCEPTANCE_PUBLICATION_SHA | unset | PENDING |
| ACCEPTED_BASELINE_SHA | unset | PENDING (**T-B6-OD-ENT-01**) |
| FORMAL_CLOSURE_PUBLICATION_SHA | unset (distinct per stage: EVOL / ENT / ARCH-FREEZE) | OPEN |
| OD-IC-01 INTERNAL_CERTIFICATION | 5c08e5e6c44d21998c76bd27ad1efab172440f45 | immutable |
| POST_IC_CONSOLIDATION | 81829712acc19ebeadb4773467b9c5642ca975e3 | immutable |
| EVOL TEST_EXECUTION_SHA | 3c4a641b3e48a13f437e830a0562e2a80b03aabb | reusable evidence |
| ENT TEST_EXECUTION_SHA | 89c76eb7b53a0a3ce6f75ab395cf4a2802638a91 | reusable evidence |
| EXECUTION_WAVE1_SHA (legacy label) | 18a0f68fc6c8107e6344fdb4641ecec1945dae5e | superseded by B1/B2–B4 rungs above |

**Rules:** Do not overwrite G6 / B1 / B2–B4 rungs. **FREEZE_CERTIFICATION_SHA** is the single consumer SHA for L3 FREEZE, ECAP §2.C, VBN independent accept, and authorized FRZ promotions.
