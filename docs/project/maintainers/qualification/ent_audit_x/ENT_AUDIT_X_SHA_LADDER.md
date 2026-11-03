# ENT-AUDIT-X — SHA ladder (SSOT)

**PLAN_BASELINE_SHA (pre-plan HEAD):** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**PLAN_PUBLICATION_SHA (initial master plan):** `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e`  
**Branch:** `development`  
**Authority:** [ENT_AUDIT_X_FINAL_IMPLEMENTATION_AND_B6_READINESS_MASTER_PLAN.md](./ENT_AUDIT_X_FINAL_IMPLEMENTATION_AND_B6_READINESS_MASTER_PLAN.md) (execution SSOT); legacy completion plan retained for gap tables.

| Role | SHA | Notes |
| --- | --- | --- |
| PLAN_BASELINE_SHA | `8a41d32043b89e555d7c18e052cf7022b8c94681` | Last commit before ENT plan publication wave |
| PLAN_PUBLICATION_SHA | `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e` | First publication: master plan + annexes (8 files) |
| PLAN_RECONCILIATION_SHA | `1656cb8659c0b5c8cac422f861856a59401f7ca1` | P-11/P-12/P-13 documentation corrections only |
| IMPLEMENTATION_SHA | `86832319b08b3837cfb4ae9c90919389f202cc37` | Tier-0 runner + manifest + CLI + qual L1/L2 |
| FINAL_PLAN_PUBLICATION_SHA | `0cee41a88274a53d14bb3558752d55bdde0d0b01` | W0–W6 B6 readiness wave landed on `development` |
| IMPLEMENTATION_SHA (ENT-AUDIT-X closure baseline) | `1fa5347a56c569fa62ac2c777ff3947d11e906b4` | Audited HEAD @ post-implementation closure master plan |
| IMPLEMENTATION_SHA (R0 trust wave) | `34e03425b2322f415a99a7cfd290fe0982bf6970` | OD verify, dry-run, evidence eligibility, readiness fail-closed, qual tests |
| POST_IMPLEMENTATION_FINAL_CLOSURE_PLAN_PUBLICATION_SHA | `f7243601dbeec93bf26d258c05f545bb1704c7eb` | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) execution SSOT publication |
| IMPLEMENTATION_SHA (R0 one-pass trust wave) | `4f87e5bb20fd8691033a8f9704d205dad18c48c6` | Evidence/VBN/baseline/launchers + qual harness (agent one-pass partial) |
| IMPLEMENTATION_SHA (wave 2 provenance + semantic acceptance) | `e0f64ed7fd57180efc9b93798a3595b9578497af` | VBN/evidence fail-closed, baseline digest drift, 85 semantic wrappers |
| TEST_EXECUTION_SHA | `0cee41a88274a53d14bb3558752d55bdde0d0b01` | 164 passed — log `.tmp/session/ent-audit-x-build/pytest.log` |
| EVIDENCE_CANDIDATE_SHA | *unset* | Pre–independent-audit bundle |
| EVIDENCE_PUBLICATION_SHA | *unset* | Qual artifacts @ descendant of TEST_EXECUTION_SHA |
| B6_ACCEPTANCE_PUBLICATION_SHA | *unset* | Protocol 03 ENT-AUDIT-X PASS |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* | OD-ENT-08 + roadmap §5 |
| ACCEPTED_BASELINE_SHA | *unset* | Operator OD-ENT-01 after first green FREEZE |

## Lineage rules

1. Do **not** rewrite historical SHAs; append new rows for each documentation or evidence publication commit.
2. `merge-base --is-ancestor TEST_EXECUTION_SHA EVIDENCE_PUBLICATION_SHA` required before B6.
3. Tests execute @ `TEST_EXECUTION_SHA`; doc-only commits use manifest split (`qualification_sha` ≠ `audit_execution_sha`).
4. Reconciliation commits (plan fixes) do not substitute for TEST_EXECUTION_SHA.

## EVOL predecessor pins (orchestration only)

| Role | SHA |
| --- | --- |
| EVOL TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVOL EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| EVOL B6_ACCEPTANCE_PUBLICATION_SHA | **PENDING** |
