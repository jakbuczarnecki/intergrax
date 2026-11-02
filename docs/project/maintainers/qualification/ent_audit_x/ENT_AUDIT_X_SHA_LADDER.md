# ENT-AUDIT-X — SHA ladder (SSOT)

**PLAN_BASELINE_SHA (pre-plan HEAD):** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**PLAN_PUBLICATION_SHA (initial master plan):** `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e`  
**Branch:** `development`  
**Authority:** [ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md)

| Role | SHA | Notes |
| --- | --- | --- |
| PLAN_BASELINE_SHA | `8a41d32043b89e555d7c18e052cf7022b8c94681` | Last commit before ENT plan publication wave |
| PLAN_PUBLICATION_SHA | `6328f0cf915ed1f4cee13acb5eb68f07a7438b8e` | First publication: master plan + annexes (8 files) |
| PLAN_RECONCILIATION_SHA | `1656cb8659c0b5c8cac422f861856a59401f7ca1` | P-11/P-12/P-13 documentation corrections only |
| IMPLEMENTATION_SHA | *pending — first commit containing `intergrax/platform/enterprise_audit/`* | Product runner + qual suite |
| TEST_EXECUTION_SHA | *unset* | Clean-checkout pytest for L3 / FREEZE |
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
