# ENT-AUDIT-X — SHA ladder (SSOT)

**START_HEAD / PLAN_PUBLICATION_SHA:** `8a41d32043b89e555d7c18e052cf7022b8c94681`  
**Branch:** `development`  
**Authority:** [ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_COMPLETE_IMPLEMENTATION_AND_CLOSURE_MASTER_PLAN.md)

| Role | SHA | Notes |
| --- | --- | --- |
| PLAN_PUBLICATION_SHA | `8a41d32043b89e555d7c18e052cf7022b8c94681` | Master plan + annexes (remediation wave) |
| IMPLEMENTATION_SHA | *unset* | First merged ENT product commit |
| TEST_EXECUTION_SHA | *unset* | Clean-checkout pytest for L3 / FREEZE |
| EVIDENCE_CANDIDATE_SHA | *unset* | Candidate bundle before independent audit |
| EVIDENCE_PUBLICATION_SHA | *unset* | Published qualification artifacts @ descendant |
| B6_ACCEPTANCE_PUBLICATION_SHA | *unset* | Protocol 03 ENT-AUDIT-X independent PASS |
| FORMAL_CLOSURE_PUBLICATION_SHA | *unset* | Operator OD-ENT-08 + roadmap §5 row |
| ACCEPTED_BASELINE_SHA | *unset* | Post-first-green FREEZE audit; operator OD-ENT-01 |

## Lineage rules (aligned with EVOL-X V-0)

1. `merge-base --is-ancestor TEST_EXECUTION_SHA EVIDENCE_PUBLICATION_SHA` required before B6.
2. Audit execution @ `TEST_EXECUTION_SHA` — **not** at documentation-only commits unless doc-only waiver recorded.
3. Publishing evidence creates a new commit (`EVIDENCE_PUBLICATION_SHA`) — re-run **FAST** smoke on publication commit or pin audit to **parent** `TEST_EXECUTION_SHA` in manifest (`qualification_sha` vs `audit_execution_sha` split).
4. Non-ancestor baseline → **BLOCKED** drift run (exit 2).
5. Wrong repo identity / missing `origin` → exit 2.
6. Tamper detection: manifest content hashes (`ARTIFACT_DIGESTS`) — mismatch → exit 3.

## EVOL predecessor pins (orchestration inputs — not ENT closure)

| Role | SHA |
| --- | --- |
| EVOL TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVOL EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| EVOL B6_ACCEPTANCE_PUBLICATION_SHA | **PENDING** |
