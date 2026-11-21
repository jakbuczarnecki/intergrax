# EQO-W7 / EQO-W8 — Deferred until FCS and M3

**Status:** `WAIT_FOR_INTEGRATION` (plan §4, §5). EQO does not execute merge train → `development`.

## Preconditions not met

| Gate | Requirement | Current state @ 2026-11-21 |
|------|-------------|----------------------------|
| **FCS** | Merge commit: product tree ≡ G2; parents G2 + development | **Absent** — `TRAIN_PIN` `28c27cb` not on `development`; G2 = post PT-Q02 (not asserted). |
| **R (RB-2)** | `git diff G2 R` documentation only | **Not published** on observed refs. |
| **M3** | `FREEZE_CERTIFICATION_SHA = TEST_EXECUTION_SHA` | **Absent** — EBH-7 lock node not complete in EQO scope. |

## W7 tasks (blocked)

- **W7-01** — Verify FCS definition on actual merge SHA.  
- **W7-02** — Change-impact on 12 development-only commits (addendum 7, EQO, CODEOWNERS) + merge.  
- **W7-03** — Evidence reconciliation (`STILL_VALID` / `REPLAYED` / `INVALIDATED`).  
- **W7-04** — Cite PT-Q02..Q13 / lock nodes or exact blocker.

## W8 Campaign D (blocked)

- **W8-01** — Completeness challenge vs M3 candidate.  
- **W8-02** — CL-01 if blocker.  
- **W8-CLS** — Input to W9 terminal status.

## Exact blockers for program closure

1. #680 / T-IGX-043-A not on train (`EQO-W2-01`).  
2. Severity BLOCKER open: A2-002, A4-003 (`EQO-X-01/02`).  
3. IIAC not `IN_FORCE` on train (`WB01` pytest, `X-03`).  
4. Campaign B DIS/AUD incomplete (WB02–WB12).  
5. EDER mechanical triage not applied (`EQO-E-02` report only).

**EQO terminal:** remains `EQO_NOT_CLOSED_EXACT_BLOCKERS` until W8-CLS + W9-01.