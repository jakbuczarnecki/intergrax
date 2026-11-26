# RED_FIRST V5 — CODE_FIX classification root (six residuals)

**Checkpoint:** `origin/development` `1bc28033cf0fdf95b169ec3b16b3ea5d930c0f09`  
**Inventory:** 39 `CODE_FIX` rows. Aggregate counts stay **33** `RED_AT_PREFIX` / **6** `RED_FIRST_UNREPRODUCIBLE`.  
**Authority used:** MEP-AF2 v2.1 §V5.2–V5.3, `rw_plan.v1.json`, `RED_FIRST_V5_DISPOSITION.md`, `RED_FIRST_V5_SPECIAL_DISPOSITION.md`.

No operator card is **DECIDED** in this document. `scripts/governance/verify_rw_plan_phase_check3.py` is unchanged. No inventory row is removed. `GREEN_AT_PREFIX` is not treated as `RED_AT_PREFIX`.

## Chain

Mechanical prefix probes are exhausted (V5.3 steps 1–5). No product remediation commit is named for these six rows. Further prefix replay has no evidence value.

| FRZ | Contract sentence | Prefix result | Canonical class | Recommended disposition (not DECIDED) |
| --- | --- | --- | --- | --- |
| `FRZ-PLG-02` | external structural implementation works without a core patch | pytest exit 0 @ `22993254b13a`: `register_plugins` accepts an explicit external class | MEP: PLG-02 **N/A WITH EVIDENCE** under SCOPE-03; DS-EXC still open | **OUT** of the red-first obligation. Full admission/composition replay is not justified: the recorded finding is already satisfied at the prefix. |
| `FRZ-PLG-07` | runtime extensions cannot self-expand authority | pytest exit 0 @ `22993254b13a`: `ToolAccessPolicy` refuses tool ids outside the declared contract | No later-wave or N/A mark. Owner `T-IGX-043-J`. | **OUT**. The historical authority path already enforces the bound. No bypass was located. |
| `FRZ-RPL-04` | vendor/backend switch does not modify consumers | pytest exit 0 @ `754ee000675d`: worker call sites bind `MessageConsumer` | MEP **DP-36**; owner node `T-AF2-RW-TYP` (W3) | **OUT** for red-first. Consumer independence already holds. W3 product work stays forbidden while SCOPE is open. |
| `FRZ-REL-07` | no hidden infinite retry/fallback | pytest exit 0 @ `66664e2bffa1`: `RetryEngine.decide` stops at `max_retries` | REL remains OPEN in the MEP table; no separate defect SHA | **OUT**. The bounded decision is already present. REL-09 is not evidence for this row. |
| `FRZ-PRD-08` | production-only bypass count = 0 | inventory gate exit 0 @ `754ee000675d` | MEP **split**: architecture gate plus live **PROD-Q** | **OUT** for the CODE_FIX red-first obligation. The live PROD-Q half stays a later environment gate. |
| `FRZ-FRZ-10` | post-freeze change policy active | `_self_test_cases` `AttributeError` is not that policy | Owner `T-B7-PFP-ACTIVATE` (**W5**); verify `T-B7-PFP-VERIFY` (**W8**) | **Re-home** to W5/W8. Do not implement W5 to satisfy this gate. |

## Contradiction

Phase-check 3 requires every `CODE_FIX` row, including these six, to be `RED_AT_PREFIX` before `RED_FIRST_EVIDENCE=COMPLETE` (V5.2). The same plan says **OUT** is the step after the mechanical path is exhausted (V5.3.6), and it forbids treating `RED_FIRST_UNREPRODUCIBLE` as complete. Applying OUT inside `rw_plan.v1.json` or the validator without an operator citation would change the qualification contract.

Enterprise impact of leaving the six rows unpromoted: the four mechanisms and the PRD-08 inventory gate are already correct on the recorded prefixes and on the current tree. No product defect is being left open by refusing another replay. Closing SCOPE, G2, or ARCH-FREEZE on this record would skip the operator citation V5.3.6 requires.

## Recommended authorization

One operator packet under **T-AF2-RED-CLASSES**, not a validator edit:

1. Mark the red-first obligation of `FRZ-PLG-02`, `FRZ-PLG-07`, `FRZ-RPL-04`, `FRZ-REL-07`, and `FRZ-PRD-08` as **OUT**, with this file as the evidence reference.
2. Re-home `FRZ-FRZ-10` to `T-B7-PFP-ACTIVATE` / `T-B7-PFP-VERIFY` so it is not a pre-SCOPE red-first row.
3. Keep the 39-row inventory until that citation names the exact status change.
4. Independent RW-PLAN-AUDIT stays **NOT_PERFORMED** as a full verdict until the citation exists. A structural pass does not replace it.

Until that citation, `RED_FIRST_EVIDENCE` stays **INCOMPLETE**, **SCOPE** stays **OPEN**, and W3/W5 product edits stay out of scope.

**Recorded:** 2026-11-26. Recommendation only.
