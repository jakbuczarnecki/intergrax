# Signing delegation record for T-IGX-050-L (validator fixes)

**Status:** record of an Operator decision, written by the ARCH-FREEZE orchestrator session. It is an English rendering of a choice the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the ticket and digest named below, and it is not an audit verdict.

## Decision

On 2026-11-17 the Operator chose "Authorize the delegate (Recommended)" in a question dialog about the approval of ticket T-IGX-050-L. The ticket's pull request touches `scripts/` and `tests/`, which the standing rule OD-AF-20 does not cover, so the delegate needs this one-time authorization. The Operator gave the same kind of one-time authorization for T-IGX-050-B on 2026-11-16 (ARCH-FREEZE session record `OPERATOR_PACKET_2.md`, a session artefact under `.tmp/`, not tracked); that authorization is not widened by this record.

## Ticket and digest covered

| Ticket | Subject | Approved plan digest (sha256, `plan_digest.py`) | Operator authorization |
|---|---|---|---|
| T-IGX-050-L | follow-up fixes (findings F1-F4 and F6) of the independent review of the amended ARCH-FREEZE traceability validator | `18a38d7170e9fa33f59061a3400466896549707e8664c945f055245fd02d43c6` | 2026-11-17, one time |

The independent review of the pull request (PR #653, head `ba40168fe7da19b9003082d8add46df1af984684`) returned REVIEW_PASS in the ARCH-FREEZE session record `REVIEW_PR653.md` (a session artefact under `.tmp/`, not tracked). The delegate signs only this digest. Any other digest of this ticket needs a new decision and a new record file.

## What the delegation does not cover

- The merge of the pull request: the Operator merges it by hand, because it changes `scripts/` and `tests/`.
- Any other ticket, any further amendment of this plan, and any gate-meaning change.

## Ancestry and merge rule

The approval commit of T-IGX-050-L must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `development`, or a later commit of `development` that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request that adds this file merges with a merge commit, not a squash.
