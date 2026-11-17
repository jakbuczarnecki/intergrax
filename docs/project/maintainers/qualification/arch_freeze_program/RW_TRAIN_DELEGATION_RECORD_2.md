# RW train — signing delegation record 2 (amended plan digest of T-IGX-043-G)

**Status:** record of Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It supplements `RW_TRAIN_DELEGATION_RECORD.md` (record 1, merged into `arch-freeze/rw-train` as `e2c519b6d8ea4fbfb9b66cf57df766625f08ea36`), which stays unchanged. It grants the signing delegate nothing beyond the ticket and digest named below, and it is not an audit verdict.

## Why a second record

Record 1 states that an amended plan digest needs a new Operator approval and a new record file. The independent review of the PT-WA implementation (PR #642) found that the plan of T-IGX-043-G needed an amendment. The amended plan was reviewed independently and returned REVIEW_PASS in the ARCH-FREEZE session record `REVIEW_AMENDMENT_1.md` (a session artefact under `.tmp/`, not tracked).

## Decision

On 2026-11-17 the Operator answered two questions in a question dialog by choosing, each time, the option marked "(Recommended)" (English rendering):

| Question | Choice |
|---|---|
| Approve the new plan digest of T-IGX-043-G (PT-WA) instead of the earlier one? The amendment adds the LKW `manifest.py` (the web-search qualifier is enabled only when the profile has a search provider), a new `tool_availability.py` and 25 tests. | Approve the new digest. |
| KR-18 in PT-WA: variant P2 (new `intergrax/tools/contracts/tool_availability.py` with a lazy catalog import; the EBH-2A gate cannot see the moved edge; the node is green only while the import stays lazy, pinned by test NT-WA-KR18-3) or variant P1? | P2, with the pinned test. |

## Ticket and digest covered by this record

| Ticket | Node | Approved plan digest (sha256, `plan_digest.py`) | Operator approval |
|---|---|---|---|
| T-IGX-043-G | PT-WA | `5b7d2827185ae3f06562d5e8d76a4cdec067f26b5283929c0e1ebcc4333ac8d1` | 2026-11-17, the choices above |

The delegate signs T-IGX-043-G only for this digest. It must not sign any other digest of this ticket, including `0c0cd5236637c50b7eb5460d09694b1d9d0f29ca19c6dcd729c6e311ed45e714` of record 1.

## What the delegation does not cover

- Any further amendment of the plan of T-IGX-043-G: it needs a new approval and a new record file.
- Any gate-meaning change or any PROD-Q run.
- The merge of the train into `development`, the RB-2 re-pin and every Operator-only file.

## Ancestry and merge rule

The approval commit of T-IGX-043-G must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request merges with a merge commit, not a squash.
