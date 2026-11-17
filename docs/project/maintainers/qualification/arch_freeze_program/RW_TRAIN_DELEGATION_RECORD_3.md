# RW train — signing delegation record 3 (wave packet 2, lane 2)

**Status:** record of Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It supplements `RW_TRAIN_DELEGATION_RECORD.md` (record 1) and `RW_TRAIN_DELEGATION_RECORD_2.md` (record 2), which stay unchanged. It grants the signing delegate nothing beyond the tickets and digests named below, and it is not an audit verdict.

## Decision

On 2026-11-17 the Operator approved wave packet 2 of the RW train under OD-AF-22 (one approval packet per wave, then the signing delegate). The packet is the ARCH-FREEZE session record `w3-packet-2/PACKET_2.md` (a session artefact under `.tmp/`, not tracked). The Operator accepted the recommended answers of the Operator-level open questions of lane 2, and then chose the recommended option to approve all five tickets (English rendering) in a question dialog that named the one point of caution: ticket T-IGX-044-F edits 5 of the 6 IIAC material governance paths.

## Tickets and digests covered

| Ticket | Node | Approved plan digest (sha256, `plan_digest.py`) | Operator approval |
|---|---|---|---|
| T-IGX-043-B | PT-FC2 | `14e4de953265462bc6e449c40d862d7e8c501b8118d31e56b9837ec46db3daff` | 2026-11-17, packet 2 |
| T-IGX-043-D | PT-FC4 | `2dfe0ad9f28e5b583c3288ba079a272ad6df940fbc2c2abd1382bc0b3466dd30` | 2026-11-17, packet 2 |
| T-IGX-051-H | T-AF2-DEPS-BATCH | `62053d90e6201d1fcf383e995e773bdd13b08ee1a471c6d3bc8c7b04261e4b7d` | 2026-11-17, packet 2 |
| T-IGX-044-F | PT-E05 | `531ef8e1a60f1e4e1c42011935fcc847ca5e289336970466e7a0351846b5fb0e` | 2026-11-17, packet 2 |
| T-IGX-046-D | PT-D08 | `ddc1b231df1efe6a5c2fe550c4f916a1075d3e423a1eaf9f65b3c44ce67ef424` | 2026-11-17, packet 2 |

The delegate signs each ticket only for its digest above. Earlier digests of these tickets, including approvals recorded on `development` for older text, are void for the amended plans and must not be signed.

## What the delegation does not cover

- Any further amendment of one of these plans: it needs a new approval and a new record file.
- Tickets of other lanes, any PROD-Q run, and Part G of the PT-WB ticket.
- The merge of the train into `development`, the RB-2 re-pin itself and every Operator-only file.

## Ancestry and merge rule

The approval commit of each covered ticket must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request merges with a merge commit, not a squash.
