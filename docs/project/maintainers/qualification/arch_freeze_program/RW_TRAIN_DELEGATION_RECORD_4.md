# RW train — signing delegation record 4 (wave packet 3, lane 3)

**Status:** record of Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It supplements `RW_TRAIN_DELEGATION_RECORD.md`, `RW_TRAIN_DELEGATION_RECORD_2.md` and `RW_TRAIN_DELEGATION_RECORD_3.md`, which stay unchanged. It grants the signing delegate nothing beyond the tickets and digests named below, and it is not an audit verdict.

## Decision

On 2026-11-17 the Operator approved wave packet 3 of the RW train under OD-AF-22 (one approval packet per wave, then the signing delegate). The packet is the ARCH-FREEZE session record `w3-packet-3/PACKET_3.md` (a session artefact under `.tmp/`, not tracked). The Operator chose the recommended option, which accepts the draft defaults of ten Operator-level open questions of lane 3 (Q-FC1-01, Q-FC1-02, Q-FC1-04, Q-E-1, Q-E-2, Q-E-4, Q-WD-11, Q-WD-12, Q-WD-13, Q-RUN-02) and approves the five tickets below. The dialog named the point of caution that two of the tickets change the meaning of gates; for those two, the Operator's approval of the packet is the Operator's own decision for that change.

## Tickets and digests covered

| Ticket | Node | Approved plan digest (sha256, `plan_digest.py`) | Operator approval |
|---|---|---|---|
| T-IGX-043-I | PT-WD | `79b760a7318877254952c24b08d799ae5134338c0f761254a5dc1296cfecabc7` | 2026-11-17, packet 3 (gate-meaning) |
| T-IGX-043-A | PT-FC1 | `f9c0b109c357239d80fc9516acdbd20cbcea9872b17c4b43d356e6669f5e1147` | 2026-11-17, packet 3 |
| T-IGX-043-E | PT-FC2-E | `b641f93fcffd7c7f0743239e1b214e6222784e26b0218f87561c3f03bc861f53` | 2026-11-17, packet 3 (gate-meaning) |
| T-IGX-051-D | T-AF2-RW-RUNTIME | `5297254e206eeb5966f7b88f9538f4cac8ac67c249258612163a91b767ef1ca0` | 2026-11-17, packet 3 |
| T-IGX-051-E | T-AF2-RW-RECOVERY | `40040bf51bd46e10f0e5c89b7d235eda4fb516a2c1796ad17ec9bbd81820dc4c` | 2026-11-17, packet 3 |

The delegate signs each ticket only for its digest above. Earlier digests of these tickets, including approvals recorded on `development` for older text, are void for the amended plans and must not be signed.

## What the delegation does not cover

- Any further amendment of one of these plans: it needs a new approval and a new record file.
- Tickets of other lanes, any PROD-Q run, and Part G of the PT-WB ticket.
- The merge of the train into `development`, the RB-2 re-pin itself and every Operator-only file.

## Ancestry and merge rule

The approval commit of each covered ticket must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request merges with a merge commit, not a squash.
