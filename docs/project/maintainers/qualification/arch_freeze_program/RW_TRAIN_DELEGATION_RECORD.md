# RW train — signing delegation record (OD-AF-22)

**Status:** record of Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the tickets named in the table below, and it is not an audit verdict.

## Decision (OD-AF-22)

On 2026-11-17 the Operator chose, among the options of a question dialog, "one approval packet per wave plus the signing delegate" for the approval lines of RW train tickets. English rendering of the option: the orchestrator prepares one packet per train wave (ticket, plan file, plan digest); the Operator approves that packet once; a signing delegate session then writes the `Approved:` line of each ticket in the packet. The merge of the train into `development` stays the Operator's decision.

The decision is recorded as OD-AF-22 in the ARCH-FREEZE session record `OPERATOR_PACKET_3.md` (a session artefact under `.tmp/`, not tracked).

## Tickets covered by this record

Each ticket is covered only for the plan digest that the Operator approved in the wave packet. This record names exactly the tickets and digests approved on the date shown. A later wave, or an amended plan digest, gets a new Operator approval and a new record file; this record is never widened in place.

| Ticket | Node | Wave packet | Approved plan digest (sha256, `plan_digest.py`) | Operator approval |
|---|---|---|---|---|
| T-IGX-043-G | PT-WA | packet 0 (lane 0) | `0c0cd5236637c50b7eb5460d09694b1d9d0f29ca19c6dcd729c6e311ed45e714` | 2026-11-17, an explicit choice in a question dialog |
| T-IGX-043-H | PT-WB | packet 1 (lane 1) | `f5f0d6304c8b8f270dd7b74981027c6d829d6112a3788328021ce0d00dc5ccc9` | 2026-11-17, rulings Q-WB-09 (edit the two gate tests inside the ticket), Q-WB-13 (no file ruled a bypass), Q-WB-14 (architecture guards, no change of gate meaning) and A4-040 (Part G out of scope), given with the Operator's statement (English rendering) that they decide now so that the work proceeds automatically |

Conditions of packet 1: Part G of T-IGX-043-H is excluded; PT-WB starts only after PT-WA is merged into `arch-freeze/rw-train`.

## What the delegation does not cover

- Part G of T-IGX-043-H (hosted strict serving, A4-040): the Operator ruled it out of scope on 2026-11-17, so it is `cond` and no approval line covers it.
- Any gate-meaning change or any PROD-Q run: each needs its own Operator decision.
- The merge of the train into `development`, the RB-2 re-pin and every Operator-only file.

## Ancestry rule

The approval commit of each covered ticket must come after the commit that adds this record, so that the `Delegation:` reference (this file at the commit that added it) is a strict ancestor of the approval commit.

## Merge rule

PR #643, which adds this file, merges into the train with a merge commit (not a squash), so that the commit that adds the file stays an ancestor of later approval commits.

## Amended plan digests

If an independent review makes a plan amendment necessary, as it did for T-IGX-043-G after the PT-WA review, the amended digest needs a new Operator approval. It is then recorded in a new record file, and the delegate signs that digest only after that file is in the train.
