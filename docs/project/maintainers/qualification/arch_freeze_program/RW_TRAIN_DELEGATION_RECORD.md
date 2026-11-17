# RW train — signing delegation record (OD-AF-22)

**Status:** record of Operator decisions, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the tickets named in the table below, and it is not an audit verdict.

## Decision (OD-AF-22)

On 2026-11-17 the Operator chose, among the options of a question dialog, "one approval packet per wave plus the signing delegate" for the approval lines of RW train tickets. English rendering of the option: the orchestrator prepares one packet per train wave (ticket, plan file, plan digest); the Operator approves that packet once; a commissioned, dedicated, non-implementing signing delegate then writes the `Approved:` line of each ticket in the packet. The merge of the train into `development` stays the Operator's decision.

The decision is recorded as OD-AF-22 in the ARCH-FREEZE session record `OPERATOR_PACKET_3.md` (a session artefact under `.tmp/`, not tracked).

## Tickets covered by this record

Each ticket is covered only after the Operator has approved the wave packet that lists it. This record names exactly the tickets whose packet was approved on the date shown. A later wave gets a later record file; this record is never widened in place.

| Ticket | Node | Wave packet | Plan digest (sha256, `plan_digest.py`) | Operator approval of the packet |
|---|---|---|---|---|
| T-IGX-043-G | PT-WA | packet 0 (lane 0) | `0c0cd5236637c50b7eb5460d09694b1d9d0f29ca19c6dcd729c6e311ed45e714` approved; amended digest pending (see below) | 2026-11-17 |
| T-IGX-043-H | PT-WB | packet 1 (lane 1) | `f5f0d6304c8b8f270dd7b74981027c6d829d6112a3788328021ce0d00dc5ccc9` | 2026-11-17 |

Packet 0 was approved for the plan digest `0c0cd5236637c50b7eb5460d09694b1d9d0f29ca19c6dcd729c6e311ed45e714`. The independent review of the implementation found that a plan amendment was needed; the amended plan has a different digest. The delegate signs T-IGX-043-G only for a digest that the orchestrator has recorded in this table after an independent review of the amendment, and the approval line then names that digest in the ticket text. Until the table carries that digest, the T-IGX-043-G row is not final.

## What the delegation does not cover

- Part G of T-IGX-043-H (hosted strict serving, A4-040): the Operator ruled it out of scope on 2026-11-17, so it is `cond` and no approval line covers it.
- Any gate-meaning change or any PROD-Q run: each needs its own Operator decision.
- The merge of the train into `development`, the RB-2 re-pin and every Operator-only file.

## Ancestry rule

The approval commit of each covered ticket must come after the commit that adds this record, so that the `Delegation:` reference (this file at the commit that added it) is a strict ancestor of the approval commit.
