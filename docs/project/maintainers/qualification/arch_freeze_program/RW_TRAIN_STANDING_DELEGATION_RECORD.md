# RW train — standing signing delegation record (OD-AF-23, S1 and S2)

**Status:** record of a standing Operator decision, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the rules below, and it is not an audit verdict. It supplements the per-wave records `RW_TRAIN_DELEGATION_RECORD*.md`, which stay unchanged.

## Decision

On 2026-11-17 the Operator chose the recommended option "accept S1 to S5" in a question dialog about five standing authorizations (ARCH-FREEZE session record `OPERATOR_PACKET_3.md`, addendum OD-AF-23, a session artefact under `.tmp/`, not tracked). This record states rules S1 and S2 for the signing delegate. The other three (S3 recommended answers in advance, S4 merge authority, S5 background-agent implementation) do not concern signing and are not restated here.

## Series named by this record

This record names the ticket series of the RW product train: `T-IGX-043-*`, `T-IGX-044-*`, `T-IGX-046-*` and `T-IGX-051-*`. The wildcards exist so that the strict approval grammar accepts the record for any ticket of these series. They do **not** authorize signing by themselves.

## S1. When the delegate may sign

The delegate writes the approval block of a train ticket only when both conditions hold:

1. **The digest is approved.** The plan digest of the ticket equals a digest that the Operator approved in a wave packet (recorded in the ARCH-FREEZE session record of that packet), or it is an amendment of such a ticket that passes S2.
2. **The review is independent.** The plan, and for an amendment its diff, has an independent review verdict REVIEW_PASS.

The delegate recomputes the plan digest from the plan file, refuses on any mismatch, and reports the signed ticket and digest to the orchestrator.

## S2. Amendment envelope

An amendment of an approved ticket may be signed at its new digest without a new Operator question only when **all** of the following hold:

- the independent review of the amendment is REVIEW_PASS;
- the amendment adds only test or documentation paths inside the module directories the ticket already edits, or one path that an in-scope failing guard test requires;
- it makes no gate-meaning change beyond what the approved ticket states;
- it adds no IIAC material path, no Operator-only file (the roadmap, the FRZ checklist), no path in `scripts/` or `.github/`, and no PROD-Q run;
- the ticket is neither a gate-meaning ticket nor an IIAC-material ticket.

## Exclusions (the Operator decides each case)

- Any ticket that changes the meaning of a gate or edits an IIAC material governance path. The ticket T-IGX-044-F (the PT-E05 node) is such a ticket: an amendment of it always needs the Operator's own decision on its new digest.
- Part G of the PT-WB ticket, any PROD-Q run, the merge of the train into `development`, the RB-2 re-pin and every Operator-only file.
- Any ticket that is not listed in an approved wave packet.

## Ancestry and merge rule

The approval commit of each signed ticket must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request that adds this file merges with a merge commit, not a squash.
