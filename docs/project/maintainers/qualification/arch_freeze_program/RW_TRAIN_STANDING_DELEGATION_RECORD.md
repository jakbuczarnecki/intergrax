# RW train — standing signing delegation record (OD-AF-23, S1 and S2)

**Status:** record of a standing Operator decision, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the rules below, and it is not an audit verdict. It supplements the per-wave records `RW_TRAIN_DELEGATION_RECORD*.md`, which stay unchanged. For amendments inside the S2 envelope, S2 replaces the requirement of those records for a new approval and a new record file; every other limit of those records still applies.

## Decision

On 2026-11-17 the Operator chose the recommended option to accept S1 to S5 (English rendering) in a question dialog about five standing authorizations (ARCH-FREEZE session record `OPERATOR_PACKET_3.md`, addendum OD-AF-23, a session artefact under `.tmp/`, not tracked). This record states rules S1 and S2 for the signing delegate. The other three (S3 recommended answers in advance, S4 merge authority, S5 background-agent implementation) do not concern signing and are not restated here.

## Series named by this record

This record names the ticket series of the RW product train: `T-IGX-043-*`, `T-IGX-044-*`, `T-IGX-046-*` and `T-IGX-051-*`. The wildcards exist so that the strict approval grammar accepts the record for any ticket of these series. They do **not** authorize signing by themselves.

This record applies only to approval commits on `arch-freeze/rw-train` made before the train is merged into `development` (FCS). It authorizes nothing on `development` or on any other branch, and it authorizes nothing new after FCS.

## S1. When the delegate may sign

The delegate writes the approval block of a train ticket only when both conditions hold:

1. **The digest is approved.** The plan digest of the ticket equals the digest that the Operator approved for that ticket in a wave packet (recorded in the ARCH-FREEZE session record of that packet), or it is an amendment of such a ticket that passes S2.
2. **The review is independent.** An independent review verdict REVIEW_PASS exists for the same plan digest, and for an amendment for the same amendment diff.

The delegate recomputes the plan digest from the plan file, refuses on any mismatch, and reports the signed ticket and digest to the orchestrator. The approval commit message names the packet record and its digest row.

## S2. Amendment envelope

An amendment of an approved ticket may be signed at its new digest without a new Operator question only when **all** of the following hold:

- the independent review of the amendment is REVIEW_PASS;
- the amendment adds only test or documentation paths inside the directories the approved plan already edits, or at most one further path that a guard test already named in the approved plan's verification requires because the ticket's own change makes it fail; that path is itself a test or documentation path, or lies inside the directories the approved plan already edits, and it is never a guard baseline, allowlist, ratchet, threshold or exception register, nor a `conftest.py` or pytest configuration change;
- it changes no other part of the approved plan (edit scope, steps, acceptance criteria, thresholds) except the text that lists and verifies the added paths;
- it makes no gate-meaning change beyond what the approved ticket states;
- it adds no IIAC material path, no Operator-only file (including `PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md` and `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md`), no path in `scripts/`, `.github/`, `agent-env/`, `.claude/`, `.cursor/` or `governance/` (other than the ticket's own file), no change to `pyproject.toml` or `uv.lock`, no new or changed governance guard, gate or ratchet test (that counts as a gate-meaning change), and no PROD-Q run;
- the ticket is neither a gate-meaning ticket nor an IIAC-material ticket.

## Exclusions (the Operator decides each case)

- An amendment of any ticket that changes the meaning of a gate or edits an IIAC material governance path. Such a ticket is signed only under S1 at a digest the Operator approved for that ticket in a wave packet; S2 never applies to it. T-IGX-044-F (PT-E05) is such a ticket: every new digest of it needs the Operator's own decision.
- Part G of the PT-WB ticket and A4-040 (the delegate does not sign a PT-WB digest whose plan has Part G as in-scope work), any PROD-Q run, the merge of the train into `development`, the RB-2 re-pin and every Operator-only file.
- Any ticket that is not listed in an approved wave packet.

## Ancestry and merge rule

The approval commit of each signed ticket must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `arch-freeze/rw-train`, or a later train commit that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request that adds this file merges with a merge commit, not a squash.
