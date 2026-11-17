# Standing signing delegation record for docs and ticket-only ARCH-FREEZE tickets (OD-AF-20)

**Status:** record of a standing Operator decision, written by the ARCH-FREEZE orchestrator session. It is an English rendering of choices the Operator made in the orchestrator chat; it is not a verbatim quote. It grants the signing delegate nothing beyond the scope below, and it is not an audit verdict.

## Decision

On 2026-11-16 the Operator chose the recommended option in a question dialog about a standing authorization for pull requests that change only documentation and ticket files (ARCH-FREEZE session record `OPERATOR_PACKET_2.md`, addendum OD-AF-20, a session artefact under `.tmp/`, not tracked). English rendering: a pull request that changes only documentation and/or ticket files is approved without a separate question once it has an independent review verdict REVIEW_PASS and its CI is green. The approval line is written by a commissioned signing delegate, and the orchestrator merges with a merge commit after checking that the Operator has not already merged it. The same day the Operator confirmed that they merge READY pull requests by hand where the standing rule does not apply (OD-AF-21a).

## Scope of this record

This record names the ticket series that the ARCH-FREEZE program uses for its own documentation, register and ticket work: `T-IGX-050-*` (waves W0 to W2) and `T-IGX-052-*` (waves W4 to W8). A delegate approval that points at this record is valid **only** for a ticket whose pull request changes only paths under `docs/` and `governance/tickets/`, checked with `git diff --name-only` against the merge base. Any pull request that changes a path outside that list is outside this record.

## Exclusions (the Operator decides each case)

- Any pull request that changes a path outside `docs/` and `governance/tickets/`, in particular `scripts/`, `tests/` or `.github/`, a product path under `intergrax/`, `agents/` or `applications/`, a GitHub setting, an IIAC material path, the roadmap or the FRZ checklist. For those, the delegate needs a separate one-time Operator authorization, recorded as a decision addendum and a tracked delegation record of its own; this record is no such authorization, although the series wildcard also names tickets of that kind.
- Tickets of the RW product train (the 051 series and the EQO-A series): they follow OD-AF-22 and their own wave records.

## Ancestry and merge rule

The approval commit of each covered ticket must come after the commit that adds this record. The `Delegation:` reference is the merge commit of the pull request that adds this file into `development`, or a later commit of `development` that contains the file unchanged; it is never an intermediate commit of the pull request branch. The pull request that adds this file merges with a merge commit, not a squash.
