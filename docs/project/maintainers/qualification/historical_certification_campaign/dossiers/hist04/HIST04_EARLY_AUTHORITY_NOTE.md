# HIST04 EARLY AUTHORITY NOTE (interim, time-critical for HIST-01/02/03)

Author: HIST-04 worker (session id UNKNOWN). Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d`. Written 2026-11-08 (UTC ~07:40).
Status: INTERIM. Verified by reading repository files at the baseline; nothing was executed. Not an audit, no PASS claimed. Full detail follows in HIST04_IIAC_AUTHORITY_RECONCILIATION.md.

## 1. What the delegated signatures rest on (verified)
- Amendment 01 (`IIAC_AMENDMENT_01_...md`): header says "IN FORCE AS OPERATOR-ADOPTED POLICY TEXT (2026-11-07) - not audited, not mechanically enforced". It was changed from PROPOSED by commit d82f8b722 (2026-11-07T15:37:50+01:00), which edited ONLY the status and "Applies to policy" lines. The body was NOT reconciled and still says: SS3 "Not in force ... any record carrying `operator-delegate` must be treated as invalid by every existing validator; no agent may write one"; SS5.3 "PROPOSED - NOT IN FORCE"; SS6 activation preconditions 1-6 (code T-IGX-021-B, audit T-IGX-021-C, new policy version, operator activation record) never done (ticket T-IGX-021-A: C waived, B not done; `Approved:` line "pending"; S-1..S-3 neither struck nor confirmed; OQ-1/OQ-3 open).
- "Applies to": certifications STARTED AFTER d82f8b722. SS3/SS5.4/SS5.5 and the ticket (Constraints, Out of scope) say in terms that it does NOT unblock the running EVOL-X P3/B6 chain and that OD-EVOL-12 / OD-EVOLX-08 still require the operator-written line.
- No validator recognises delegation: `grep operator-delegate scripts/` = 0 hits. `evol_x_closure_publication_preflight.py:369` `_RECORD_SIGNATURE = ^Approved (.+?): operator @ <ISO>$` accepts the plain form from ANY writer; it has no authorship, session-id, delegation or expiry check.
- Delegation records are agent-authored restatements of chat commissions. None carries the operator-signed line `Approved delegation <id>: operator @ ...` (template S-1) nor a designated session id. The campaign delegations are gitignored session artifacts (`.tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md`, sha256 `5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab` - I re-computed this hash and it matches; `..._2026-11-08_MASTER.md` exists, not yet hashed by me for any decision).

## 2. EFFECTIVE authority - position per decision family
| Family | Started after d82f8b722? | Form written | Position |
|---|---|---|---|
| SIGN-1 OD-QUALX-ACCEPT, OD-QUALXP0-VERIFIED (11-07 18:20:51) | yes (AUD-QUALX run 11-07) | `operator-delegate` + Delegation line | EFFECTIVE as operator-commissioned, `process_enforced`, within DELEG-2026-11-07 (expiry 23:59:59 not exceeded). Not cryptographic; no validator input; scoped exactly as signed (Population A, d9ea0d6e8). |
| SIGN-2 OD-SKILLX-ACCEPT (18:18:48) | yes | same | EFFECTIVE, CONDITIONAL on DS-5 (see 3). |
| SIGN-5 OD-ROUTE-DS0..DS5, OD-SSOT-D1..D3 (18:24:31-32) | yes | same | EFFECTIVE as route/label decisions. D3 = operator-scoped acceptance only (not audit). D2: audits ordered, patch 10 NOT applied. |
| SIGN-3 / SIGN-4 plan approvals | yes | in ticket plan-approval lines | not yet verified by me (tickets T-IGX-022-A/B, 023-A/B); pending. |
| EVOL-X: OD-EVOLX-P3-B6, OD-EVOLX-PRE-P3, OD-EVOLX-B6-ACCEPT (15:41:06), OD-EVOL-12 (18:17:34), OD-EVOLX-08 (18:21:06) | NO - chain started earlier; Amendment text excludes it | PLAIN `operator @` (indistinguishable from the operator) | **AUTHORITY_GAP**: written by agent sessions, justified only by per-decision "signing delegation" files that (a) have no operator signature line, (b) no designated session id, (c) rest on a chat quote the repo cannot corroborate, (d) conflict with the Amendment's own non-retroactivity/EVOL-X exclusion. The 11-07 MASTER delegation EXPRESSLY EXCLUDES EVOL-X and says the plain form "is NOT commissioned"; the 11-08 delegation also excludes EVOL-X/ENT-AUDIT-X. Neither "invalid" nor "valid" can be concluded from the repo: it needs the OPERATOR PERSONALLY to ratify (see 4). |

## 3. DS-5 and SKILL-X (feeds HIST-01) - verified
- OD-ROUTE-DS5 effective "only from the recording in the repository of OD-QUALX-ACCEPT-2026-11-07". That recording exists at baseline: commit 9d0fc1ca1 (delegated decisions doc) and b0286f769 (roadmap L480/L2057), both ancestors of ba07ebaa3. => DS-5 textual prerequisite is MET at baseline; roadmap L480-481 already carries the conditional wording.
- SKILL-X acceptance therefore STANDS at its recorded scope (baseline 552e379, install/bind/enable only, EC9 catalog-witness, no GOV-X2 replay, no FRZ promotion), and automatically falls if QUAL-X Population A audit is reopened/withdrawn. HIST-01 must not reopen or widen QUAL-X Population A without telling HIST-04/MASTER.
- Limits HIST-01 must NOT overstate: QUAL-X acceptance is bound to d9ea0d6e8 (214 files changed since); FRZ-REG-01..10 stay OPEN; original 2026-10-23 "independently accepted" SKILL-X label stays unvalidated; TOOL-LIFE-X: audit CONDITIONAL/STOP @ 1548640708 (C-1 four red tests, C-2 F3 cancel test), re-audit only after correctives (OD-ROUTE-DS3) - not closed, not waived.
- TOOL-LIFE-X and DIST-SCALE-X independent acceptance are also conditional on DS-5 per its text.

## 4. What the OPERATOR must do (cannot be done by any agent/delegate - Amendment S-3 and DELEG-2026-11-08 exclusions)
1. Personally decide on ratification of the delegated EVOL-X signature lines (OD-EVOLX-P3-B6, PRE-P3, B6-ACCEPT, OD-EVOL-12, OD-EVOLX-08), or instruct re-signing by the operator. Candidate text: HIST04_CANONICAL_PATCH_PROPOSALS.md (to be written; proposal only, unsigned).
2. Personally resolve the Amendment 01 status contradiction (header vs SS3/5.3/6) and S-1..S-3 / OQ-1..4; any change to the protocol/verifiers is a NEW audited policy version (T-IGX-021-B/C + activation) - HIST-04 will only deliver a proposal.
Until then: all downstream claims (HIST-01/02/03, ENT-AUDIT-X intake, ARCH-FREEZE) must cite EVOL-X as "CLOSED - INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED), delegated process_enforced signatures, operator ratification pending", never as cryptographically/operator-written.

## 5. Safe-to-assume for HIST-01/02/03 right now
- FRZ baseline re-verified by me against the worktree: 225 = 120 OPEN + 105 PASS; the 85 historical OPEN counts match the brief exactly (no divergence).
- Nothing in the delegated decisions promotes any FRZ row or closes any stage parent; do not cite them as such.
- `operator-delegate` lines are NOT inputs to any IIAC/EVOL-X validator; do not build gates that assume they are.
