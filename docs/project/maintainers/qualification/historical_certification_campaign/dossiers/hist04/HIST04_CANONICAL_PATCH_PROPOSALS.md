# HIST04 - canonical documentation patch proposals (CANDIDATES; MASTER integrates)

Baseline ba07ebaa3. Nothing here is applied to the worktree. Nothing is signed. Decision records are NOT manufactured: section C is an unsigned proposal for the operator.

## A. Ready-to-apply unified diff
- `PATCH_H4_01_evolx_ssot_reconciliation.diff` (same folder; `git apply --check` OK at baseline). Three hunks:
  1. roadmap L489 body: replaces "Protocol 03 B6 PENDING_AUTHORITY; FRZ-EVOL-01..10 OPEN; FORMAL_CLOSURE_PUBLICATION_SHA unset; not CLOSED" with the published closure facts (SHAs copied from `EVOL_X_SHA_LADDER.md`; signatures described as delegated, `process_enforced`).
  2. roadmap L489 status cell: `[ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY` -> `[x] CLOSED - INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST SCOPED) - PUBLISHED ...` (identical to §3.0.9 L1565 and L561).
  3. roadmap L2065 ledger: "W7-06 ... pending" -> published (7a4f279fd); EVOL_X_SHA_LADDER L81: "remain pending" -> historical + published.
  Reviewer must be a session other than the author. No FRZ status touched. Does not alter the ENT-AUDIT-X handoff.

## B. Proposals needing an operator/MASTER choice (text only)
### B1. TOOL-LIFE-X label (roadmap L482) - F-H4-08
Current: `**[x] CLOSED - INDEPENDENTLY ACCEPTED** (acceptance baseline @ 1548640708...; publication **READY FOR INDEPENDENT PUBLICATION AUDIT** on GitHub)`.
Option A (annotation, status unchanged, recommended minimum):
append `; **independent exact-SHA audit AUD-TOOLLIFE returned CONDITIONAL / chain STOP @ 1548640708 (C-1 four red tests, C-2 F3 cancel test); re-audit authorised after correctives by OD-ROUTE-DS3-2026-11-07; no external review artefact exists in the repository; this label is NOT supported by an independent PASS and must not be cited as current-HEAD proof**`.
Option B (status): `[ ] CONDITIONAL - AUDIT OPEN` (operator decision; DS-3 expressly left this undecided). Neither option may be signed by a delegate (it changes a stage status; DELEG-2026-11-08 excludes closing/reopening parents without audit evidence).
### B2. SKILL-X headline (L481)
Headline `[x] CLOSED - INDEPENDENTLY ACCEPTED` already followed by "CONDITIONAL on the QUAL-X predecessor regularisation (DS-5)". Proposed: move "CONDITIONAL" into the bold headline (`**[x] CLOSED - INDEPENDENTLY ACCEPTED (CONDITIONAL, delegated)**`) so a status-column scan cannot miss it. Cosmetic, SSOT_DRIFT-prevention.
### B3. DIST-SCALE-X label (L483)
Keep `CLOSED - OPERATOR SCOPED ACCEPTANCE`. The row text says "WL-7 14400 s deliberately deferred non-blocking" (verified) but my grep did not find the words "NOT PROVEN" or "independent audit not performed" in the row; propose appending: `14400 s NOT PROVEN (3600 s substitute only); no independent audit performed; DS4-authorised audit pending`. Re-read the full row before patching.
### B4. Publication of already-signed unpublished records (MASTER lane, verbatim copy method of 9d0fc1ca1)
SIGN-6 (TRACE L395 with C-1 link fixes + C-2 row text; L404), SIGN-7, SIGN-8, SIGN-9 (DS-7, DS-8, D-5 16 rows, patch 08 4 rows, D-6, R-03, R-06, R-01/R-02), SIGN-10 (patch 13, F-04, WBS B7 rows, Path B). Patch 10 must stay unapplied. Publish byte-preserving copies of DELEGATION_2026-11-07/08_MASTER.md (F-H4-04) under a tracked evidence path. Each publication commit reviewed by a non-author session; ticket ids that collide on origin (022-A, 022-B, 023-A) need fresh ids and fresh plan approvals.
### B5. Roadmap L2034 (ledger row "PROD-Q-R1-PACKAGE-P - OPERATOR B6 SCOPED ACCEPTANCE", clause "PROD-Q-P0 B6 PENDING") - low priority
Prefix `Historical (pre-2026-10-21):` to the clause; no status change. Re-read the row first: I only saw the clause out of context.
### B6. NOT proposed (explicit)
- No edit to `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` L4/L91 (material path; F-H4-07).
- No edit to `IIAC_POLICY_ACTIVATION_RECORD.json`, validators, tests, `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` (uncommitted operator change must stay untouched).
- No FRZ status change; no stage closure; no historical finding rewritten.

## C. Operator-only proposals (UNSIGNED; "Do not pre-fill"; no agent or delegate may write the signature)
### C1. Ratification of delegated EVOL-X signature lines (candidate text for a record to be created by the operator)
```text
EVOL-X delegated signature ratification (candidate)
Decision ID: OD-EVOLX-RATIFY-<date chosen by operator>
Stage: EVOL-X
Policy version: IIAC-2026-11-05.1
Ratifies, as operator decisions with effect from their original timestamps, the plain-form lines written by operator-designated signing sessions: OD-EVOLX-P3-B6-2026-11-07 (15:41:06+01:00), OD-EVOLX-PRE-P3-2026-11-07 (15:41:06+01:00), OD-EVOLX-B6-ACCEPT-2026-11-07 (15:41:06+01:00), OD-EVOL-12 (18:17:34+01:00), OD-EVOLX-08 (18:21:06+01:00); text digests as in the per-decision delegation files (recomputed OK by HIST-04 for OD-EVOL-12 and OD-EVOLX-08).
Alternative the operator may choose instead: re-sign these lines personally; or decline ratification (then EVOL-X formal closure and FRZ-EVOL promotion must be re-assessed - HIST-04 makes no recommendation to decline).
Approved <Decision ID>: operator @ ________________   (to be written by the operator personally)
```
### C2. Amendment 01 reconciliation (candidate directions for the operator)
(a) Declare, in an operator-written note, that Amendment 01 is in force as process-enforced policy for certifications started after d82f8b722, that its SS3/SS5.3/SS6 "not in force" language is superseded by the header, that S-1 is [kept|struck] (delegation records are then [to be operator-signed|chat-commissioned]), S-2 [kept|struck], S-3 kept, and that EVOL-X is [covered by C1 ratification|excluded]; or (b) schedule T-IGX-021-B/C and a new audited policy version. Body edits to the amendment are an "amendment of this rule" (S-3) and are therefore operator-written.

## D. Independent audit / activation work required (item 20)
| Work | Needed? | Why |
|---|---|---|
| Doc review of PATCH_H4_01 | yes (non-author session) | status text on canonical roadmap |
| Independent audit of Amendment 01 + validators (T-IGX-021-C) | only if mechanical enforcement is chosen | material governance path |
| New policy activation | only for (b) | material path edits void activation |
| TOOL-LIFE-X re-audit | yes (HIST-01) | label F-H4-08 |
| DIST-SCALE-X independent audit | yes (MASTER auditor, DS4) | G-06 gap |
| TRACE L395/L404 | publication only | audits done |
