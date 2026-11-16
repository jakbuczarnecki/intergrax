# HIST04 - IIAC activation, Amendment 01 and delegated-authority reconciliation

Baseline ba07ebaa3. Read-only; analysis from repository text, the gitignored SIGNING dossiers (read only) and source lines. No policy/activation/validator file was edited. Assurance of every item below: `process_enforced` at best.

## 1. IIAC activation (mission item 5, part 1)
- `IIAC_POLICY_ACTIVATION_RECORD.json`: IN_FORCE, policy IIAC-2026-11-05.1, effective_from_sha = policy_activation_sha = f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d (phase 1), Session B verdict PASS, operator line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ 2026-11-05T18:14:38+01:00` in `IIAC_OD_ACTIVATION_IIAC-OD-01-2026-11-05.md` (operator personally; the only activation-class line).
- Scope of IIAC (protocol Purpose): EVOL-X and ENT-AUDIT-X only. OD-ROUTE-DS0 (SIGN-5) re-confirms this and routes QUAL-X/SKILL-X/TOOL-LIFE-X/DIST-SCALE-X through Protocol-03-style independent audit, not IIAC.
- Protocol L4/L91 still read NOT_IN_FORCE (stale; material path; retained, F-H4-07). Migration matrix L3 + roadmap carry the supersession pointer.
- Everything the activation trust chain covers is unchanged by Amendment 01: the amendment commit touched no material path. The existing trust chain is NOT invalidated by anything HIST-04 proposes.

## 2. Amendment 01 - what it is and is not (item 5, part 2)
| Aspect | Text | Verified state |
|---|---|---|
| Status | header: IN FORCE AS OPERATOR-ADOPTED POLICY TEXT; SS5.3 + SS6: PROPOSED - NOT IN FORCE until 6 preconditions | **Self-contradictory.** d82f8b722 changed only the Status and Applies lines. |
| Applies to | certifications started after d82f8b722 (2026-11-07T15:37:50+01:00) | EVOL-X P3/B6/W7 chain started before; QUAL-X/SKILL-X audits (11-07) after |
| EVOL-X | SS3, SS5.4, SS5.5, ticket Constraints/Out of scope: does NOT unblock EVOL-X; OD-EVOL-12/OD-EVOLX-08 still need the operator-written line | Contradicted by the SHA ladder / handoff, which say the EVOL-X signatures were delegated "under Amendment 01" |
| Safeguards S-1..S-3 | agent-proposed, strikeable | never struck nor confirmed (ticket Approved line pending, OQ-1 open) |
| Audit | T-IGX-021-C | waived by operator instruction, not performed |
| Code | T-IGX-021-B | not done; no validator knows `operator-delegate` |
| R-5 expiry | operator sets `Expires` per order | delegation records are agent-authored; expiry values (23:59:59 same day) are the agent's |
| R-6 | "operator, or another user acting as the operator" may commission | identity of "the operator" in a chat is unprovable; no restriction on who "another user" is |
Conclusion: Amendment 01 is a valid statement of operator will for the *non-IIAC* certifications started after 2026-11-07T15:37:50+01:00, as process-enforced policy; it is NOT mechanical, NOT audited, and NOT an authority for EVOL-X.

## 3. Effective authority per decision family (items 8, 16)
| Family | Signed (delegate) | Delegation / bound | Effective authority | Limits stated by the signer |
|---|---|---|---|---|
| SIGN-1 QUAL-X Pop. A + QUAL-X-P0 | 2026-11-07T18:20:51+01:00 | DELEG-CERTCAMPAIGN-SIGN-2026-11-07 (sha 5264bca3.. recomputed OK), expiry 23:59:59 | EFFECTIVE, process_enforced, scope as signed | Population A, d9ea0d6e8 only; FRZ-REG open; P0 not absorbed |
| SIGN-2 SKILL-X | 18:18:48 | same | EFFECTIVE, CONDITIONAL on DS-5 | EC9/EC2 scope, F-04 lifecycle note, no FRZ |
| SIGN-5 DS0-DS5, D1-D3 | 18:24:31-32 | same | EFFECTIVE (route + label decisions); DS5 effective from repo recording (met, see HIST04_EARLY_AUTHORITY_NOTE s3) | D3 = operator-scoped, not audit; D2 = audits, patch 10 not applied |
| SIGN-6 TRACE L395/L404 | 11-07 21:32:22 | same | EFFECTIVE as signed (unpublished); L395 conditional on C-1/C-2 in the publication commit | design lock only |
| SIGN-3/4/7 plan approvals | 11-07 | same | bound to tickets that are NOT in the repo; not verifiable here | SIGN-4 refused OD-TLXCORR-C |
| SIGN-8/9/10 | 2026-11-08 06:36-06:39 | DELEG-2026-11-08 (expiry 11-08 23:59:59) | EFFECTIVE as signed (unpublished) within its Excluded list (no FRZ promotion, no EVOL-X/ENT, no S-3, no WL-7 waiver) | several REFUSED (SEC-10, OD-ARCH-FREEZE-01, R-05, patch 11) - correct fail-closed outcomes |
| EVOL-X P3 / PRE-P3 / B6-ACCEPT / OD-EVOL-12 / OD-EVOLX-08 | plain `operator @` form | per-decision EVOL-X delegation files (no operator signature line, no session id); campaign delegations EXCLUDE EVOL-X | **AUTHORITY_GAP** | needs operator ratification |
For all families: commit authorship gives no independence signal (every commit in the chain is authored by the single git identity jakbuczarnecki; 5fac36949 by "x").

## 4. Declared policy vs implemented validators (item 6)
| Declared | Implemented |
|---|---|
| Operator signs B6/OD records "never by an agent" (B6 formats doc, ticket Constraints) | `_RECORD_SIGNATURE` regex only; no authorship signal possible |
| Delegation: record, session id, digest, expiry, S-3 exclusions (Amendment SS4.1 B-1..B-5) | not implemented (0 hits for `operator-delegate` in scripts/) |
| Designated session dedicated / not implementer (R-2) | process_enforced only; delegate sessions have no registry entry or manifest field |
| IIAC role separation implementer != auditor != certifier | enforced mechanically only for manifest fields and auditor write-surface (`verify_iiac_audit_manifest`, `verify_iiac_role_separation`), not for the operator-signature writer |
| Gate B accepts exactly the plain operator form | a delegate-form line is *rejected* by the B6 allow-list (fail closed) - good for safety, but it forces delegates to use the plain form to pass |

## 5. Can the mechanism permit false acceptance or authority expansion? (item 7) - YES, by design limits
Analytic adverse cases from source (lines cited from `scripts/qualification/evol_x_closure_publication_preflight.py`); NOT executed - execution plan in section 6.
| ID | Case | Expected outcome today | Control that actually stops it |
|---|---|---|---|
| AC-1 | any agent appends `Approved <id>: operator @ <ISO>` | ACCEPTED (regex L369 only) | process / review only |
| AC-2 | delegate form `operator-delegate` in a B6 record | REJECTED (allow-list L2281-2297) | n/a (fail closed) |
| AC-3 | back-/forward-dated timestamp | accepted unless older than the certification publication (W4-F04 `not_before`) | partial |
| AC-4 | expired delegation | not checked (no parse) | process |
| AC-5 | delegate signs an undelisted decision / widens scope | not checked | signer discipline (observed: SIGN-4/9/10 refused) |
| AC-6 | signer is an implementer session | not checked (no signer id) | process |
| AC-7 | fenced / quoted signature line | ignored (parse_operator_record) | good |
| AC-8 | delegation record edited after signing | no in-repo binding; campaign records untracked | line carries sha256 of the delegation file (verifiable only locally) |
| AC-9 | "another user acting as the operator" (R-6) | allowed by text | none |
| AC-10 | delegate alters the operator-prepared text before signing | allowed; observed ("wording tightened" in DS0, DS7, DS8; added conditions in DS3/D3) | signer states what changed; text hash recorded |
Net: false acceptance requires either a mis-operating agent or an unauthorized chat user; nothing in code prevents it. The observed delegates behaved conservatively (at least 5 refusals recorded: OD-TLXCORR-C, R-05, patch 11, SEC-10, OD-ARCH-FREEZE-01). This is a residual assurance limit, not evidence of misuse.

## 6. Test plan (not run; needs heavy lock and Operator budget)
One `uv run pytest` over the existing preflight unit tests plus a throw-away adverse-case module placed under `.tmp/session/hist04/` (not in tests/): call `parse_operator_record`/`operator_record_errors` with AC-1..AC-10 fixtures and assert the table. Expected: AC-1,3(partial),4,5,6,9 accepted; AC-2,7 rejected. Run only on explicit MASTER go-ahead; result would be diagnostic, not exact-SHA evidence.

## 7. Corrective PROPOSAL (item 20) - not an implementation
1. Operator ratification record for EVOL-X delegated lines (section C of HIST04_CANONICAL_PATCH_PROPOSALS.md) - operator personally.
2. Operator decision on Amendment 01 status/S-1..S-3 (proposal text, same file).
3. If mechanical enforcement is wanted: T-IGX-021-B (grammar + delegation validation + role separation), independent audit T-IGX-021-C incl. B-4 delta, new policy version `IIAC-<date>.2`, operator-written activation, two-phase SHA publication. Impact: editing material paths makes `verify_iiac_policy_activation --require-in-force` fail until re-activation and voids OD-3 cond. 3 (B-4 corrective) until its delta audit. The EVOL-X acceptance chain (f8c29fb8d activation) stays valid under the OLD policy; the new version must state non-retroactivity and say how already-delegated lines are treated (grandfathered vs ratified) - this must be the operator's choice.
4. Independent governance audit required: yes, for (3); a documentation-only audit of items 1-2 is optional.
