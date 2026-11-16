# HIST04 delta addendum: ba07ebaa3 -> 64ce09385 (ENT-AUDIT-X PRs #399-#402)

Read-only. Baseline findings and previously delivered files are unchanged. ENT-AUDIT-X material was read only as a dependency, and I draw no conclusion about ENT-AUDIT-X's own certification. Nothing was executed except `git show`/`git diff`/`git apply --check` on throwaway trees (`D:\Projekty\_wt\_scratch\hist04\n2`, `n3`, outside every worktree); no tests were run.

Delta: 8 commits. #399 docs/ticket/overlay (f1e42feaa, d5d890ab1, T-IGX-024-A), #400 validator fix (0c775b47f), #401 baseline.py IIAC-aware OD-ENT-01 (58c9bee88), #402 closure preflight and OD-ENT-02 verifiers (a90d791fa). The six IIAC `MATERIAL_GOVERNANCE_PATHS` (`iiac_activation_gate.py` L40-47) are NOT in `git diff --name-only ba07ebaa3 64ce09385`, so the stale-audit trigger of the activation gate is not tripped by this delta. I read the list and the diff; I did not run `verify_iiac_policy_activation`.

## 1. PATCH_H4_01 against 64ce09385
Why the old diff fails: its first hunk carries context lines around roadmap L489; #399 rewrote the adjacent L490 (ENT-AUDIT-X row), so the context no longer matches (`patch failed ... :486`).

| Original change | State at 64ce09385 | Action |
|---|---|---|
| L489 body: "Protocol 03 B6 PENDING_AUTHORITY; FRZ-EVOL-01..10 OPEN; FORMAL_CLOSURE_PUBLICATION_SHA unset; not CLOSED" | still present exactly once | REMAINS |
| L489 status cell "[ ] FORMAL PROTOCOL 03 B6 - PENDING_AUTHORITY ..." | still present exactly once | REMAINS |
| L2065 ledger "W7-06 handoff pin and D10 attestation pending" | replaced by #399: "W7-06 formal handoff pin `7a4f279fd...` and D10 digest-chain attestation published (PASS, 2026-11-08)" | SATISFIED by #399, dropped |
| EVOL_X_SHA_LADDER L81 "closure SHAs remain pending" | untouched by #399 (that file is not in the delta; #399 edited ENT_AUDIT_X_SHA_LADDER) | REMAINS |
Note: #399 also rewrote L561 and the checklist intro (L47) to say FRZ-EVOL rows are "promoted per the EVOL-X IIAC closure". After #399 the EVOL-X stage row L489 is now the only canonical place still saying FRZ-EVOL OPEN / PENDING_AUTHORITY, which makes the drift more visible.

New candidate: `PATCH_H4_01_REBASED_64ce093.diff`: 2 files, 3 replacements (L489 body, L489 cell, ladder L81). Verified with `git apply --check` against a tree whose two files were produced by `git show 64ce09385:<path>` (result: OK). Not applied to any worktree, not committed. The old PATCH_H4_01 is superseded for this tip and kept for the record. Wording is unchanged from the original, so the authority caveat stays ("delegated, `process_enforced`").

## 2. Items 6-7 re-assessed for the NEW code only
### 2.1 What it is
- `intergrax/platform/enterprise_audit/baseline.py` (#401): `verify_operator_decision_record` accepts `record_origin: operator-delegate` ONLY for decision `OD-ENT-01` (`_DELEGATE_SIGNATURE_RE` fixes `OD-ENT-\d{2}` and the check requires `OD_ENT_01_DECISION_ID`). Agent origins (`agent|cursor|automation`) never authorize promotion.
- `scripts/qualification/ent_audit_x_formal_closure_preflight.py` Gates A/B (#402): IIAC IN_FORCE gate, manifest (stage, verdict PASS/PASS_SCOPED, policy version, pairwise-distinct implementation/auditor/certifier sessions, digests, limitations, assurance level), SHA-ladder ancestry/ordering, headline exactly once, pin ordering. Predecessor check N-10 consumes the EVOL-X status JSON.
- `verify_od_ent_02_compensating_control.py`, `verify_ent_audit_branch_protection.py`: read-only `gh`-based live/compensating enforcement checks (an external-state check, not an authority-of-signature check).
- Tests added (not run by me): `test_baseline_promotion_iiac.py` (7 test functions incl. expiry-day, rejection parametrization), `test_formal_closure_preflight_gates.py`, `test_od_ent_02_verifiers.py`.

### 2.2 What it enforces mechanically (OD-ENT-01 delegate path)
1. IIAC activation record status IN_FORCE and the record's `policy_version` must equal the activation's (reads the JSON; fails closed if missing).
2. Signature grammar: `Approved OD-ENT-01: operator-delegate @ <ISO with TZ>` plus `Delegation: <path>@<40-hex>`.
3. The delegation is read from a git blob at the cited commit (`git show sha:path`), after `assert_git_ancestor(sha)`: the delegation must be published, not merely edited in the tree. Path must lie under `iiac_evidence/ENT-AUDIT-X/operator_decisions/`.
4. The delegation must contain an operator signature line `Approved delegation <id>: operator @ <ts>` whose id equals the `Delegation ID` field; template/blank fields are rejected (no signature = fail).
5. Field equality: decision id = OD-ENT-01, policy version, record path (repo-relative), `designated_session_id` equality, and `SHA-256 of the text to be signed` = canonical JSON digest of the record minus its two signature lines.
6. Implementer exclusion: designated session not among the delegation's listed implementer sessions; also not equal to the manifest's implementation or auditor session; manifest sessions present and pairwise distinct (the designated session may equal the certifier: R-4).
7. Time window: operator delegation timestamp <= signature timestamp <= `Expires` (date-only means end of that day); timezone required.
8. An accepting independent audit manifest bound to the record pins (`test_execution_sha`, `evidence_publication_sha`, stage ENT-AUDIT-X, digest of the manifest file).
This is substantially Amendment 01 SS4.1 B-1..B-4 (S-1 operator-signed delegation, S-2 grammar, per-decision binding), implemented in application code for one decision. It is NOT T-IGX-021-B as specified: the EVOL-X preflight, the B6 record grammar/format doc, and `verify_iiac_*` role-separation (B-3 for the signing session across the stage) are untouched.

### 2.3 What it still cannot prove
- Authorship: the "operator signature" on the delegation is a text line; any writer can produce a well-formed one and a ancestor commit shows publication, not identity (AC-1/AC-9 unchanged for that line). The grammar and the published-ancestor requirement raise the cost and make the act visible/auditable; they do not authenticate.
- "New session, dedicated, not a continuation, not implementation" (R-2): `designated_session_id` is a self-asserted string compared to another self-asserted string; implementer exclusion works only against sessions someone listed (the delegation's list; the manifest's ids).
- Sessions of other stages' implementers are not cross-checked; no registry of designated sessions; no check that the delegation's author differs from the signer.
- The IN_FORCE check reads the JSON status; it does not run the full activation gate in this path.
- Scope is OD-ENT-01 only; other ENT operator decisions (e.g. OD-ENT-08) and all EVOL-X records keep their previous (process-only) assurance. I found no signature/authorship check in the lines added to the formal-closure preflight (grep for operator/signature/approved found only manifest, session-distinctness and ordering checks).

### 2.4 Adverse-case table, re-run analytically for the new path (not executed)
| Case | Old mechanism (EVOL-X preflight) | New OD-ENT-01 path |
|---|---|---|
| AC-1 agent writes operator line | accepted | delegation's operator line still text-only: accepted if well formed and published; delegate line itself needs the published delegation |
| AC-2 delegate form | rejected | accepted only for OD-ENT-01 with full delegation |
| AC-4 expired delegation | not checked | rejected (window check) |
| AC-5 scope widening / other decision | not checked | rejected (decision id and path bound) |
| AC-6 signer is implementer | not checked | rejected if listed/manifest-equal |
| AC-8 delegation edited after signing | no binding | mitigated: published blob + digest binds the record text |
| AC-10 signer alters the signed text | allowed | text changes after delegation break the digest (changes BEFORE delegation remain possible) |
| AC-9 "another user as the operator" | allowed | still allowed |

### 2.5 Does it change my conclusions?
- **Amendment 01 / F-H4-02:** No change in substance; reinforced. The ENT track independently records the same inconsistency (`ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md` L101: "Amendment 01 status text inconsistent: L3 vs L51/L121/L140; operator's call; recorded in delegation record O-1") and `T-IGX-024-A` L54/L73 excludes editing Amendment 01 and treats the operator's signed delegation (O-1) as non-delegable. That is the S-1 operator-signed delegation reading, and the code now enforces it for OD-ENT-01. So S-1 is de facto applied on the ENT track while the amendment text still leaves it undecided (OQ-1). The operator decision requested in F-H4-02 remains open and is now more concrete: confirm S-1 is kept.
- **Declared vs implemented / F-H4-03:** narrowed for ENT OD-ENT-01 only; remains OPEN for (a) the EVOL-X preflight (plain-form accepted from any writer), (b) the general IIAC verifiers, (c) the Amendment's T-IGX-021-B scope. A divergent, decision-specific grammar (`OD-ENT-\d{2}`) is also a fragmentation risk: the corrective policy version should state one grammar. Whether #401 itself needs an independent code audit is the ENT track's call; I note only that it is product-path code that mechanically gates a baseline promotion.
- **EVOL-X AUTHORITY_GAP / F-H4-01:** UNCHANGED. The new code does not touch EVOL-X records or its preflight. By comparison with the new standard, the EVOL-X per-decision delegation files (no operator signature line, no designated session id, plain operator-form lines) would not meet S-1 as #401 enforces it; this is a comparison of standards, not a statement that the EVOL-X lines are invalid (the grammar is OD-ENT-01-specific and the EVOL-X lines are not delegate-form). Ratification by the operator is still the only closure.
- The activation trust chain (f8c29fb8d) is not invalidated by the delta.

## 3. New or propagated misleading claims in the four PRs (item 3)
No new CLOSED/PASS headline for ENT-AUDIT-X: #399 states plainly "ENT-AUDIT-X is NOT closed ... This overlay promotes nothing" and FRZ-AUD-01..10 OPEN. Findings:
| # | Where | Observation | Severity |
|---|---|---|---|
| D-1 | `ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md` M-06 "EVOL-X predecessor formally closed: CLOSED; FRZ-EVOL-01..10 promoted; D10 attestation PASS (process_enforced)" | Propagates EVOL-X CLOSED without the authority caveat (only the D10 attestation carries `process_enforced`; the Amendment-scope/plain-form issue is absent). Consequence: a downstream reader treats the EVOL-X predecessor as operator-signed. | MEDIUM (propagation of F-H4-01; wording caveat only, no status change proposed) |
| D-2 | Roadmap L489 (EVOL-X row) | Still says PENDING_AUTHORITY / FRZ-EVOL OPEN, now contradicting L561, the checklist intro, ENT ladder rows and the ENT overlay. Fixed by the rebased patch. | MEDIUM (already F-H4-05) |
| D-3 | ENT SHA ladder: `EVOL_IIAC_B6_ACCEPTANCE_SHA ... supersedes the EVOL B6 row above` | Correct, consistent with the EVOL-X ladder. | none |
| D-4 | ECAP "B6 ... human Protocol 03 is historical for EVOL-X and ENT-AUDIT-X" and 'Superseded by IIAC' pointer | Consistent with the activation record and DS-0 (IIAC scope EVOL-X and ENT-AUDIT-X). | none |
| D-5 | ENT overlay "Branch protection none ... not equivalent to live protection" | Honest statement; no claim. | none |
No other CLOSED/PASS claim was found in the changed roadmap/ladder/checklist lines. TOOL-LIFE-X L482 and DIST/PROD-Q rows were not touched by the delta (F-H4-08 etc. unchanged).

## 4. needs_master / blockers
Blockers unchanged (operator ratification of EVOL-X lines; Amendment 01 status/S-1; TOOL-LIFE-X label). Changed in `needs_master` only: integrate `PATCH_H4_01_REBASED_64ce093.diff` instead of PATCH_H4_01; optional wording caveat for ENT matrix M-06 (ENT track's file, proposed text: "(EVOL-X operator signatures delegated, process_enforced, operator ratification pending)"). Note for the operator: the new S-1 enforcement on the ENT track means the operator's O-1 delegation record is a prerequisite for OD-ENT-01 promotion; EVOL-X ratification could use the same operator-signed form.
