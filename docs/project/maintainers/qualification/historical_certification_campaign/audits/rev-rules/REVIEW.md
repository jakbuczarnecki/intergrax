# REVIEW rev-rules - branch `hist-cert/rules-update` (worktree int-rules), commit d1d7f607a on base 15347255a

Reviewer: rev-rules (read-only, did not author). Scope: `IIAC_AMENDMENT_01_OPERATOR_DESIGNATED_SIGNING_SESSION.md` and `governance/tickets/README.md`.

## Verdict

**d1d7f607a: ACCEPT_WITH_CHANGES.**

The policy intent is faithful to the Operator's instruction and R-1..R-6 are untouched. Three defects need one consolidated correction batch:

- One tooling statement is FALSE (E-1).
- One internal contradiction survives (E-2).
- Two governance files are now stale or in conflict (E-3, E-4).

None blocks the policy itself. They would, however, leave a false "tooling fact" and a validator that contradicts the text in a normative record.

## Checks

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1a | R-1..R-6 and R-2(a)-(d) unchanged | PASS | `git diff 15347255a HEAD -U0 -- <amendment>` filtered for `^[-+]R-[1-6]` and `(a)..(d)` gives no matches (rc=1). Reading both versions side by side shows the same wording. |
| 1b | Struck safeguards are exactly those the Operator rejected | PASS | S-1 (operator-signed delegation record) and S-3 (non-delegable list, "written by the operator personally") are struck. S-2 (disclosure) is kept as a strikeable default. This matches the instruction ("nothing may require me to sign personally", "chat commission suffices"). |
| 1c | Operator quote | PASS, note | Sec. 0 quote labelled as an English rendering by the orchestrator, English text governs. The Polish original of 2026-11-07 is no longer reproduced. This is acceptable under the English-only rule, but the verbatim Polish originals are now only in session records. |
| 1d | Extension beyond the instruction | NOTE | Sec. 0 and 4.2 P-1 make "amendments of this rule" and "IIAC policy activation" delegable. The Operator said "any certification, validation and signature", so this is a reasonable reading, but it is an inference. It should be stated as such (E-6). |
| 2 | Internal consistency | FAIL (2 items) | See E-2 and E-5. No leftover "PROPOSED - NOT IN FORCE", "not delegable" or "personally" sentence remains in the amendment. `Grep` for `PROPOSED - NOT IN FORCE\|not delegable\|S-3` shows only struck or explanatory uses. |
| 3 | Tooling statements | see table below | |
| 4 | Material paths | PASS | `git diff --name-only 15347255a HEAD` lists exactly the two files. Count of the six MATERIAL_GOVERNANCE_PATHS (`scripts/governance/iiac_activation_gate.py:40-47`) touched: 0. `IIAC_POLICY_ACTIVATION_RECORD.json` and `IIAC_OD_ACTIVATION_*`: 0. `intergrax/ agents/ applications/ scripts/ tests/`: 0. No ACT-N11 impact (`iiac_activation_gate.py:569,576`). |
| 5 | README forms vs validators | PASS, with caveats | `verify_plan_approval.py:17,47` and `verify_ticket_lifecycle.py:20,77` use `^Approved:\s*.+$` (MULTILINE). Both forms pass; neither checks the `@ ISO8601` part or the next-line `Delegation:`. README says exactly this. Caveats: E-4 and wording "accepted forms" (E-7). |
| 6 | Honesty of residual risk | PASS, with one overstatement | Sec. 1.2 and 4.4 state `process_enforced`, no proof of operator identity, session novelty or dedication. Overstatement: E-5. |
| 7 | English only | PASS | `Grep [^\x00-\x7F]` finds no non-ASCII text in either changed file. The commit message is English. |

### Tooling statements

| Statement (location) | Result | Evidence |
|----------------------|--------|----------|
| "No existing validator recognises `operator-delegate` (tooling fact)" (amendment Sec. 3, bullet "Validator recognition") | **FALSE** | `scripts/enterprise_audit/iiac_baseline_promotion.py:62,77-78,150-158` accepts `Approved OD-ENT-NN: operator-delegate @ <ts>` plus `Delegation: <path>@<40-hex>`. It was added in commit dda99cf80, which is in base 15347255a. Tests: `tests/qualification/ent_audit_x/unit/test_baseline_promotion_iiac.py:77,201,204`. The statement is true only for the EVOL-X preflight and the IIAC verifiers. |
| That validator imposes no operator-signature requirement | **FALSE (not stated in the amendment)** | `iiac_baseline_promotion.py:217-222` rejects a delegation record that has no `Approved delegation <id>: operator @ <ts>` line. It also rejects blank fields (`:223-226`), requires the record under `iiac_evidence/ENT-AUDIT-X/operator_decisions/` (`:207-210`), and requires `signed_text_digest` and `record_path` to match (`:237-243`). This enforces S-1 (struck) and the "operator signs personally" requirement the Operator rejected. A chat-only commission transcribed per Sec. 1.2 would be refused for OD-ENT-01. |
| `_RECORD_SIGNATURE` accepts exactly `Approved <id>: operator @ <ISO>` (Sec. 4.1 B-1, 5.4) | VERIFIED | `scripts/qualification/evol_x_closure_publication_preflight.py:369` (`^Approved (.+?): operator @ (<ISO>)\s*$`); used at `:883` and `:2281`. |
| B6 allow-list; finding `B-11` (Sec. 4.1 B-1/B-4) | VERIFIED | preflight `:2264-2288` ("B-11 ... only allow-listed lines"). |
| Test `test_w401_ra_06_format_document_matches_the_code` (B-1) | VERIFIED | `tests/unit/qualification/test_evol_x_closure_preflight_b4.py:3548`. |
| Format document says "written by the operator, never by an agent"; validator cannot distinguish authors (Sec. 5.4) | VERIFIED | `docs/project/maintainers/qualification/evol_x/EVOL_X_B6_RECORD_AND_FRZ_ROW_FORMATS.md:14`; the regex has no author information. |
| Protocol "Policy activation" item 3 requires exact line `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` (Sec. 6a) | VERIFIED | `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md:97`; gate regex `iiac_activation_gate.py:34`. |
| Editing any of the six material paths makes `--require-in-force` fail (ACT-N11 stale audit) (Sec. 5.1, 6a) | VERIFIED | `iiac_activation_gate.py:569` ("ACT-N11 stale audit after material governance change"). Also `:576` for product paths. |
| `verify_iiac_policy_activation --require-in-force` "continues to exit 0" (Sec. 5.3) | UNVERIFIABLE here | Not executed (review is read-only, no tests). Rests on the diff touching no material or product path (check 4). |
| `d82f8b722` = 2026-11-07T15:37:50+01:00 and is an ancestor of base | VERIFIED | `git log -1 --format=%cI d82f8b722`; `merge-base --is-ancestor d82f8b722 15347255a` is true. |
| EVOL-X B6 / OD-EVOL-12 / OD-EVOLX-08 lines carry the plain form (Sec. 3, 5.5) | VERIFIED | `iiac_evidence/EVOL-X/b6_acceptance/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md:16`, `.../operator_decisions/EVOL_X_OD_EVOL_12_DECISION_RECORD.md:12`, `EVOL_X_OD_EVOLX_08_DECISION_RECORD.md:12`. |
| Those lines were "written by sessions designated by the operator's chat commissions of 2026-11-07"; chain "started before d82f8b722" (Sec. 3) | UNVERIFIABLE | The plain-form lines carry no author information. The B6 acceptance line is timestamped 15:41:06, which is after d82f8b722 (15:37:50). Whether the chain "started before" is a judgement, not a repository fact (E-6). |

## Required edits (one consolidated batch)

**E-1 (required). Amendment, Sec. 3, bullet "Validator recognition".**
- Old: `No existing validator recognises \`operator-delegate\` (tooling fact, unchanged by this document). A record meant for such a validator follows S-2 (plain form only if the commission names it). Nothing in this amendment changes a validator.`
- New: `The EVOL-X closure preflight and the IIAC verifiers do not recognise \`operator-delegate\` (tooling fact, \`evol_x_closure_publication_preflight.py:369\`). The ENT-AUDIT-X OD-ENT-01 baseline promotion (\`scripts/enterprise_audit/iiac_baseline_promotion.py\`) does recognise it, but it additionally requires a delegation record carrying an operator-written line \`Approved delegation <id>: operator @ <ISO-8601>\` under \`iiac_evidence/ENT-AUDIT-X/operator_decisions/\` (lines 77-78, 207-222). That requirement conflicts with S-1 being struck (Sec. 0, 1.2): until the validator is changed, a chat-only commission is not sufficient for OD-ENT-01 promotion. A record meant for any validator follows S-2 (plain form only if the commission names it). Nothing in this amendment changes a validator.`
- Also add the same limit as item (c) to Sec. 6a and one sentence to Sec. 1.2 "Transcription" (the operator signature line of the template is not required by this rule but is required by the OD-ENT-01 validator).

**E-2 (required). Amendment, R-6 final sentence vs Sec. 6.**
- R-6 ends: `Statement of the operator's will; it takes effect only through R-1..R-5 and the activation preconditions of section 6.` Sec. 6 is now headed "Preconditions for MECHANICAL enforcement (summary; not preconditions of validity)" and says the rule applies from d82f8b722 regardless of steps 1-6. These contradict each other.
- R-6 is the Operator's wording, so do not edit it. Add an immediately following sentence in Sec. 0 "Effect of this revision" (not in R-6): `The closing clause of R-6 ("... and the activation preconditions of section 6") is read as: activation concerns only mechanical recognition; validity of a commission does not depend on it (section 6, last paragraph).` Note it as a reading, not a change of R-6.

**E-3 (required). Stale cross-reference: `docs/project/maintainers/qualification/IIAC_OPERATOR_SIGNING_DELEGATION.template.md`.** Not in the commit, so out of date after it. Edit in the same correction batch:
- Line 3 old: `...is \`PROPOSED - NOT IN FORCE\`. This template creates no authority and must not be filled in or used until that amendment is activated by the operator. Safeguard S-1 (one delegation record per delegation) is agent-proposed and may be struck by the operator.` New: `...is IN FORCE as operator-adopted policy text (revised 2026-11-08). A delegation record is a transcription of a commission given in chat, not a condition of validity (S-1 struck).`
- Line 28 old: `Not delegable (S-3): ... \`[ ] the decision named above is none of these\`` New: delete the line (S-3 struck).
- Lines 31-35 "Operator signature line ... To be written by the operator personally ... No agent writes this line." New: mark the section optional for the rule: `Optional under Amendment 01 (S-1 struck). Required only by validators that read it literally (currently OD-ENT-01, see Amendment Sec. 3).` Keep the line shape, because `iiac_baseline_promotion.py:79-81` parses it.
- Do not change the field labels (`iiac_baseline_promotion.py:82-92` parses them literally).

**E-4 (required). Stale governance file: `governance/agent-control-plane/plan_approvers.yaml`.** It lists approvers `operator` ("Human operator with repo write access") and `maintainer`. README now accepts a third author class (a commissioned session) that this file does not list. The file is not read by any gate (only `inventory.py:19` lists it), so nothing breaks, but the two documents disagree. Add to `approvers:`:
`- id: operator-delegate` / `description: Session commissioned by the operator under IIAC Amendment 01; must carry a Delegation line (process_enforced)` / `may_approve_plans: true`, or state in README that the file is not exhaustive. Alternatively record this as a known gap in the correction batch notes.

**E-5 (required). Overstated disclosure property.** Sec. 0 says S-2 means "a delegated line discloses itself as delegated". Sec. 2 S-2 and Sec. 3 then allow a designated session to write the plain `operator @` form, which discloses nothing in the line itself. Old (Sec. 0): `S-2 is kept as the default and remains strikeable by the operator: a delegated line discloses itself as delegated.` New: `S-2 is kept as the default and remains strikeable by the operator: a delegated line discloses itself as delegated (\`operator-delegate\` plus \`Delegation:\`). Where a validator accepts only the plain form, the disclosure is carried by the delegation record and the commit message, not by the line.`

**E-6 (recommended). State inferences as inferences.**
- Sec. 0 bullet "No decision class is excluded": append `This applies the Operator's "any certification, validation and signature" to the delegation record, IIAC policy activation and amendments of this rule; it is the orchestrator's reading and the Operator may narrow it.`
- Sec. 3 bullet 2: replace `those carry the plain form ... written by sessions designated by the operator's chat commissions of 2026-11-07` with `those carry the plain form; whether they were written by sessions designated by chat commission is not derivable from the records and is a matter for the Operator's confirmation under section 1.2`. Also remove or qualify "started before d82f8b722": the B6 acceptance line is timestamped after that commit.

**E-7 (recommended). README wording.** Item 3 reads "one of two accepted forms". Change to `one of two permitted forms (policy; not enforced by tooling)`, so the word "accepted" is not read as validator behaviour. The last sentence already says the validators test only `^Approved:`; keep it.

**E-8 (recommended). Pronoun.** Sec. 1.2 "Earlier chains": `written on his authority` becomes `written on the operator's authority`. The Operator's pronouns are not stated in any record, and the neutral form is safer in a policy text.

**E-9 (recommended). Delegation reference consistency.** S-2 and README allow `<path>@<sha>` or `path@sha256` for a local session record. B-1 specifies only `<path>@<40-hex sha>`, and `iiac_baseline_promotion.py:78` accepts only `@<40-hex>`. Add to B-1: `(the sha256 form of S-2 is not accepted by the B-1 grammar; a local session record must first be published and referenced by commit)`.

## Residual risks stated in the change (confirmed adequate)

- Identity of the commissioning party and the delegate cannot be proven (Sec. 1.2, 4.4). "Another user acting as the operator" (R-6) is not defined (Sec. 1.2).
- `process_enforced` assurance; no independent audit; no mechanical enforcement.
- Not stated but implied: because plain-form lines written by a delegate are indistinguishable from operator lines, the delegation record and commit message are the only evidence (covered by E-5).

## Could not verify

- Execution of `verify_iiac_policy_activation --require-in-force` (Sec. 5.3). I did not run any gate or test.
- The Operator's actual chat wording of 2026-11-07 and 2026-11-08. I can only check consistency of the rendering with the instruction relayed to me in the task text.
- Whether the EVOL-X lines were written by commissioned sessions (see tooling table).
- The pre-commit gates: not rerun. Nothing in the diff is Python or in a gated product path.
- Content of `T-IGX-021-A` ticket beyond grep lines 17, 24, 31-33, 59. Its text still says the rule "cannot take effect before the follow-up code ... activated" (line 24), which contradicts the amendment's status. The ticket is closed, so I list it as an informational stale statement, not a required edit.

---

# Delta re-review

Scope: NET branch state `git diff 15347255a HEAD` at `bcbacf993` (v2; `d1d7f607a` is v1, which I reviewed above). Five files changed: the amendment, the delegation template, `plan_approvers.yaml`, `TRUSTED_POLICY_PIN.yaml`, `governance/tickets/README.md`. Read-only; no gate or test was run. Line numbers are at HEAD.

## Delta verdict

- **d1d7f607a: ACCEPT_WITH_CHANGES, superseded.** Its required edits E-1..E-5 are applied by bcbacf993. It must not be merged without v2.
- **bcbacf993 (and the net state): ACCEPT_WITH_CHANGES.** All nine edits are applied. v2 introduces one new false tooling statement (D-1, required, one sentence) and one trivial typo (D-2).

## E-1..E-9 status

| Edit | Status | Evidence |
|------|--------|----------|
| E-1 correct "no validator recognises" | APPLIED | Amendment line 69 (Sec. 3 "Validator recognition") now limits the claim to the EVOL-X preflight and IIAC verifiers, names `iiac_baseline_promotion.py:62,77-78` as recognising it, and states the operator-line requirement (`:79-81,207-226`) and the S-1 conflict. Sec. 6a(c) (line 162) and the Sec. 1.2 "Transcription" sentence (line 52) added as asked. Caveat: the new 6a(c) wording contains D-1. |
| E-2 R-6 closing clause | APPLIED | Reading note at line 22 ("Stated reading of R-6 (R-6 itself is not edited)"), placed in Sec. 0, with a pointer to Sec. 6 last paragraph. R-6 is untouched (see below). |
| E-3 delegation template | APPLIED | Template line 3 now states IN FORCE and S-1 struck; the "Not delegable (S-3)" line is removed; the signature section (lines 31-34) is marked optional and names the OD-ENT-01 validator. No `PROPOSED`, `personally`, `Not delegable` or `S-3` text remains (Grep: no matches). |
| E-4 `plan_approvers.yaml` | APPLIED | `governance/agent-control-plane/plan_approvers.yaml` gains `operator-delegate` with `may_approve_plans: true` and the description I proposed. The pin was updated, see the pin section. |
| E-5 disclosure overstatement | APPLIED | Amendment line 19 now says the disclosure is carried by the delegation record and commit message where only the plain form is accepted. |
| E-6 inferences as inferences | APPLIED | Line 18 appends "it is the orchestrator's reading and the operator may narrow it". Line 68 (EVOL-X bullet) drops "started before d82f8b722", says authorship by chat commission is not derivable from the records, and notes the B6 line is timestamped 15:41:06, after d82f8b722. |
| E-7 README "accepted" wording | APPLIED | `governance/tickets/README.md:15` "one of two permitted forms (policy; not enforced by tooling)". |
| E-8 pronoun | APPLIED | Amendment line 55 "on the operator's authority". Grep for `his|he|him` in the amendment and template: no matches. |
| E-9 `Delegation:` reference form | APPLIED | Amendment line 78 (B-1) states the 40-hex form only and that a local session record must first be published; README line 17 matches (`<path>@<40-hex sha>`). Consistent with `iiac_baseline_promotion.py:78`. |

## Newly introduced defects

**D-1 (required). Amendment Sec. 6a(c), line 162: false statement about consequences.** The text says changing the OD-ENT-01 validator is "a product-path change, with the ACT-N11 consequences of section 5.1". That is FALSE:
- `scripts/enterprise_audit/iiac_baseline_promotion.py` is outside `PRODUCT_PATH_PREFIXES = ("intergrax/", "agents/", "applications/")` (`scripts/governance/iiac_common.py:22`) and outside the six `MATERIAL_GOVERNANCE_PATHS` (`iiac_activation_gate.py:40-47`).
- The module's own docstring (`iiac_baseline_promotion.py:4-7`) says it was placed outside `intergrax/` precisely so that ACT-N11 does not fire.
- ACT-N11 (`iiac_activation_gate.py:569,576`) therefore would not trigger on a change to it.

Old: `(a product-path change, with the ACT-N11 consequences of section 5.1)`
New: `(a change under scripts/enterprise_audit/, which is outside the ACT-N11 product-path prefixes and outside the material governance paths (iiac_common.py:22, iiac_activation_gate.py:40-47), so it does not by itself invalidate the IIAC activation; its other consequences for the ENT-AUDIT-X evidence are not assessed here)`

**D-2 (trivial). Amendment Sec. 5.1, line 131: missing comma.** `scripts/governance/iiac_common.py` `verify_iiac_policy_activation.py` should read `scripts/governance/iiac_common.py`, `verify_iiac_policy_activation.py`. The enumeration itself is now correct: it names exactly the six paths of `iiac_activation_gate.py:40-47`. This also corrects v1's loose phrase "the `verify_iiac_*` scripts", because `verify_iiac_g0_inventory_completeness.py` exists and is not a material path.

**Informational (no edit required).** Template line 21 still carries the checkbox `[ ] confirmed by the operator` for the operator's dedication statement. With chat commissions it can be ticked by the transcribing orchestrator. Not wrong, just a mild mismatch with "chat is sufficient"; leave as is unless the Operator wants it removed.

## Tooling facts re-verified at HEAD

| Statement | Result | Evidence |
|-----------|--------|----------|
| EVOL-X preflight does not recognise `operator-delegate` (Sec. 3, `preflight.py:368-369`) | VERIFIED | `evol_x_closure_publication_preflight.py:368` (`_HANDOFF_SIGNATURE`, `operator @`) and `:369` (`_RECORD_SIGNATURE`, `operator @`); no `operator-delegate` in `scripts/` other than `iiac_baseline_promotion.py` (Grep). |
| IIAC verifiers / activation gate do not recognise it | VERIFIED | Same Grep over `scripts/`; gate signature `iiac_activation_gate.py:34` is `operator @` only. `intergrax/` has no match either. |
| OD-ENT-01 validator recognises it (`:62,77-78`) | VERIFIED | `:62` `DELEGATE_RECORD_ORIGIN`; `:77` `_DELEGATE_SIGNATURE_RE`; `:78` `_DELEGATION_LINE_RE` (40-hex only). |
| It requires an operator line `Approved delegation <id>: operator @ <ts>`, record under `iiac_evidence/ENT-AUDIT-X/operator_decisions/` (`:79-81,207-226`) | VERIFIED | `:79-81` regex; `:66` and `:207-210` path rule; `:217-222` rejection; `:223-226` blank-field rejection. |
| Material path list (Sec. 5.1) | VERIFIED | `iiac_activation_gate.py:40-47`. |
| ACT-N11 stale audit / product path (Sec. 5.1, 6a) | VERIFIED | `iiac_activation_gate.py:569,576`; prefixes `iiac_common.py:22`. |
| Protocol "Policy activation" item 3 literal `operator @` line (Sec. 6a) | VERIFIED | Protocol `:97`; gate `:34`. |
| README: validators test only `^Approved:` | VERIFIED | `verify_plan_approval.py:17,47`; `verify_ticket_lifecycle.py:20,77`. Both files are unchanged in the net diff. |
| Sec. 5.3 "expected to continue to exit 0 ... (not executed ...)" | UNVERIFIABLE by me (honestly hedged) | Rests on the net diff touching no material or product path (confirmed below). |
| README "tooling that follows Amendment 01 section 4.1 B-1 accepts only that form" | VERIFIED as a statement about the specification | Amendment line 78 (B-1). No such tooling exists yet. |
| Template "currently the OD-ENT-01 baseline promotion" requires the operator line | VERIFIED | `iiac_baseline_promotion.py:217-222`. |

## Required confirmations

| Check | Result | Evidence |
|-------|--------|----------|
| R-6 untouched (and R-1..R-5) | CONFIRMED | `git diff 15347255a HEAD -U0 -- <amendment>` filtered for `^[-+]R-[1-6]`: 0 lines. R-6 is at line 45 of the HEAD file. |
| None of the six MATERIAL_GOVERNANCE_PATHS changed | CONFIRMED | `git diff --name-only 15347255a HEAD` filtered against the six paths: 0. |
| No activation record changed | CONFIRMED | `IIAC_POLICY_ACTIVATION_RECORD.json`, `IIAC_OD_ACTIVATION_*`, `iiac_evidence/`: 0 changed files. |
| No product path, script or test changed | CONFIRMED | `git diff --name-only 15347255a HEAD -- intergrax agents applications scripts tests`: 0. |
| Template keeps field labels and signature-line shape parsed at `iiac_baseline_promotion.py:79-92` | CONFIRMED | v2 template diff touches only lines 3, 28 (removed) and 31-33; lines 5-7, 12-14, 19, 23, 27 (labels: `Delegation ID:`, `Policy version:`, `Date (ISO-8601):`, `Decision ID (decision to be signed):`, `Record path (repo-relative path of the record to be signed):`, `SHA-256 of the text to be signed:`, `Designated session id:`, `Implementer session id(s) of this stage`, `Expires (ISO-8601):`) and the signature line at `:34` (`Approved delegation ____: operator @ ____`) are byte-identical to base and match `_DELEGATION_LABELS` and `_OPERATOR_DELEGATION_RE`. Field values remain backtick-wrapped as `_delegation_field` requires. |
| README forms vs `verify_plan_approval.py` / `verify_ticket_lifecycle.py` | CONFIRMED | Both forms pass `^Approved:\s*.+$` (MULTILINE); neither the `@ <ISO8601>` part nor the `Delegation:` line is checked, as the README states. |
| English only | CONFIRMED | No non-ASCII text in the amendment or template (Grep); README and YAML diffs are ASCII English. |

## Pin: is pinning `plan_approvers.yaml` appropriate?

- **Hash correctness: VERIFIED.** `git show HEAD:governance/agent-control-plane/plan_approvers.yaml | sha256sum` = `119493ce52955d9b5cdb27ed26029ea77d13577803bac25817511387a5490312`, equal to `TRUSTED_POLICY_PIN.yaml:16`. The base file hashed to the old pin (`8a53eff2...`), so the pin was consistent before. The other six pins are unchanged and their files are unchanged. This matches MASTER's `verify_trusted_policy_pin.py` OK (hashing is of git blob bytes at HEAD per `verify_trusted_policy_pin.py:20-38`).
- **Pinning is not new and not a choice.** The file was already pinned in the base. Without re-pinning in the same commit the advisory check would fail, so updating the one hash is the minimal and correct change. Only that line changed.
- **What the pin does and does not give.** The pin is explicitly advisory (`TRUSTED_POLICY_PIN.yaml:2`, `enforcement: advisory`). Re-pinning in the same commit as the edit means it proves consistency, not independent approval of the change; it cannot detect a malicious edit by whoever commits both. That is the intended design of the file, not a defect of this commit.
- **Substance to flag to the Operator.** The edit widens the set of principals that `plan_approvers.yaml` lists as allowed to approve plans. The file is read by no gate (no match in `intergrax/` or `tests/`; in `scripts/` only `inventory.py:19` lists it), so nothing changes mechanically, and the description honestly says `process_enforced`. The change makes the file agree with the README and the Operator's instruction. I judge it appropriate and low risk; it should be named in the PR description as a governance-policy edit, which the commit message already does.

## Remaining unverified

- Execution of `verify_trusted_policy_pin.py` and of `--require-in-force`: I recomputed the pin hash myself instead of running either.
- The Operator's actual chat wording of 2026-11-07 and 2026-11-08, and whether the EVOL-X plain-form lines were written by commissioned sessions (now correctly stated as not derivable).
- Consequences of changing `iiac_baseline_promotion.py` for ENT-AUDIT-X evidence (qualification-sensitive paths): not assessed; D-1 asks the text to say so.

---

# Final confirmation

Checked `git diff bcbacf993 937c39820` in the int-rules worktree (HEAD = 937c39820).

- **Nothing else changed.** The diff is 1 file (the amendment), +2/-2, at exactly two lines: 131 and 162.
- **D-2 APPLIED as specified.** Line 131 (Sec. 5.1): `iiac_common.py` and `verify_iiac_policy_activation.py` are now separated by a comma. The enumeration still names exactly the six paths of `iiac_activation_gate.py:40-47`.
- **D-1 APPLIED as specified.** Line 162 (Sec. 6a(c)): the phrase "a product-path change, with the ACT-N11 consequences of section 5.1" is replaced by my wording verbatim: a change under `scripts/enterprise_audit/`, outside the ACT-N11 product-path prefixes and the material paths (`iiac_common.py:22`, `iiac_activation_gate.py:40-47`), so it does not by itself invalidate the IIAC activation; other ENT-AUDIT-X consequences not assessed.
- **Net invariants re-checked at 937c39820 (`git diff 15347255a 937c39820`).** Same 5 files as before. Zero of the six MATERIAL_GOVERNANCE_PATHS changed. Zero `R-1..R-6` lines changed. No activation record, product path, script or test changed. `plan_approvers.yaml` and the pin are untouched since bcbacf993, so the verified pin hash still holds.

**FINAL verdict, net branch state at 937c39820: ACCEPT.**

All nine original edits (E-1..E-9) and both delta defects (D-1, D-2) are applied; no required edit remains. Residual limits are those stated in the document itself (`process_enforced`, no proof of delegate identity, not independently audited, OD-ENT-01 validator still demands an operator-signed line) and the unverifiable items listed above, none of which are defects of the change.
