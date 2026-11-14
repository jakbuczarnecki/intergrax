# PLAN T-IGX-040-B — PDC-X finding PDC-BL-F01: correct the activation-status wording in the IIAC protocol header (lines 4 and 91)

Parent: PDC-X campaign, node PDC-P7-00 (IIAC ACT-N11 re-activation). Origin: finding `PDC-BL-F01` in `PDC_BASELINE_AND_SCOPE.md`; Operator decision answer 5 of `operator_decisions/DELEGATION-PDC-STANDING.md` ("In the PDC activation (PDC-P7-00)"; decision text 5: the wording is corrected in the same batch as the PDC activation so that one rebind covers both). This plan is written by the implementer lane and is **unsigned**; a separate delegate session signs it. The edit it describes is **not** performed by the pull request that carries this plan (docs: ticket and plan only).

## Problem

The protocol `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` states the activation status twice as `NOT_IN_FORCE`, while `docs/project/maintainers/qualification/IIAC_POLICY_ACTIVATION_RECORD.json` says `IN_FORCE`. Read at `development` tip `622a61b96`:

- Line 4 (header): `**Activation:** [IIAC_POLICY_ACTIVATION_RECORD.json](./IIAC_POLICY_ACTIVATION_RECORD.json) — **NOT_IN_FORCE**. Session B re-audit PASS is preserved evidence. IN_FORCE requires the fail-closed gate (--require-in-force) and a separate operator activation record.`
- Line 91 (section "Policy activation", first paragraph): `Status remains **NOT_IN_FORCE** until a certification session after the ACT-01 gate correction. Session B PASS and the historical PARTIAL record are evidence, not activation.`

A copy of the status in prose can only go stale; the checklist and ECAP already treat the header wording as superseded. Correcting it by changing the copy to `IN_FORCE` would only reintroduce the same drift at the next status change, so the fix removes the copy and names the record as the single source of truth for the status.

## Deliverable (performed later, under this ticket, after the plan is signed)

One edit to one file: the protocol named above, lines 4 and 91 only. Target wording (the implementer of the edit uses these sentences; any other change returns to the signer):

- Line 4: `**Activation:** [IIAC_POLICY_ACTIVATION_RECORD.json](./IIAC_POLICY_ACTIVATION_RECORD.json) is the single source of truth for the activation status of this policy; this document does not restate it. Session B re-audit PASS is preserved evidence. IN_FORCE requires the fail-closed gate (--require-in-force) and a separate operator activation record.`
- Line 91: `The activation status is recorded only in IIAC_POLICY_ACTIVATION_RECORD.json (single source of truth); this document does not restate it. Before IN_FORCE, a certification session after the ACT-01 gate correction is required. Session B PASS and the historical PARTIAL record are evidence, not activation.`

(The link markup and bold markup of the two original lines are kept as in the original lines.) Properties the edit must have:

- No change of any gate meaning: the conditions list that follows line 91 (items 1 to 5 and everything below), the role and separation rules, the ACT rules and every other line of the protocol stay byte-for-byte unchanged. The sentence "IN_FORCE requires the fail-closed gate and a separate operator activation record" and the ordering "certification session after the ACT-01 gate correction before IN_FORCE" keep their meaning; only the status word `NOT_IN_FORCE` and the implied claim of a current status leave the two lines.
- The tokens that the inventory pattern rules rely on stay in the file (`fail-closed`, `IIAC`, `certifier`; `iiac_g0_inventory.py` classifies this path in categories 1 and 2); the edit checks this with `tests/unit/governance/test_iiac_g0_inventory.py`.
- `git diff` of the edit shows exactly two changed lines (plus no whitespace or line-ending change elsewhere).

## Write set

Exactly one path: `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`. No product path (`intergrax/`, `agents/`, `applications/`), no `scripts/`, no `tests/`, no `.github/`, no activation record, no other qualification document. The ticket line `Lifecycle state` and the signer's approval lines in `governance/tickets/T-IGX-040-B.md` are ticket bookkeeping, not part of the write set of the edit. Any need to write elsewhere is not delegated and returns to the Operator.

## Ordering constraint (binding for the PDC-P7-00 sequence)

The protocol is a material governance path: it is in `MATERIAL_GOVERNANCE_PATHS` of `scripts/governance/iiac_activation_gate.py`, and rule ACT-N11 rejects any change of such a path after the `audit_execution_sha` of the activation record. Therefore:

1. The edit of this plan is merged **before** the IIAC `audit_execution_sha` of PDC-P7-00 is fixed, and the protocol, in its edited form, is **inside the scope** of that audit. No further edit of the protocol after that `audit_execution_sha` (a later change makes the activation red again and needs a new audit).
2. `INTEGRATED_BASELINE_SHA` in `PDC_SHA_LADDER.md` (currently `146e7acf1434fffdf74ef6a2e91fffc10fd7b1ec`, stale: product paths changed in PR 548 and PR 549, `development` tip `622a61b96`) is re-fixed **after** the edit is merged, to a tip that contains it (docs-only descendants do not change it). It is not re-fixed by this plan or by the pull request that carries it.
3. Resulting order: (a) this ticket+plan PR (docs only) merges; (b) a separate delegate signs the plan; (c) the edit PR (one file) merges; (d) the baseline row is re-fixed; (e) the independent auditor session audits the delta from the previous `audit_execution_sha` (`2d3eceb6c`) to the baseline, including the protocol; (f) certifier preflight and publication; (g) a fresh delegate session writes the activation line; (h) a separate certifier writes the activation record phases. Steps (e) to (h) are PDC-P7-00 and are not owned by this ticket.
4. Roles: the session that edits the protocol is not the delegate that signs this plan (R-3) and is not the delegate that writes the activation line, and none of them certifies.

## Protocol

1. One `uv run pytest` at most per verification step (several paths in one command); no broad suites; logs under `.tmp/session/`.
2. Verification of the edit: `git diff --stat` shows one file and two changed lines; `uv run pytest tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_activation_gate.py -q` shows no new failure compared with `development` (the known ACT-N11 product-path red of `test_canonical_policy_is_in_force_and_verifies` and `test_n12_...` is expected until PDC-P7-00 and is not a result of the edit; the implementer records which failures existed before the edit); `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` is expected to stay red with ACT-N11 until PDC-P7-00 and is recorded, not fixed.
3. EBH-7 executable-identity gate: the protocol file lies under `docs/project/maintainers/qualification/`, which is on the allowlist of `verify_ebh7_p03_immutable_evidence.py`, so the edit does not break the identity gate. The ticket and plan files under `governance/tickets/` are not on that allowlist (as for every earlier ticket and plan PR of the campaign) and are covered by the final EBH-7 re-pin of the campaign (`DELEGATION-PDC-WAVE2.md` decision 6).
4. Commit with exactly one `Ref: governance/tickets/T-IGX-040-B.md` trailer and no AI attribution; set the upstream to `origin/development` before the first push; push with an explicit refspec; open a PR against `development`; the implementer never merges. Merge authority follows the signature (docs/governance merges are covered by `DELEGATION-PDC-STANDING.md`; the signer states the reference).
5. The implementer never writes `Approved:` or `Delegation:` and never certifies.

## Acceptance

- The protocol states the activation status nowhere in prose; lines 4 and 91 name the activation record as the single source of truth; the string `NOT_IN_FORCE` no longer appears in the two lines (other occurrences of the token, if any exist elsewhere in the file, are outside this plan and are listed in the checkpoint, not changed).
- Exactly one file and two lines changed; no gate meaning changed; no other protocol text touched.
- `test_iiac_g0_inventory.py` green; no new failure in the IIAC governance tests compared with `development`.
- Merged before the PDC-P7-00 `audit_execution_sha`; the audit scope lists the protocol path.

## Test contract

```text
git diff --stat origin/development...HEAD          # one file, two changed lines
uv run pytest tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_activation_gate.py -q
uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force   # expected red (ACT-N11) until PDC-P7-00; recorded
```

## Out of scope

- Any product path, any script, any test file, the activation record `IIAC_POLICY_ACTIVATION_RECORD.json`, `MATERIAL_GOVERNANCE_PATHS`, the IIAC gate code and every other IIAC document.
- Re-fixing `INTEGRATED_BASELINE_SHA`, the EBH-7 re-pin, the IIAC audit, certifier work, the activation line and the activation record (PDC-P7-00), OD signatures, certification.
- Any rewording of the protocol beyond lines 4 and 91, including other mentions of the status or of history in the protocol.
