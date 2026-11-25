# CT-ENT-01 / CT-ENT-02 — implementation plan, verification and risks

Role: architect-planner (read-only), worker `ct_ent_plan`, label `CT-ENT-PLAN`.
Base `2f6664d53e3fcd1cb6010e33506e2d22201dc9f3` (= worktree HEAD, branch `cert-wave/ct-ent-plan`). Nothing in this document is implemented, approved or signed.

## 1. Independent verification of the NF-01 / NF-02 claims

| Claim (from ENT-AUDIT-X dossier) | Verdict | My evidence (logs under `logs/`) |
| --- | --- | --- |
| NF-01: `validate_post_implementation_master_plan.py` fails at base | **CONFIRMED** | `uv run python scripts/enterprise_audit/validate_post_implementation_master_plan.py` → exit 1, `disposition must declare READY, PLAN PARTIAL, or TECHNICAL PARTIAL constant` (`logs/nf01_validator.log`) |
| NF-01 cause: plan line 3 rewritten in `477017f15`, validator constants not extended | **CONFIRMED** | `git show 477017f15^:<plan>` line 3 = `` `ENT-AUDIT-X — TECHNICAL PARTIAL / EXACT BLOCKERS` ``; `477017f15` line 3 = `` `ENT-AUDIT-X — INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED` ``. `master_plan_ssot_lib.py` last changed in `8ae58734d`, an ancestor of `477017f15`; no diff to the lib or the CLI since `477017f15`. |
| NF-01: the disposition constant is the **only** validator error (so the fix is sufficient) | **CONFIRMED (new)** | In-memory probe (`logs/nf01_probe.log`): as-is → exactly 1 error; with any accepted constant present → `[]`. No second error is hiding behind the first. |
| NF-01 consequence: `GAP-RC-SSOT-DRIFT` is `CLOSED` on evidence "exit 0 @ HEAD" | **CONFIRMED false at base** | Plan line 163 (`…\| CLOSED \| validate_post_implementation_master_plan.py exit 0 @ HEAD \|…`); validator exits 1 at base. |
| NF-02: verifier requires `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast|pr-gate` | **CONFIRMED** | `scripts/governance/verify_ent_audit_branch_protection.py:15-18` |
| NF-02: real check-run names are bare `ent-audit-fast`, `ent-audit-pr-gate` | **CONFIRMED** | `gh api repos/jakbuczarnecki/intergrax/commits/dd07d15ff…/check-runs` (names only, `logs/nf02_checkrun_names.log`); workflow jobs `ent-audit-fast` / `ent-audit-pr-gate` have no job-level `name:` (`.github/workflows/enterprise-audit.yml:30,70`). The `Workflow / job` string is a UI label, not the status-check context. |
| NF-03: ruleset on `development` has only `deletion` + `non_fast_forward`; verifier never reads rulesets | **CONFIRMED** | `gh api repos/…/rules/branches/development` → `[deletion, non_fast_forward]`, ruleset 20844442 (`logs/nf03_rules_branch.json`). Verifier reads only `branches/{b}/protection`; its exit-1 text is the classic 404 (`logs/nf02_verifier.log`). |
| NF-04: required-check sequencing | **CONFIRMED by construction** | Making `ent-audit-fast` required before CT-ENT-01 lands would block every PR (2 failing tests). |

Two further defects found by this planner (not in the ENT-AUDIT-X dossier), both in scope of CT-ENT-02:

- **NF-02b — verifier cannot parse the ruleset endpoint even if it called it.** `GET /rules/branches/{b}` returns a top-level JSON **list**; `verify_branch_protection` does `payload = json.loads(body)` then `payload.get(...)` (dict API), and `_collect_required_contexts` only handles a dict with a `"rules"` key. A naive "just call the second endpoint" fix would crash or silently collect nothing.
- **NF-02c — the OD-ENT-02 operator packet is the source of the wrong strings.** `OD_ENT_02_OPERATOR_PACKET.md:12-13` (column "GitHub check name (as shown on PR)") and three other docs repeat the prefixed names. An admin following them either cannot find the check or configures a context that never reports, which would **block every PR forever** (required check that no job produces). So the doc fix is as safety-relevant as the verifier fix.

Also noted (hazard for the approval gate, §6): `scripts/governance/verify_plan_approval.py` accepts any non-empty text after a line-start `Approved:` (`^Approved:\s*.+$`). A placeholder such as `Approved: UNSIGNED` would pass the verifier. Therefore every draft uses a heading `## Approved` with prose body and **no** line beginning `Approved:`.

## 2. Remit check: certification-control correctives, not product features

| Test | CT-ENT-01 | CT-ENT-02 |
| --- | --- | --- |
| Touches `intergrax/`, `agents/`, `applications/` | No | No |
| Changes the Tier-0 audit runner/manifest/checks behaviour | No (SSOT *validator* acceptance set only) | No |
| New capability / scenario / SEF-X / CFX | No | No (one extra read-only GitHub endpoint in an existing verifier) |
| Alters a verdict, FRZ status, OD record, signature, IIAC anchor | No | No |
| Widens historical certification scope | No | No |
| Vendor integration / Docker / production config (C10/C15 freeze) | No | **No** — read-only `gh api` calls; **the verifier does not configure protection**. Enabling required checks stays an operator/admin act (OD-ENT-02). |

Conclusion: both are defect corrections of existing certification-control tooling → inside the campaign remit. CT-ENT-02's *live outcome* (green verifier) additionally needs an operator/admin action outside any ticket.

## 3. Material governance path check (hard constraint)

`MATERIAL_GOVERNANCE_PATHS` (`scripts/governance/iiac_activation_gate.py:40`) = `verify_iiac_policy_activation.py`, `iiac_activation_gate.py`, `iiac_common.py`, `verify_iiac_audit_manifest.py`, `verify_iiac_role_separation.py`, and the IIAC protocol `.md`.

| Planned path | Material? | Other pinning / inventory check |
| --- | --- | --- |
| `scripts/enterprise_audit/master_plan_ssot_lib.py` | No | In `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` as a **path-only** inventory entry (no content hash); matrix is pinned to `inventory_source_sha afe9e8ab…` so an edit does not invalidate it. Not in `TRUSTED_POLICY_PIN.yaml` / `policy-registry.yaml` (grep: 0 hits). |
| `scripts/governance/verify_ent_audit_branch_protection.py` | No | Same: path-only matrix entry, category 3, `PRESERVE — no IIAC authority substitution`. Not pinned. |
| `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` | No | path-only matrix entry. |
| new `tests/unit/governance/test_verify_ent_audit_branch_protection.py` | No | New file after the inventory SHA; the completeness check enumerates the *old* ref, so no effect (verify by running `verify_iiac_g0_inventory_completeness.py` once as a no-regression check). |
| 4 ENT docs (§5) | No | None is cited by hash in the activation record (activation record contains no `enterprise_audit`/branch-protection reference). |

**No blocking finding**: neither fix requires editing the IIAC protocol, IIAC verifiers/gate, the activation record, or the trust anchor. Stop condition for the implementer: if during implementation any of those six material paths, `IIAC_POLICY_ACTIVATION_RECORD.json`, or `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` would need to change → STOP and escalate (it would be a new blocking finding, not a silent edit). The impact matrix in particular is not to be regenerated, staged or restored by these tickets.

## 4. CT-ENT-01 — implementation plan

**Decision recommended: Option A — extend the validator's accepted set (leave the published plan text untouched).** Rejected: Option B (rewrite plan line 3 back to a legacy constant) — it would edit a published canonical SSOT to hide a status the operator-recorded closure work established, i.e. rewrite history to suit the tool; and Option B′ (relax to a regex) — hides divergence.

### 4.1 Minimal change

`scripts/enterprise_audit/master_plan_ssot_lib.py`
1. After line 94 add `INTERNALLY_CLOSED_DISPOSITION = "ENT-AUDIT-X — INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED"` (byte-identical to plan line 3, same em-dashes).
2. Add `ACCEPTED_DISPOSITIONS = (READY_DISPOSITION, PARTIAL_DISPOSITION, TECHNICAL_PARTIAL_DISPOSITION, INTERNALLY_CLOSED_DISPOSITION)` and replace the 3-way `not in` test (line 433) with `not any(c in text for c in ACCEPTED_DISPOSITIONS)`; error text → `disposition must declare one accepted constant (READY, PLAN PARTIAL, TECHNICAL PARTIAL, or INTERNALLY CLOSED)`. (No test or doc matches the old message — grep.)
3. Mirror the existing READY guard (line 581): `if INTERNALLY_CLOSED_DISPOSITION in text and errors: errors.append("INTERNALLY CLOSED disposition incompatible with validation errors")`. *(Optional, approver may strike; 2 lines; makes the new "closed" claim fail-closed like READY.)*
4. Do **not** touch `FORBIDDEN_SELF_CERT`, `_line_has_negated_cert`, the SHA ladder rows, Matrix C/D parsing, or `apply_master_plan_ssot_repair.py` / `reconcile_c075_s2.py` (they import only the existing constants; names unchanged).

`tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py` (append only; the two currently-red tests turn green without edits)
- `test_plan_disposition_line_declares_an_accepted_constant` — **the anti-divergence guard**: find the `**Disposition:**` line in the real plan; assert it contains exactly one member of `lib.ACCEPTED_DISPOSITIONS`. Fails the day the plan text and the validator diverge again (the NF-01 failure mode), with a message naming both sides.
- `test_validator_fails_when_no_disposition_constant` (negative): replace the disposition line with `**Disposition:** ENT-AUDIT-X — SOMETHING ELSE`; expect the new error and nothing about other rows.
- `test_validator_accepts_each_accepted_disposition` (parametrised over `ACCEPTED_DISPOSITIONS`, substituting into the mutated plan): the disposition error is absent. (READY substitution may legitimately raise other errors from existing rules; assert only that the *disposition* error is absent.)
- `test_internally_closed_with_other_errors_is_rejected` (negative; only if step 3 is kept): break `C072` edge as the existing test does; expect the "incompatible" error.
- `test_internally_closed_does_not_whitelist_self_cert_phrases` (negative): append a line containing `B6 PASS` (un-negated); expect the FORBIDDEN_SELF_CERT error still fires → proves the new constant does not weaken the guard.

### 4.2 Follow-on (separate owner gate — NOT in the implementer's file list)
`GAP-RC-SSOT-DRIFT` (plan line 163) is CLOSED on evidence that is false at base. After the fix is merged, its evidence cell should be re-recorded with the fix commit SHA and a fresh exit-0 line, **keeping the historical contradiction visible** (e.g. note "red 2026-11-05 `477017f15` … fix `<sha>`"). That is an edit to a canonical register inside the master plan → **proposed patch only**, applied by the authorized ENT plan owner after independent audit (a worker/implementer editing it would be self-recording its own closure). The implementer produces `GAP_RC_SSOT_DRIFT_PROPOSED_PATCH.md` in its dossier; it does not apply it.

### 4.3 Tests (budget: one command at a time, mutex `heavy-test.lock`)
Existing — must be green after: `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py`, `…/unit/test_task_id_registry.py`, `…/unit/test_mechanical_readiness.py`, plus `uv run python scripts/enterprise_audit/validate_post_implementation_master_plan.py` (expect exit 0) and `uv run python scripts/enterprise_audit/mechanical_readiness_check.py` (expect `wbs_ssot_validator_pass` OK; the other FAILs in a fresh worktree are the environmental ones documented in ENTAUDITX_CI_FAILURE_ANALYSIS §5 and must not be "fixed" here).
Single pytest invocation: `uv run pytest tests/qualification/ent_audit_x/unit -q` (the CI step runs unit+integration+l3; the ticket requires the CI result of the PR for the full suite rather than a local full run).
Mutation check (once, by the implementer, log kept): revert the lib edit → the new guard test and the two originally red tests fail; restore.

### 4.4 Risk / rollback
- Risk: low. Validator acceptance set widens by exactly one string that is the published plan's own disposition. Residual semantic risk: "INTERNALLY CLOSED … ACCEPTED" is a *claim string*; the validator checks presence, not truth. Mitigation: it is not a stage/FRZ/B6 certification (plan text itself says OD tasks remain NOT_QUALIFIED; formal closure is separate); step 3 and the self-cert test keep the existing guards. The auditor must state this explicitly so the accepted-constant is not later cited as formal ENT closure.
- Risk: substring check lets the constant appear anywhere in the doc; the new line-scoped guard test closes the practical drift path without changing runtime semantics (deliberately not tightening the lib: out of scope).
- Rollback: `git revert` the single implementation commit; restores the red state exactly as at base; no data, registry or evidence mutated.

## 5. CT-ENT-02 — implementation plan

### 5.1 Minimal change

`scripts/governance/verify_ent_audit_branch_protection.py` (still read-only `gh api`, same CLI flags, exit 0/1)
1. Replace `_REQUIRED_CONTEXTS` by bare check names `("ent-audit-fast", "ent-audit-pr-gate")`; keep a private `_LEGACY_PREFIXED = ("Enterprise audit (ENT-AUDIT-X) / ent-audit-fast", "… / ent-audit-pr-gate")`.
2. Evidence sources, union of:
   - classic `repos/{repo}/branches/{b}/protection` — a 404 whose body says "Branch not protected" is *tolerated* (recorded as "classic: none"); **any other non-zero gh exit or non-JSON body → fail closed**.
   - rulesets `repos/{repo}/rules/branches/{b}` — top-level **list**; collect `rule["type"] == "required_status_checks"` → `parameters.required_status_checks[].context`. gh failure or non-list/non-JSON → fail closed.
   Fix `_collect_required_contexts` accordingly (classic dict path unchanged: `required_status_checks.contexts` and `.checks[].context`).
3. Verdict: PASS only if both bare names are in the union. Errors list which required name is missing and which sources were read (`classic: none|N contexts`, `ruleset: N rules`), so today's state prints "ruleset has deletion, non_fast_forward only; no required_status_checks".
4. If a collected context equals a `_LEGACY_PREFIXED` string → explicit error `context 'X' never reports (check-run name is 'ent-audit-fast'); it would block every PR` (turns the NF-02c misconfiguration into a loud failure rather than silent pass/fail confusion). Keep the nightly-only guard.
5. PASS output prints the bare names and the source(s) that supplied them.

`tests/unit/governance/test_verify_ent_audit_branch_protection.py` (new; inline dict fixtures, `monkeypatch` `_gh_api`, **no network, no `gh`**; collected by `governance-fast`)
- positive: classic with both bare names → PASS; ruleset-only with both → PASS (classic 404); names split across classic and ruleset → PASS.
- negative: **today's real state** (classic 404 + ruleset `[deletion, non_fast_forward]`, recorded verbatim from `logs/nf03_rules_branch.json`) → FAIL naming both missing; only `ent-audit-fast` → FAIL; legacy prefixed contexts only → FAIL with the "never reports" message; nightly-only → FAIL; classic HTTP 403/500 → FAIL even if ruleset is correct; ruleset endpoint error → FAIL; non-JSON classic / non-list ruleset → FAIL; empty list → FAIL.
- drift guard: parse `.github/workflows/enterprise-audit.yml` (PyYAML is a project dependency); assert `ent-audit-fast` and `ent-audit-pr-gate` are job ids and neither job has a `name:` override (so check-run name == id); fails if the workflow is renamed without the verifier.
- `main()` exit codes 0/1 and stdout/stderr headers unchanged.
- no-network guard: assert `subprocess.run` is never reached in unit tests (patch to raise).

Docs (string correction only; each edit ≤ the rows listed; no other prose changed):
- `…/ent_audit_x/OD_ENT_02_OPERATOR_PACKET.md` lines 12-13: check name → bare job names; relabel column "GitHub status-check context = job name (the PR UI renders `Workflow / job`)"; add one sentence that a ruleset or classic rule are both acceptable and the verifier reads both.
- `…/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md:17`; `…/ENT_AUDIT_X_OPERATOR_DECISION_AND_EXTERNAL_AUTHORITY_PACKET.md:71`; `…/ENT_AUDIT_X_COMPLETE_FORMAL_CERTIFICATION_AND_CLOSURE_MASTER_PLAN.md:168`: same string. `test_adv28` only needs "branch protection" to remain in the runbook.
- **Not edited** (historical/other owners): `ENT_AUDIT_X_FORMAL_CERTIFICATION_CAMPAIGN_TERMINAL_REPORT.md`, `OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md`, `ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md`, `T-IGX-015-A.md` (these only name the verifier; no prefixed strings), any `operator_records/`, any dossier. NF-05 narrative drift stays CT-ENT-03/out of scope.

### 5.2 Tests and sequencing
- `uv run pytest tests/unit/governance/test_verify_ent_audit_branch_protection.py tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py -q` (one command) — plus `uv run python scripts/governance/verify_od_ent_02_compensating_control.py` (exit 0 unchanged) and one live run `uv run python scripts/governance/verify_ent_audit_branch_protection.py` whose **expected, honest result is still exit 1** (required checks not configured) with the new, accurate message. The implementer must not report that as a failure of the ticket nor "fix" it by any GitHub write.
- Operator/admin sequencing (outside the ticket, recorded in the approval request): enable required checks only **after** CT-ENT-01 is on `development` (NF-04), using the bare names; then the verifier turns green.

### 5.3 Risk / rollback
- Risk: low–medium (the verifier is the LIVE evidence path of OD-ENT-02). Biggest risk is a **false PASS**: union semantics, fail-closed on any endpoint error, and negative tests for the legacy strings reduce it. Ruleset `enforcement: evaluate|disabled` rulesets are not returned by `/rules/branches` (implementer must confirm with the real endpoint and say so in the report; if `evaluate`-mode rules can appear, the verifier must additionally require `enforcement == active` via the ruleset id lookup — a STOP-and-report item, not an improvisation).
- Risk: operator already holds the old packet → the approval request lists "re-issue corrected packet" as an operator notification item.
- Rollback: revert the single commit; verifier returns to base behaviour (unsatisfiable names). No GitHub state is changed by this ticket, so there is nothing external to roll back.

## 6. Roles, order and the approval gate

| Step | Role (session) | Constraint |
| --- | --- | --- |
| Plan (this document) | architect-planner `ct_ent_plan` | read-only; may not implement or approve |
| Plan approval | **Operator** | Unsigned in all drafts. Needs: `scripts/governance/plan_digest.py` digest recorded as `Plan digest:`; operator writes the `Approved:` line personally; `verify_plan_approval.py <ticket> --plan <plan>` exit 0; lifecycle `plan_pending → plan_approved` (skill `igx-devos-plan-ticket-scope`). Verifier is presence-only — the line is procedural, not cryptographic; no one may fill it on the operator's behalf. |
| Implement CT-ENT-01, CT-ENT-02 | IMPLEMENTER session(s) — may be one session for both (file trees disjoint; `T-IGX-…-A` / `-B` split permitted by tickets README) | only the files in each ticket's list; commits carry `Ref: governance/tickets/<ticket>.md`; `igx_gov ticket` checkpoint pasted into the ticket |
| Review/test | REVIEWER and TESTER roles in sessions ≠ implementer (skills `igx-devos-role-reviewer`, `-tester`) | evidence, not verdict |
| Independent audit | AUDITOR session ≠ implementer ≠ reviewer-of-record (`intergrax-independent-audit`); implementer must not set PASS / `ready_for_audit` = PASS | produces PASS_SCOPED/BLOCKED for the corrective scope only |
| Certification | CERTIFIER ≠ implementer ≠ auditor; ENT Session D certifier must differ from the CT-ENT implementer (per ENT-AUDIT-X Session D package) | this ticket pair does **not** certify ENT-AUDIT-X, B6, any FRZ-*, or close `GAP-RC-SSOT-DRIFT` / `BLK-CI-ENFORCE` |

Ticket IDs `T-IGX-021-A` (CT-ENT-01) and `T-IGX-021-B` (CT-ENT-02) are **provisional**: highest existing is `T-IGX-020-A` on `origin/development` at base; other cert-wave branches exist and may also claim numbers. The scaffolding step must re-check `git ls-tree origin/development governance/tickets/` and all `cert-wave/*` branches and renumber on collision (digests are unaffected by ticket id).

## 7. Consolidated risk register

| ID | Risk | Sev | Mitigation / owner |
| --- | --- | --- | --- |
| R1 | Accepting "INTERNALLY CLOSED" read as formal ENT closure | MAJ (interpretation) | Auditor note; ticket text and test keep FORBIDDEN_SELF_CERT; formal closure remains blocked by EVOL-X/OD-ENT-*; operator |
| R2 | Verifier false PASS | MAJ | union + fail-closed + legacy-string negative + real-state negative test |
| R3 | Verifier crash on ruleset list payload (NF-02b) | MIN | parsed as list; tested |
| R4 | `Approved:` placeholder passes `verify_plan_approval` | MAJ (process) | no `Approved:` line in drafts; operator writes it |
| R5 | Required `ent-audit-fast` enabled before CT-ENT-01 → all PRs blocked | MAJ | sequencing in approval request; operator/admin |
| R6 | Editing docs/scripts under hash-pin or inventory regimes | LOW | checked: none pinned (§3); implementer re-runs `verify_iiac_policy_activation.py` and G0 completeness once as no-regression |
| R7 | `GAP-RC-SSOT-DRIFT` re-record done by implementer = self-closure | MAJ | proposed-patch only; ENT plan owner applies after audit |
| R8 | Ticket-number collision | LOW | re-check at scaffold |
| R9 | Local replay incomplete: I ran no pytest (not needed for planning; mutex not taken) | INFO | implementer/tester run the budgeted single commands |

## 8. What was not verified
No pytest run (by design: mission allows one `uv run` per validator only). Did not execute the proposed changes. Did not exercise `/rules/branches` with an `evaluate`-mode ruleset (no such ruleset exists; flagged for the implementer). Did not read `docs/audit_results/`, `AGENT_INSTRUCTIONS.md`, other workers' dossiers or the main checkout other than the specified ENT-AUDIT-X dossier.
