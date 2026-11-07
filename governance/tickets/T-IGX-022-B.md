# T-IGX-022-B — CT-ENT-02: make the OD-ENT-02 branch-protection verifier and packet satisfiable and ruleset-aware

## Scope

Certification-control corrective (no product change, no GitHub write). Verified at base `2f6664d53`:

- **NF-02** `scripts/governance/verify_ent_audit_branch_protection.py:15-18` requires contexts `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast|pr-gate`. Actual check-run names (GitHub check-runs API on `dd07d15ff`; workflow jobs without `name:`) are `ent-audit-fast` and `ent-audit-pr-gate`. A correctly configured protection would still fail the verifier; a protection configured per the prefixed names would require checks that never report and block every PR.
- **NF-03** the verifier reads only `branches/development/protection` (404 today). `development` carries active ruleset 20844442 with `deletion` and `non_fast_forward` only; `GET rules/branches/development` returns a top-level **list** the current parser cannot read (**NF-02b**).
- **NF-02c** `OD_ENT_02_OPERATOR_PACKET.md:12-13` and three other ENT docs state the prefixed names as "as shown on PR" — the operator/admin would be told the wrong strings.

Change:

- `scripts/governance/verify_ent_audit_branch_protection.py`: required contexts = bare `ent-audit-fast`, `ent-audit-pr-gate`; evidence = union of classic protection (404 "Branch not protected" tolerated) and rulesets via `rules/branches/{branch}` (list parser); any other endpoint error or malformed payload → fail closed; legacy prefixed strings → explicit "never reports" error; nightly-only guard kept; same CLI flags and exit codes; still read-only (`gh api` GET).
- new `tests/unit/governance/test_verify_ent_audit_branch_protection.py` (inline fixtures, monkeypatched `_gh_api`, no network): positives (classic, ruleset-only, split), negatives (today's real state, single check, legacy strings, nightly-only, classic 403/500, ruleset error, non-JSON/non-list, empty list), workflow-job-name drift guard, `main()` exit codes.
- docs, string correction only: `…/ent_audit_x/OD_ENT_02_OPERATOR_PACKET.md` (rows 12-13 + one clarifying sentence: status-check context = job name; PR UI renders `Workflow / job`; classic or ruleset both read), `…/ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md:17`, `…/ENT_AUDIT_X_OPERATOR_DECISION_AND_EXTERNAL_AUTHORITY_PACKET.md:71`, `…/ENT_AUDIT_X_COMPLETE_FORMAL_CERTIFICATION_AND_CLOSURE_MASTER_PLAN.md:168`.

Files (exhaustive): the verifier, the new test, the four docs above. Material governance paths touched: **none**. Not edited: terminal report, compensating-control package, cross-stage matrix, `T-IGX-015-A.md`, `operator_records/`, IIAC protocol/verifiers/gate/activation record/impact matrix, `.github/workflows/*`.

Plan: `governance/tickets/plans/CT_ENT_02_PLAN_AND_RISKS.md` §1–§3, §5–§7 (byte-identical copy of the signed plan).

## Plan digest

Plan digest: eb8707e113cfdee3ed87cb4b1406da22e77e41c77e9c7ec1aa66162792b1b5a0

(Same plan content and digest as `T-IGX-022-A`; the plan is not edited.)

## Approved

Approved: operator-delegate @ 2026-11-07T18:20:10+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab; decision OD-CTENT02-PLAN-2026-11-07)
Plan digest: eb8707e113cfdee3ed87cb4b1406da22e77e41c77e9c7ec1aa66162792b1b5a0

Signed record: `.tmp/session/cert-campaign/SIGNING/SIGN-3_CTENT/SIGNED_DECISIONS.md` (SIGNED_WITH_CONDITIONS). The `operator-delegate` form is a process-enforced delegated record, not a cryptographic signature.

## Test contract

```powershell
uv run pytest tests/unit/governance/test_verify_ent_audit_branch_protection.py tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py -q
uv run python scripts/governance/verify_od_ent_02_compensating_control.py        # exit 0 unchanged
uv run python scripts/governance/verify_ent_audit_branch_protection.py           # EXPECTED exit 1 today: required checks not configured; message must name both missing checks and both sources
uv run python scripts/governance/verify_iiac_policy_activation.py                # exit 0 IN_FORCE (no-regression)
uv run python -m scripts.governance.igx_gov ticket                               # checkpoint pasted below
```

Acceptance: positives PASS / negatives FAIL as listed; unit tests never touch the network; the live run stays honestly red until the operator/admin acts — **no GitHub write by the implementer** (no `gh api -X PUT/POST`, no ruleset edit). `evaluate`-mode ruleset behaviour of `/rules/branches` is confirmed against the real endpoint and reported; if such rules can be returned, STOP and report instead of improvising.

## Out of scope

Configuring branch protection or rulesets, enabling required checks, choosing live vs compensating mode (OD-ENT-02), `od_ent_02_mode` in the status JSON (NF-06), NF-01 (CT-ENT-01), NF-05 narrative reconciliation (CT-ENT-03), editing any workflow, the compensating-control package, any signed/operator record, IIAC protocol/verifiers/gate/anchor/activation record/impact matrix, FRZ-* promotion, ENT-AUDIT-X/EVOL-X/B6 closure, Docker, vendor integrations, production configuration (C10/C15 freeze).

## Sequencing (outside the ticket)

Operator/admin enables required checks on `development` only after CT-ENT-01 is merged and `ent-audit-fast` is green (NF-04), using the bare names from the corrected packet; then the live verifier turns green. The previously issued OD-ENT-02 packet must be re-issued.

## Roles

IMPLEMENTER ≠ REVIEWER/TESTER ≠ independent AUDITOR (separate session) ≠ CERTIFIER (≠ ENT Session D certifier = implementer of this ticket). Implementer may not record PASS.

## Risk / rollback

Low–medium: LIVE evidence path of OD-ENT-02; main risk is a false PASS (mitigated by union + fail-closed + negatives). Rollback: revert the single commit; no GitHub state changed.

## Lifecycle state

in_progress

## Checkpoint

(empty — implementer pastes `igx_gov ticket` stdout here before push)
