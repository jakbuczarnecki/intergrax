# T-IGX-022-A — CT-ENT-01: reconcile ENT-AUDIT-X master-plan disposition with its SSOT validator

## Scope

Certification-control corrective (no product change). The plan `docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md` line 3 reads `ENT-AUDIT-X — INTERNALLY CLOSED — AUTOMATED INTERNAL AUDIT ACCEPTED — PUBLISHED` since `477017f15`; `scripts/enterprise_audit/master_plan_ssot_lib.py:433` accepts only three older constants. Result (verified at base `2f6664d53`): `validate_post_implementation_master_plan.py` exits 1 and `test_validator_passes_on_canonical_ssot` / `test_validator_cli_passes` fail, keeping `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast` and the nightly red since 2026-11-05. The disposition constant is the only validator error (probe: with any accepted constant the validator returns no errors).

Change (Option A — extend the accepted set; the published plan text is **not** edited):

- `scripts/enterprise_audit/master_plan_ssot_lib.py`: add `INTERNALLY_CLOSED_DISPOSITION` (byte-identical to plan line 3) and `ACCEPTED_DISPOSITIONS`; use them in the disposition check and update its message; optional (approver may strike): `INTERNALLY CLOSED` disposition incompatible with validation errors, mirroring the existing READY guard.
- `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py`: append anti-divergence guard (plan `**Disposition:**` line must contain exactly one accepted constant) and negative tests (no constant; INTERNALLY CLOSED + other error; INTERNALLY CLOSED does not whitelist un-negated `B6 PASS`).

Follow-on, **not** part of this ticket's file list: re-record `GAP-RC-SSOT-DRIFT` (plan line 163) with the fix SHA and a fresh exit-0 line while keeping the contradiction visible — delivered by the implementer only as a proposed patch; applied by the ENT plan owner after independent audit.

Files (exhaustive): `scripts/enterprise_audit/master_plan_ssot_lib.py`, `tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py`. Material governance paths touched: **none** (IIAC protocol, IIAC verifiers/gate, activation record, impact matrix and trust anchor are not edited; if that becomes necessary → STOP, blocking finding).

Plan: `CT_ENT_PLAN_AND_RISKS.md` §1–§4, §6–§7 (copied byte-unedited to `governance/tickets/plans/CT-ENT-PLAN.md`; the digest is content-based and unaffected by location).

## Plan digest

Plan digest: eb8707e113cfdee3ed87cb4b1406da22e77e41c77e9c7ec1aa66162792b1b5a0

(SHA-256 per `scripts/governance/plan_digest.py` of `CT_ENT_PLAN_AND_RISKS.md` as drafted. Any later edit of the plan makes it stale; recompute before approval. The same plan and digest serve ticket `T-IGX-022-B`.)

## Approved

Approved: operator-delegate @ 2026-11-07T18:20:10+01:00 (Delegation: .tmp/session/cert-campaign/SIGNING/DELEGATION_2026-11-07_MASTER.md@sha256:5264bca38037664e5cfb03cf847305648c5611ea44ca37427406426c7321bdab; decision OD-CTENT01-PLAN-2026-11-07)

Signed record: `.tmp/session/cert-campaign/SIGNING/SIGN-3_CTENT/SIGNED_DECISIONS.md` (OD-CTENT01-PLAN-2026-11-07, SIGNED_WITH_CONDITIONS; option A with the "INTERNALLY CLOSED incompatible with validation errors" guard). Conditions: plan byte-unedited; ticket-id re-check at scaffold (done: no `T-IGX-022*` at origin/development `50ddc04fb` or in local `cert-wave/*`); AUDITOR/CERTIFIER named by MASTER, distinct. The `operator-delegate` form is process_enforced, not a cryptographic signature.

## Test contract

One command at a time (Windows budget; `heavy-test.lock` mutex in the campaign):

```powershell
uv run python scripts/enterprise_audit/validate_post_implementation_master_plan.py          # expect exit 0 (was 1)
uv run pytest tests/qualification/ent_audit_x/unit/test_validate_post_implementation_master_plan.py tests/qualification/ent_audit_x/unit/test_task_id_registry.py tests/qualification/ent_audit_x/unit/test_mechanical_readiness.py -q
uv run python scripts/enterprise_audit/mechanical_readiness_check.py                       # wbs_ssot_validator_pass OK; environmental FAILs in a fresh worktree are expected, not to be fixed here
uv run python scripts/governance/verify_iiac_policy_activation.py                           # exit 0 IN_FORCE (no-regression)
uv run python -m scripts.governance.igx_gov ticket                                          # checkpoint pasted below
```

Acceptance: the two originally red tests pass unmodified; new negative tests fail when the lib edit is reverted (mutation check logged once); PR `ent-audit-fast` run is green or any residual red is explained by a failure other than NF-01.

## Out of scope

Editing the master-plan text, SHA-ladder rows, Matrix C/D, Annex registers, `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json`; applying the `GAP-RC-SSOT-DRIFT` re-record; NF-02/03/04/05/06 (other tickets / operator); any IIAC protocol/verifier/gate/anchor/activation-record/impact-matrix edit; FRZ-* promotion; ENT-AUDIT-X, EVOL-X or B6 closure or PASS; OD-ENT-* decisions; Docker, vendor integrations, production configuration (C10/C15 freeze); any `intergrax/`, `agents/`, `applications/` change; changes to `FORBIDDEN_SELF_CERT` semantics.

## Roles

IMPLEMENTER (one session) ≠ REVIEWER/TESTER ≠ independent AUDITOR (`intergrax-independent-audit`, separate session) ≠ CERTIFIER (also ≠ ENT Session D certifier's implementer role). The implementer must not record PASS or set `ready_for_audit` to PASS.

## Risk / rollback

Low. Widens accepted set by one string; residual: the constant is a claim string, not proof of closure — auditor must state it is **not** formal ENT closure. Rollback: revert the single commit (restores the red state, mutates no register or evidence).

## Lifecycle state

in_progress

## Checkpoint

Implementer verification (not a PASS; hand-over READY_FOR_INDEPENDENT_AUDIT). Logs: `.tmp/session/cert-campaign/IMPL-CTENT-01/logs/`.

- `verify_plan_approval.py` T-IGX-022-A --plan plans/CT-ENT-PLAN.md: OK (digest eb8707e1...b5a0).
- `validate_post_implementation_master_plan.py`: exit 0 (was 1 at base).
- pytest (3 files, one invocation, `-p no:xdist`): 21 passed.
- Mutation check (lib reverted to base): 11 failed, 5 passed (both originally red tests + the new tests fail); lib restored.
- `mechanical_readiness_check.py`: `wbs_ssot_validator_pass` OK; remaining FAILs are the documented environmental ones (not fixed).
- `verify_iiac_policy_activation.py`: IN_FORCE, exit 0; `verify_iiac_g0_inventory_completeness.py`: PASS.
- `igx_gov ticket`: NOT completed. Attempt 1 exit 1 (verify_commit_ticket_refs, no upstream -> full-history walk, cp1250 UnicodeDecodeError on an old commit message; environmental). Attempt 2 (`PYTHONUTF8=1`) killed by the Claude Code memory-pressure guard before start. Pre-commit ruff and diff-scoped gates passed at commit time. To be re-run by the auditor/tester.
