# T-IGX-040-B — PDC-X finding PDC-BL-F01: correct the `NOT_IN_FORCE` wording in the IIAC protocol header (lines 4 and 91)

## Scope

Docs/governance only. The IIAC protocol `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` states `NOT_IN_FORCE` at lines 4 and 91, while `IIAC_POLICY_ACTIVATION_RECORD.json` is `IN_FORCE`. By Operator decision (answer 5 of `DELEGATION-PDC-STANDING.md`) the wording is corrected in the PDC-P7-00 activation batch so that one rebind covers it. The plan describes that later edit (the two lines reworded so that the activation record is the single source of truth for the status; no change of any gate meaning; no other protocol text touched; write set = that single protocol file) and the ordering constraint: the protocol is a material governance path (`MATERIAL_GOVERNANCE_PATHS` in `scripts/governance/iiac_activation_gate.py`), so the edit merges before the IIAC `audit_execution_sha` of PDC-P7-00 and lies inside that audit's scope; `INTEGRATED_BASELINE_SHA` in `PDC_SHA_LADDER.md` is re-fixed afterwards. This ticket and its plan are the only content of the pull request that introduces them; the protocol is **not** edited by that pull request. Product code: no. Plan: `PLAN_T-IGX-040-B.md`.

## Plan digest

Plan digest: c48a13bb8895d1cf4ab5bcceaa6c79307790cb47e9bed85bb8a9b4681ad6893b

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-B.md`](plans/PLAN_T-IGX-040-B.md).

## Approved

Approved: operator-delegate @ 2026-11-14T14:41:26+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P7-ACT.md@1d47a7b170b9bed24ded5fe390ca6c481c146874

Delegate checks (scope (a) of `DELEGATION-PDC-P7-ACT.md`, all passed before this line was written; read at `development` tip `715acd353`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`c48a13bb8895d1cf4ab5bcceaa6c79307790cb47e9bed85bb8a9b4681ad6893b`).
- The write set of the plan is exactly one path, the IIAC protocol file; only its lines 4 and 91 change (both contain the stale `NOT_IN_FORCE` wording, the only two occurrences in that file); every other protocol line stays byte for byte unchanged. The ticket's own lifecycle and approval lines are bookkeeping.
- The planned text changes no rule, gate, role or ACT-N* check (the sentences on the fail-closed gate, the separate operator activation record and the certification session after the ACT-01 correction keep their meaning); no verifier, gate or activation record is edited; no registry, scheduler, engine, trust, policy or audit authority is introduced; no path lies under `intergrax/contracts/` or `intergrax/applications/_shared/`.
- Ordering claim confirmed: the protocol is in `MATERIAL_GOVERNANCE_PATHS` of `scripts/governance/iiac_activation_gate.py`, and rule ACT-N11 reports `stale audit after material governance change` for any change after `audit_execution_sha`; the edit therefore merges before that SHA is fixed.
- The plan is a docs-wording plan: it excludes the audit, certifier work, the activation line and record, and certification (Out of scope; ordering steps (e) to (h) are not owned by this ticket). It is not a certification verdict and not an activation.
- Write-set amendments are not delegated and return to the Operator.

## Test contract

```text
git diff --stat origin/development...HEAD          # for the edit PR: one file, two changed lines
uv run pytest tests/unit/governance/test_iiac_g0_inventory.py tests/unit/governance/test_iiac_activation_gate.py -q
uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force   # expected red (ACT-N11) until PDC-P7-00; recorded
```

The pull request that carries only this ticket and its plan changes two files under `governance/tickets/` and needs no pytest run beyond the plan digest check (`uv run python scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-040-B.md` equals the digest above).

## Out of scope

- Any product path, `scripts/`, `tests/`, `.github/`, the activation record, the IIAC gate code and every other IIAC document.
- Re-fixing `INTEGRATED_BASELINE_SHA`, the EBH-7 re-pin, the IIAC audit, certifier work, the activation line and record (PDC-P7-00), OD signatures, certification.

## Lifecycle state

plan_approved
