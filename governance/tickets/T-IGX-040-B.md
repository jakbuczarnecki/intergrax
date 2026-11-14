# T-IGX-040-B — PDC-X finding PDC-BL-F01: correct the `NOT_IN_FORCE` wording in the IIAC protocol header (lines 4 and 91)

## Scope

Docs/governance only. The IIAC protocol `docs/project/maintainers/qualification/INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` states `NOT_IN_FORCE` at lines 4 and 91, while `IIAC_POLICY_ACTIVATION_RECORD.json` is `IN_FORCE`. By Operator decision (answer 5 of `DELEGATION-PDC-STANDING.md`) the wording is corrected in the PDC-P7-00 activation batch so that one rebind covers it. The plan describes that later edit (the two lines reworded so that the activation record is the single source of truth for the status; no change of any gate meaning; no other protocol text touched; write set = that single protocol file) and the ordering constraint: the protocol is a material governance path (`MATERIAL_GOVERNANCE_PATHS` in `scripts/governance/iiac_activation_gate.py`), so the edit merges before the IIAC `audit_execution_sha` of PDC-P7-00 and lies inside that audit's scope; `INTEGRATED_BASELINE_SHA` in `PDC_SHA_LADDER.md` is re-fixed afterwards. This ticket and its plan are the only content of the pull request that introduces them; the protocol is **not** edited by that pull request. Product code: no. Plan: `PLAN_T-IGX-040-B.md`.

## Plan digest

Plan digest: c48a13bb8895d1cf4ab5bcceaa6c79307790cb47e9bed85bb8a9b4681ad6893b

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-040-B.md`](plans/PLAN_T-IGX-040-B.md).

## Approved

Not approved. The plan is unsigned: it awaits a separate delegate session (or the operator) to review it and write the approval line. The implementer of this ticket does not write that line.

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

plan_pending
