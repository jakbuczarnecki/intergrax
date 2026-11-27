# T-IGX-051-F — T-AF2-RW-GATES (W3)

Train node **T-AF2-RW-GATES**, ticket **T-IGX-051-F** per `MEP-AF2_v2.1.md` W3 table (`SCOPE`, `PT-WA`; PROD + NONPRODUCT).

## Scope

Close the RW-GATES gaps recorded at HEAD `fc2ccb73dea161156d067019548f88e08f033fd9` in `gate_presence.md` while preserving what is already **PRESENT**: stage-plus-ticket promotion gating (`intergrax/runtime/architecture/agent_promotion.py`, `scripts/gates/check_mvp_promotion_gates.py`, `scripts/gates/check_agent_promotion_eval_gate.py`), `RuntimeEvent` schema (`intergrax/contracts/runtime_event.py`), and eval digest contracts (`intergrax/contracts/evaluation_contracts.py`, `intergrax/runtime/evidence/eval_evidence_contracts.py`).

**Deliver (MEP row T-AF2-RW-GATES acceptance):** gate bound to stage plus ticket; one canonical scenario `get_steps` signature; runtime event payload schema reconciled; digest enforcement where still advisory; NOT_COMPOSED reachability gates that **raise** when a gated capability is composed without its contract (today `intergrax/applications/_shared/deploy_activate_pipeline.py` is pass-through only).

**MEP remediation map (allocation quote):** EX-D5-03, EX-L-01/11/13, EX-D3-04 (`get_steps(context)`), D1-R015, D5-R002, D6-R015 → owner node T-AF2-RW-GATES.

Canonical plan: `governance/tickets/plans/PLAN_T-IGX-051-F.md`.

## Plan digest

Plan digest: 8c9936aaecb16009aed52a3e743393dd293755d897c9cd169baace54ace5b91b

## Approved

Approved: operator-delegate @ 2026-11-24T18:15:00+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/continuation/OPERATOR_DECISIONS_AUTONOMY.md@0889a64bf93003548b73e3e282b09e3bc0be20b8
Delegate session: package-c-publication

## Test contract

After plan approval, one verification step at a time per `intergrax-core` test budget:

- Negative: unrelated governance ticket → promotion **DENIED** (`NT-RW-GATES-PROMO-DENY` or equivalent named in the approved plan).
- Negative: corrupted eval signature or digest → verification **DENIED** (`NT-RW-GATES-DIGEST-DENY`).
- Negative / reachability: composing a feature that is **NOT_COMPOSED** without satisfying the gate contract → **raises** (not pass-through) (`NT-RW-GATES-NC-RAISE`).
- Regression scope: existing promotion gate tests, eval digest tests, and runtime-event observability tests listed in the approved plan §7.
- Contract typing wave-packet check if any `intergrax/contracts/**` path is edited (OD-AF-11; otherwise N/A).

Exact pytest paths and gate script invocations are defined only in the approved plan.

## Out of scope

- Mass typing remediation on `intergrax/contracts/**` (owner **T-AF2-RW-TYP**, ticket **T-IGX-051-G**, OD-AF-11).
- Dependency batch / `uv.lock` port (`T-IGX-051-H`; see `deps_051h_disposition.md` in prep pack).
- Supply-chain, tenant, trust, recovery, and OBS train nodes outside the MEP T-AF2-RW-GATES write set.
- FRZ stage promotion, G6 PASS, merge to `development`, or IIAC re-pin.
- Copying an external roadmap or prior signed draft from outside this repository into tracked trees.

## Lifecycle state

in_progress

## RW traceability (GA.2R)

Traceability only (RW-PLAN-AUDIT finding M7, ticket T-IGX-078-B). This section grants no write authority and changes no scope, plan, plan digest, approval or lifecycle state.

| FRZ row of the RW CODE_FIX inventory | Negative test ids in the RW plan |
|---|---|
| FRZ-DEBT-04 | NT-DEBT-04-1, NT-DEBT-04-2 |
| FRZ-REG-09 | NT-REG-09-1, NT-REG-09-2 |

Planned test files that the write set extracted from this ticket's plan does not contain: `tests/unit/architecture/test_rw_gates_gate_completeness.py`, `tests/unit/architecture/test_rw_gates_partial_inventory.py`. The plan does not cover them yet.

FIND-AF2 mapping that names this ticket: FIND-AF2-13.

Binding: `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json`. Lane, hard predecessors, merge position and mandatory tests of this ticket are in the live plan `docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md`.
