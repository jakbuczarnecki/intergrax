# T-IGX-093-A — AHE W1-B01 canonical ADAS contracts and candidate archive

## Scope

Implement roadmap package **AHE-W1-01**: canonical ADAS typed contracts (§13.0–§13.7 identities) and append-only candidate archive foundation for the configuration-only LAB profile. Obligations **AHI-ADAS-10.1** through **AHI-ADAS-10.14**.

Canonical plan: [`plans/PLAN_T-IGX-093-A.md`](plans/PLAN_T-IGX-093-A.md).

## Plan digest

Plan digest: eb98e85bc95600ee626570ab249cfac73d7b9ac3012aff9d08f68b25f630ce16

## Approved

Operator approval pending.

Neither OD-AHE-01 nor OD-AHE-02 constitutes approval of this exact implementation plan.

## Test contract

Sequential execution (preserve exit codes and evidence):

- `uv run --no-sync pytest -q tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync pytest -q tests/unit/runtime/adaptive/ --ignore=tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync ruff check intergrax/runtime/adaptive/agent_design_search/ tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync pyright intergrax/runtime/adaptive/agent_design_search/`
- `uv run --no-sync python -m compileall -q intergrax/runtime/adaptive/agent_design_search/`
- `uv run --no-sync python -m scripts.governance.igx_gov ticket`

## Out of scope

Execution engine, strategy implementation, static gate, evaluation runner, search controller, experiment bridge, production activation or promotion; modifications to parent adaptive runtime package exports or existing profile stores; ARCH-FREEZE, Scenario Factory runtime, Intelligence master plan; governance publication authorization file changes; G6 or IIAC certification claims by implementer.

## Lifecycle state

plan_pending