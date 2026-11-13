# T-IGX-036-U — PDC-X P6 batch B-P6-M: model binding

## Scope

Activated agent revision pins the LLM profile digest and an agent's required model features are checked when the adapter is bound (XD-01 findings 9 and 10 from ticket T-IGX-036-S). Touches the hotspot files `intergrax/contracts/agent_contract_meta.py` and `intergrax/applications/_shared/llm_resolver.py` by explicit Operator decision. Product code: yes. Plan: `PLAN_T-IGX-036-U.md`.

## Plan digest

Plan digest: dadbcd8e72ac05f16485bddfad6219f660936019ba888f50181c7c1ffc701729

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-U.md`](plans/PLAN_T-IGX-036-U.md).

## Approved

_Pending: delegate session per [`DELEGATION-PDC-P6-MODEL.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-MODEL.md). The implementer never writes this line._

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_model tests/qualification/pdc_x/xd/test_xd_01_agent_model.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Anything in T-IGX-036-T, persistence schema migrations, IIAC header and activation.
- FRZ changes, OD signatures and certification.

## Lifecycle state

plan_pending
