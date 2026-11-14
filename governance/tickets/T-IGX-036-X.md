# T-IGX-036-X — PDC-X P6 batch B-P6-F: ECAP VBN-REACHABILITY, ollama embedding resolve

## Scope

Product fix so that the single `ollama` catalog row resolves the embedding category (factory dispatch on `integration_category`), the harness registration order in the `ebh_5` package C v2 helper, and pinned regression tests (finding R-04 of the P6 acceptance record; Amendment 1 extends the earlier test-only scope). Product code: yes (`intergrax/integrations/providers/{model_serving_runtime,embedding_provider}/ollama/`). Plan: `PLAN_T-IGX-036-X.md`.

## Plan digest

Plan digest: ab2817154f865ed3ca1a898d1feedfcfbc9bcce27aba4ed5a408f4d8bdb8bf48

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-X.md`](plans/PLAN_T-IGX-036-X.md).

## Approved

_Pending: delegate re-signing per Amendment 1 of [`DELEGATION-PDC-P6-ECAPFIX.md`](../../docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-ECAPFIX.md). The implementer never writes this line._

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_ecapfix tests/qualification/pdc_x/reg_p tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Catalog, registry and shared-factory changes, other providers and categories; the `pdc_x` pytest process that does not exit after its summary line.
- The EBH-7 re-pin, IIAC activation (PDC-P7-00), FRZ changes, OD signatures and certification.

## Lifecycle state

plan_pending
