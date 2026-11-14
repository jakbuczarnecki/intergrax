# T-IGX-036-X — PDC-X P6 batch B-P6-F: ECAP VBN-REACHABILITY harness registration order

## Scope

Test-only fix of the registration order in the `ebh_5` package C v2 harness so that the embedding provider of `ollama` registers after the model-serving row (finding R-04 of the P6 acceptance record), with a pinned regression test. Product code: no. Plan: `PLAN_T-IGX-036-X.md`.

## Plan digest

Plan digest: d60e5be25c7ba36c297dad5566cafb4107eaef531bc63d5618bd162e77b5f0cd

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-X.md`](plans/PLAN_T-IGX-036-X.md).

## Approved

Approved: operator-delegate @ 2026-11-14T07:57:44+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-ECAPFIX.md@ece78ade7cf4f67d5a6ec85509e0948e5d6a6cee

## Test contract

```text
uv run pytest tests/qualification/pdc_x/p6_ecapfix tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product changes to the embedding provider or the catalog; the `pdc_x` pytest process that does not exit after its summary line.
- The EBH-7 re-pin, IIAC activation (PDC-P7-00), FRZ changes, OD signatures and certification.

## Lifecycle state

plan_approved
