# T-IGX-036-W — PDC-X P6 batch B-P6-E: P6 acceptance run and EBH-7 re-pin

## Scope

P6 acceptance record (XD-01..12, DUP-X/EBH-4 gates, ECAP FULL on the integrated baseline) and the EBH-7 re-pin as a fresh L3 replay (manifest, workflow defaults, replay records, digests). Product code: no. Plan: `PLAN_T-IGX-036-W.md`; dossier: `PDC_EBH7_REPIN_DOSSIER.md`.

## Plan digest

Plan digest: ff9bcd9fb91fe6d4653741c3dc1baf375455ccb0c43cef4e0543d2d1ce110b00

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-W.md`](plans/PLAN_T-IGX-036-W.md).

## Approved

Approved: operator-delegate @ 2026-11-14T06:51:29+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P6-REPIN.md@15369c672d9c1163151f7f0d498b3b2b24e75c17

## Test contract

```text
uv run pytest tests/qualification/pdc_x -q
uv run pytest tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product repairs, `roadmap_replay_x` dispositions, `dist_scale_x` (Docker), widening the executable-identity allowlist.
- IIAC activation and header fix (PDC-P7-00), FRZ changes, OD signatures and certification.

## Lifecycle state

plan_approved
