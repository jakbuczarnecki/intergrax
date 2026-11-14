# T-IGX-036-W — PDC-X P6 batch B-P6-E: P6 acceptance run and EBH-7 re-pin

## Scope

P6 acceptance record (XD-01..12, DUP-X/EBH-4 gates, ECAP FULL on the integrated baseline) and the EBH-7 re-pin as a fresh L3 replay (manifest, workflow defaults, replay records, digests). Product code: no. Plan: `PLAN_T-IGX-036-W.md`; dossier: `PDC_EBH7_REPIN_DOSSIER.md`.

## Plan digest

Plan digest: ba4a76668177182af9e4a3d5df0624e9fbe527c9da68508468222fca171ce45d

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-W.md`](plans/PLAN_T-IGX-036-W.md).

## Approved

_Pending: Operator approval. No delegation record covers this ticket; the plan asks the Operator two decisions at signing. The implementer never writes this line._

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

plan_pending
