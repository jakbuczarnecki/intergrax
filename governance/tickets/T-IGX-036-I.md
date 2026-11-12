# T-IGX-036-I — PDC-X P5-REG batch B-REG-T: stale-gate repair (wave B, tests only)

## Scope

Repair 12 stale qualification gates diagnosed in `PDC_REG_DIAGNOSIS.md`:

- `ebh_5`: 4 tests, PDC-D3-R024;
- GR-12: 4 tests, PDC-D4-R020;
- GR-10: 4 tests, PDC-D6-R028.

Also produce a read-only diagnosis record of the other pre-existing red qualification tests (`ebh_7`, `dist_scale_x`, `roadmap_replay_x`). Tests and docs only. Plan: `PLAN_T-IGX-036-I.md`.

## Plan digest

Plan digest: b7d235f47d5f5b3ae02704ebc8ed83be8d9d547fdd3730b6ea8aae201011c494

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-I.md`](plans/PLAN_T-IGX-036-I.md).

## Approved

Approved: operator-delegate @ 2026-11-12T07:46:15+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-WAVE-B.md@1b49c58cdaa0776e519901aec02e66fb714cc28a

## Test contract

```text
uv run pytest tests/qualification/ebh_5 tests/qualification/governance tests/qualification/ebh_7 -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Product code.
- The `ollama` regression (B-REG-P).
- Fixes for `ebh_7`, `dist_scale_x` or `roadmap_replay_x`.
- FRZ changes, OD signatures and merges.

## Lifecycle state

quality_gate

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `pdc-x-reg-t` |
| Implementers | Two implementer subagents (ebh_5; GR-10) and the PDC orchestrator (GR-12, integration, commit). None of them wrote the Approved line |
| Test contract | `pytest tests/qualification/ebh_5 tests/qualification/governance tests/qualification/ebh_7`: 992 passed, 11 failed, 9 skipped. ebh_5 has 243 passed and 0 failed; governance has 403 passed and 0 failed. The 11 ebh_7 failures are the same set as before the change (pinned EBH-7 evidence; see `PDC_REG_DIAGNOSIS.md`, group B). Log: `.tmp/proof/pdc-x-reg-t/contract.{log,xml}` |
| Non-weakening | GR-10: 13 source mutations, each rejected by the updated gates. GR-12: the requirement for revision-bound authority is kept; the lab-compat default is the same pattern the other STRICT host tests use. ebh_5: expectations are derived from the SSOT modules; every N/A requires a rationale |
| Extra GR-12 fix | A fixture closes every PRODUCT host runtime. Before it, the process hung after the tests passed |
| Product paths | none changed (`git diff --name-only` has no `intergrax/`, `agents/` or `applications/` path) |
| Writes outside the B-REG-T write set (delegate note 1) | `PDC_REG_DIAGNOSIS.md`, `scripts/qualification/pdc_x_requirements.py` and the regenerated matrix outputs. Documentation and scripts only, recording evidence |
| Delegate note 2 | Forwarded to the P7 auditor IA-D3: check that every RPL-F N/A has a rationale and that PASS is limited to the active proof set |
