# T-IGX-036-D — PDC-X P3: ownership matrix, Proposed ADRs, OD drafts (no product code)

## Scope

PDC-P3 decision drafts: ownership matrix, ADR-PDC-01/03..08 Proposed, OD-PDC-SCOPE-01..05 / PATHB-01 / ADR-01 drafts (PENDING). Architecture lock (P3-08) remains an Operator gate. Depends on T-IGX-036-C. Plan: `PLAN_T-IGX-036-D.md`.

Roadmap stage `PDC-X` (§3.2); dossier `docs/project/maintainers/qualification/platform_domain_completeness/`.

## Plan digest

Plan digest: 57a1755d8248a01683ca45944e7e05336d73bb11d3ec84d3d4ddc992101ca16e

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-D.md`](plans/PLAN_T-IGX-036-D.md).

## Approved

Approved: operator-delegate @ 2026-11-12T06:36:29+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P1-P3.md@94d20d9b110d0b03a87553d51fe86ab80da667d0

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/docs/check_docs_domain_pairs.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code under `intergrax/`, `agents/`, `applications/`; IIAC protocol/scripts; MCP-SERVER-X files; FRZ status changes; OD signatures; ADR acceptance; merges.

## Lifecycle state

plan_approved
