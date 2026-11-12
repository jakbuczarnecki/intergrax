# T-IGX-036-C — PDC-X P2: requirements matrix, gap register, FRZ crosswalk (no product code)

## Scope

PDC-P2 requirements & gap audit: final `PDC-Dn-Rnnn` matrix (23 fields), reuse-delta evidence with per-domain replays, FRZ crosswalk (existing IDs only), gap register. Depends on T-IGX-036-B. Plan: `PLAN_T-IGX-036-C.md`.

Roadmap stage `PDC-X` (§3.2); dossier `docs/project/maintainers/qualification/platform_domain_completeness/`.

## Plan digest

Plan digest: 92635081648d5a229e2836698096b0bde2cf3736ab12aea76bf95174662cc333

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-C.md`](plans/PLAN_T-IGX-036-C.md).

## Approved

Approved: operator-delegate @ 2026-11-12T06:36:29+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-P1-P3.md@94d20d9b110d0b03a87553d51fe86ab80da667d0

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_requirements.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code under `intergrax/`, `agents/`, `applications/`; IIAC protocol/scripts; MCP-SERVER-X files; FRZ status changes; OD signatures; ADR acceptance; merges.

## Lifecycle state

plan_approved
