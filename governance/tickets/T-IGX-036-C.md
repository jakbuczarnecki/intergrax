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

closed

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `pdc-x-p1p3` |
| Audit head | `1ebd2ab4c2e6b7b03c858f0ede172466cac1f089` (zero product-path delta vs `origin/development` `bc9ef52b`) |
| `pdc_x_requirements.py --check` | `[OK] 172 rows, schema valid, FRZ 225/26` |
| `test_pdc_x_requirements.py` | 9 passed (with inventory tests: 17 passed) |
| P2-09 replay (one bounded pytest, `--extra dev-ci --extra dev-unit-cert`) | 1514 passed, 13 failed, 2 skipped (549.70s). Failures are pre-existing on development: `ebh_5` 4, `governance` 8 (GR-10, GR-12), `plug_03` 1. Recorded as PDC-D3-R024, PDC-D6-R028, PDC-D4-R020, PDC-D1-R022; reuse of EBH-5 automatically downgraded to requalification |
| `igx_gov ticket` | `[INFO] igx_gov ticket: OK` |

**Scope note:** the replay findings are routed to a regression-repair lane `PDC-P5-REG` (root cause first: stale gate vs real regression). That lane is not in the master-plan WBS; P4 places it, and any product fix in it is gated by OD-PDC-SEQ-01.

**Closure:** merged by the Operator in PR #463 (`dd2055e19c64`), 2026-11-12. Not a stage closure.
