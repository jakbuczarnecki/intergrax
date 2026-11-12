# T-IGX-036-E — PDC-X P3-08: architecture lock record (Operator decisions, ADR acceptance)

## Scope

Record the Operator's P3-gate decisions ("Accept all A", 2026-11-12) in OD-PDC-SCOPE-01..07, OD-PDC-PATHB-01, OD-PDC-ADR-01 and OD-PDC-IIAC-SCOPE. Set ADR-PDC-01..08 to Accepted. Apply the selected variants to the requirements matrix. Documentation and scripts only. Plan: `PLAN_T-IGX-036-E.md`.

Note: the master plan assigned the letter E to P4. P4 therefore moves to the next free ticket letter.

## Plan digest

Plan digest: a41da8115b29bbf40945da7300e9212619fe7603ad30b161ed9019c382efa127

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-036-E.md`](plans/PLAN_T-IGX-036-E.md).

## Approved

Approved: operator @ 2026-11-12T07:11:25+01:00

Provenance: the Operator answered the P3-gate question in chat with the words "Accept all A". This ticket only records those decisions. The line was transcribed on the Operator's instruction by the implementer session; it is the Operator's decision, not an implementer sign-off.

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_requirements.py tests/unit/governance/test_pdc_x_inventory.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

OD-PDC-FRZ-01, OD-PDC-ARCHF-01 and OD-PDC-SEQ-01; IIAC activation; P4 design; product code; FRZ status changes.

## Lifecycle state

quality_gate

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `pdc-x-p1p3` (PR #463) |
| ODs approved (Option A) | OD-PDC-SCOPE-01..07, OD-PDC-PATHB-01, OD-PDC-ADR-01, OD-PDC-IIAC-SCOPE (10) |
| ADRs | ADR-PDC-01..08 Accepted |
| Matrix after resolution | 172 rows; 5 `DEFERRED_OUTSIDE_FREEZE_SCOPE` (each cites an approved OD); 1 `BLOCKED_EXACT_REASON` (PDC-D6-R017, awaiting MCP-SERVER-X P8); 107 freeze blockers with closure lanes |
| `pdc_x_requirements.py --check` | OK |
| Unit tests | 19 passed |
| Still pending | OD-PDC-FRZ-01, OD-PDC-ARCHF-01, OD-PDC-SEQ-01; IIAC header sub-question |
