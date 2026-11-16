# PLAN T-IGX-036-E — PDC-P3-08 architecture lock record

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md), §G task PDC-P3-08 ("Accept ADRs (OD-PDC-ADR-01) and publish"; owner Operator / orchestrator). T-IGX-036-D drafted the decisions and explicitly left signing out of scope. This ticket records the Operator's decisions only.

## Operator decision being recorded

On 2026-11-12 the Operator answered the P3 gate question ("Accept the recommended option (A) for all P3-gate ODs?") with **"Accept all A"**.

The P3-gate ODs are:

- OD-PDC-SCOPE-01 .. OD-PDC-SCOPE-07
- OD-PDC-PATHB-01
- OD-PDC-ADR-01
- OD-PDC-IIAC-SCOPE (gate "P3, before P7")

OD-PDC-FRZ-01 (gate P2), OD-PDC-ARCHF-01 (gate P0) and OD-PDC-SEQ-01 (gate P4) are not P3-gate decisions and stay PENDING.

## Deliverables

1. Set each of the ten OD files to `APPROVED — Option A`. Each gets an operator record transcribed with provenance.
2. Update the OD register to match.
3. Set ADR-PDC-01..08 to **Accepted** (by OD-PDC-ADR-01), and fold the selected scope variants into the ADR decision sections:
   - SCOPE-01: implement skill lifecycle (ADR-PDC-01).
   - SCOPE-02: exclude APP-HOST-6/7/9E (ADR-PDC-04).
   - SCOPE-03: exclude the LLM provider plugin layer and the OCI factory (ADR-PDC-03).
   - SCOPE-04: implement governed download (ADR-PDC-08).
   - SCOPE-05: implement recurrence (ADR-PDC-07).
   - SCOPE-06: implement the freshness contract (ADR-PDC-02).
   - SCOPE-07: implement plugin provenance (ADR-PDC-06).
4. Requirements matrix: resolve the `BLOCKED_EXACT_REASON` rows that waited on these ODs. Rows where the OD chose implementation become implementation work with a P5 lane. Rows where it chose exclusion become `DEFERRED_OUTSIDE_FREEZE_SCOPE`, citing the authorizing OD. A validation rule ensures that every DEFERRED row cites an approved OD.
5. Record the architecture lock in `PDC_ARCHITECTURE_DECISIONS.md` and the SHA ladder (P3_PUBLICATION_SHA is set at merge).

## Acceptance

- Ten ODs approved, each with provenance.
- ADRs accepted.
- No `BLOCKED_EXACT_REASON` row remains for a decided OD.
- Remaining blocked rows have exact reasons (for example, MCP-SERVER-X P8).
- FRZ counts unchanged.
- No product code.

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_requirements.py tests/unit/governance/test_pdc_x_inventory.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

OD-PDC-FRZ-01, OD-PDC-ARCHF-01 and OD-PDC-SEQ-01; IIAC activation (ACT-N11); P4 design; product code; FRZ status changes; creating the artifact feature doc pair (that is P4/P5 work after this lock).
