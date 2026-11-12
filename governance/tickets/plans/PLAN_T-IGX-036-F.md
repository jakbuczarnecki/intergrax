# PLAN T-IGX-036-F — PDC-X remaining Operator decisions (FRZ-01, ARCHF-01, SEQ-01)

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md), §G. T-IGX-036-E recorded the P3-gate decisions and explicitly excluded these three. This ticket records them only.

## Operator decision being recorded

On 2026-11-12 the Operator answered the question "Approve the three remaining ODs (FRZ-01, ARCHF-01, SEQ-01) as option A?" with **"Approve all A"**.

## Deliverables

1. **OD-PDC-FRZ-01 → APPROVED — Option A.** PDC requirements map to existing FRZ identifiers only. No `FRZ-PDC-*` family is added, and the 225 / 26 counters stay unchanged. Rows are promoted only through the checklist update protocol and the certifier role (PDC-P9-03).
2. **OD-PDC-ARCHF-01 → APPROVED — Option A.** The ARCH-FREEZE WBS predecessor gate stays note-only, as applied in T-IGX-036-A. The task count stays 54, and the generator and `EXPECTED_WBS_TASKS` are unchanged.
3. **OD-PDC-SEQ-01 → APPROVED — Option A.** PDC-P0..P4 proceed in parallel with MCP-SERVER-X. PDC-P5 implementation may proceed in isolated worktrees, but product merges wait until the MCP-SERVER-X P8 certification is published.
4. Update the OD register, the FRZ coverage matrix header, the architecture decisions document and the dossier index to show the approved status.

## Acceptance

- All PDC-X operator decisions drafted so far are decided. The only remaining open item is the IIAC protocol header sub-question.
- FRZ counts are unchanged.
- No product code.

## Test contract

```text
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python scripts/governance/verify_arch_freeze_plan_traceability.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; FRZ status changes; IIAC activation; the IIAC header sub-question; P4 design.
