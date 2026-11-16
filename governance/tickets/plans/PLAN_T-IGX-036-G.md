# PLAN T-IGX-036-G — PDC-P4 implementation design

Parent: PDC-X master plan [`PLAN_T-IGX-036-A.md`](PLAN_T-IGX-036-A.md) §G "PDC-P4". Inputs:

- the locked architecture: ADR-PDC-01..08 Accepted under OD-PDC-ADR-01;
- the requirements matrix `PDC_DOMAIN_REQUIREMENTS_MATRIX.json` (172 rows);
- OD-PDC-SEQ-01: product merges wait for MCP-SERVER-X P8.

Documentation, scripts and tests only. No product code.

## Deliverables

1. **PDC-P4-01 — implementation matrix.** Add `scripts/qualification/pdc_x_wbs.py`. It reads the requirements matrix and a checked-in batch table, then renders `PDC_IMPLEMENTATION_WBS.md`. The WBS assigns every requirement that needs work beyond replay (gap classes, plus `CURRENT_HEAD_REQUALIFICATION` rows that depend on a red gate) to exactly one batch. Each batch records:
   - lane, ticket ID reserved for P5;
   - files owned (write set), requirement IDs, tests, ADR;
   - migration/COMPAT step, dependencies, wave.

   `--check` fails on:
   - an unassigned gap row, or a row assigned to two batches;
   - two batches in the same wave owning the same file or directory prefix;
   - a stale rendered output.
2. **PDC-P4-02 — dependency DAG.** `PDC_DEPENDENCY_DAG.md`, generated from the same batch table: a mermaid graph, the critical path, the waves, and the Operator gates (OD-PDC-SEQ-01, MCP-SERVER-X P8, ACT-N11).
3. **PDC-P4-03 — cross-domain test matrix.** `PDC_CROSS_DOMAIN_TEST_MATRIX.md`: XD-01..XD-12 with a concrete planned test module under `tests/qualification/pdc_x/` for each, the domains involved, the real-integration requirement (no mocks on mandatory paths), the requirement IDs proven, and the owner batch (PDC-P6).
4. **PDC-P4-04 — audit checkpoints and evidence spec.** For each batch: the exact-SHA evidence required (implementation SHA, publication SHA, test log path) and the IIAC auditor session that will certify it (from OD-PDC-IIAC-SCOPE). P5 lane ticket IDs are **reserved** in the WBS. Ticket files are scaffolded at P5 entry, each with its own plan, so that draft tickets do not go stale while P5 merges wait on MCP-SERVER-X.
5. **ECAP traceability note for D8** (decision PDC-P3-D04): one line in the ECAP baseline notes that points to the accepted artifact-lifecycle feature (ADR-PDC-08).
6. **Placement of the `PDC-P5-REG` lane** in the WBS, using the red-gate diagnosis produced during wave A. Fixes that touch only tests and docs can run before MCP-SERVER-X P8. Product-path fixes follow OD-PDC-SEQ-01.
7. A unit test `tests/unit/governance/test_pdc_x_wbs.py` covering the `--check` rules.

## Acceptance

- 100% of the work-requiring requirement rows are mapped to exactly one batch.
- No two same-wave batches share a write path.
- The critical path and Operator gates are explicit.
- XD-01..12 each have a planned real-integration test.
- FRZ counts are unchanged.
- No product code.

## Test contract

```text
uv run python scripts/qualification/pdc_x_wbs.py --check
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run pytest tests/unit/governance/test_pdc_x_wbs.py tests/unit/governance/test_pdc_x_requirements.py -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product code; P5 implementation; scaffolding P5 ticket files; FRZ status changes; IIAC activation.
