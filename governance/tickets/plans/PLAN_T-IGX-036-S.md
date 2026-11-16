# PLAN T-IGX-036-S — PDC-P6 batch B-P6: cross-domain integration

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P6**: cross-domain integration XD-01..12, DUP-X/EBH-4 replay, full ECAP change-delta) and `PDC_CROSS_DOMAIN_TEST_MATRIX.md`. No ADR. Requirements: PDC-D2-R018, PDC-D6-R019, PDC-D7-R014, PDC-D8-R019 and the scenario-level cross-domain cases XD-01..XD-12.

This batch starts only after every P5 batch (B-D1 .. B-D8 and B-REG-P) is merged on `development`, and those merges wait for MCP-SERVER-X P8 (OD-PDC-SEQ-01). Its own PR is cut from the integrated `development`, writes tests only, and is merged under the same gate. A product merge invalidates the IIAC activation (ACT-N11), so PDC-P7-00 re-activates it.

## Deliverables

1. **XD-01..XD-12 (PDC-P6-01).** One real-integration qualification module per scenario, named as in `PDC_CROSS_DOMAIN_TEST_MATRIX.md` (`tests/qualification/pdc_x/xd/test_xd_01_agent_model.py` .. `test_xd_12_evol_ent_audit.py`). Mandatory paths use the canonical composition roots and real stores (SQLite where a store is needed); fakes are allowed only for external vendors (provider, object storage, credential vendor). Each module proves the requirement rows listed for it in the matrix: PDC-D2-R018 and PDC-D8-R019 (evidence artifacts survive compaction, XD-09), PDC-D6-R019 (cross-tenant negatives in XD-02, XD-06, XD-10) and PDC-D7-R014 (drain plus lease release, XD-05).
2. **Re-point the lane stand-ins.** The lanes wrote cases under `xd_05`, `xd_06`, `xd_08`, `xd_09` and `xd_10` names inside their own `d4`, `d5`, `d6` and `d8` directories, using narrower fixtures (for example XD-05 drove in-flight work through the hosting active-work port instead of the real `WorkerRuntime`). The XD modules here are the integrated versions on the composition roots; the lane copies stay as unit-level proof and are not deleted.
3. **Duplicate-authority and bypass gates (PDC-P6-02).** Replay the DUP-X suite (`tests/qualification/dup_x`) and the EBH-4 composition-root and bypass gates on the integrated baseline, and report zero new duplicate authorities. The implementer locates the EBH-4 gates from the existing qualification and gate inventory and states which files were run.
4. **Full ECAP change-delta (PDC-P6-03).** Run `uv run python scripts/enterprise_audit/run_enterprise_audit.py run --mode FULL` on the integrated baseline once, record the report location, and route every finding to the orchestrator (the integration owner). Findings are not repaired in this batch.
5. **Findings handling.** Product-path findings from the XD runs, the gates or ECAP are recorded with exact file and reason and handed to the orchestrator, who opens a separate integration-repair ticket for them. This batch writes tests only.

Design constraints: real integration on the canonical composition roots, no mocks on mandatory paths, no new test-only authority, and no edits outside `tests/qualification/pdc_x/xd/`.

## Write set

`tests/qualification/pdc_x/xd/`

Nothing outside this set changes. Any need to write elsewhere stops the batch and goes to the orchestrator.

## Protocol

1. Start from the integrated `development` once all P5 batches are merged.
2. Write the XD modules; run them against the integrated baseline. A failing XD case is a finding, not a reason to weaken the case.
3. One `uv run pytest` per verification step (several paths in one command); ruff and pyright on touched files. Do not pass `-p no:cacheprovider`. Process-based cases carry hard timeouts and clean up only their own PIDs.
4. Commit with `Ref: governance/tickets/T-IGX-036-S.md`, push to a branch, open a PR against `development`, and do not merge before the gate.
5. Record the integrated baseline SHA as `INTEGRATED_BASELINE_SHA` input for the SHA ladder; the publication SHA is recorded after the merge.
6. The implementer never certifies the batch and never writes the `Approved:` line.

## Acceptance

- All twelve XD scenarios exist and pass on the integrated baseline, with no mocks on mandatory paths.
- The four requirement rows above have passing tests at the implementation SHA.
- DUP-X and the EBH-4 gates report no new duplicate authority.
- The ECAP FULL run is recorded and every finding is routed.
- Test log under `.tmp/proof/pdc-x-p5/B-P6/`.

## Test contract

```text
uv run pytest tests/qualification/pdc_x -q
uv run python scripts/qualification/pdc_x_requirements.py --check
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

Product-code repairs (a separate integration-repair ticket opened by the orchestrator); FRZ changes; OD signatures; IIAC activation; certification.
