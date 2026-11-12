# T-IGX-037-C — SCENARIO-FACTORY-X P2: gap analysis, proposed architecture, ADRs and contract designs (docs only)

## Scope

Publish ADR-SFX-001..006 (all Proposed), the SFX contract schema drafts, the canonicalization and metric specifications, and the consolidated gap analysis (master plan R1, units SFX-P2-01..02, wave W1).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-C.md`.

## Plan digest

Plan digest: fcc32ab23746abbeee8d939d5545e2be001aeccc89e95e6d72a0c3b9e016946c

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-C.md`](plans/PLAN_T-IGX-037-C.md).

## Approved

Approved: operator-delegate @ 2026-11-12T15:06:03+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

Superseded approval (plan digest before Amendment 1): see commit `250d3aeeb`.

## Test contract

```text
uv run python scripts/maintenance/check_harness_adr.py
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADR acceptance (OD-SFX-06); implementing schemas as code (SFX-P5+).
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

quality_gate

## Lease holder

Worktree `_wt/intergrax/sfx-w1-research`, branch `sfx-w1-research`.

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `sfx-w1-research` (upstream `origin/development`), contains `origin/development` @ `0140bc6b5` and the W0 follow-up corrections |
| Implementers | Orchestrator session and implementer/research subagents. None of them wrote an `Approved:` line. Approval: delegate re-approval after plan Amendment 1 (`85ab15622`) |
| Test contract | `tests/unit/governance`: 164 passed, 1 skipped, 8 failed. All 8 failures are in `test_evol_x_real_repo_fresh_clone.py` with `Filename too long` (Windows MAX_PATH: the 174-character evidence path under the 16-character worktree name `sfx-w1-research`). The same file passes 15/15 in the shorter worktree `sfx-p0`. The file is unchanged by SFX and the long path exists on `development`; this is environmental. Log: `.tmp/session/sfx-w1/contract.log` |
| Gates | `igx_gov ticket` OK (push-fast PASS); `pdc_x_inventory.py --check` OK (354 items, 0 UNMAPPED); pre-commit `check_docs_domain_pairs` OK |
| Acceptance | `check_harness_adr.py` exits 1 with 24 findings, all pre-existing ADR index gaps; the acceptance criterion is "no finding about ADR-SFX-*", which holds (the six ADRs are indexed in `docs/project/technical/adr/README.md`); a fully clean run is not attainable until the 24 pre-existing gaps are fixed. The same 24 are the set accepted for SEF P1 by OD-SEF-P1-EXCEPTION. 17 schemas valid JSON, each with `$schema` (JSON Schema 2020-12) and a `$id` equal to the file name without the `.schema.json` suffix (for example `sfx.discovery_record.v1` for `sfx.discovery_record.v1.schema.json`), and `additionalProperties:false`. All six ADRs are `Proposed`; none claims `Accepted` |
| Product paths | none (`intergrax/`, `agents/`, `applications/`, `scripts/`, `agent-env/` unchanged); IIAC ACT-N11 unaffected |
| Review corrections | R-W1-01..11 handled in the follow-up commit; see the PR |
