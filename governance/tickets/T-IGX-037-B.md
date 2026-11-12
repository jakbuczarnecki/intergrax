# T-IGX-037-B — SCENARIO-FACTORY-X P1: research, competitive analysis and capability inventory (docs only)

## Scope

Publish the cited competitive evidence matrix, the IDE/MCP tooling facts and the re-pinned closed-world capability inventory for SFX (master plan R1, units SFX-P1-01..03, wave W1).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-B.md`.

## Plan digest

Plan digest: a8142bfebf5c48499b2ff11bc5542b7d67aaf8f50cb8bffd64a1e3ba0fbb2a99

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-B.md`](plans/PLAN_T-IGX-037-B.md).

## Approved

Approved: operator-delegate @ 2026-11-12T15:06:03+01:00
Delegation: docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-PRE-FREEZE.md@0465c7f7d253e230c7b25a9223278192a1b01bce

Superseded approval (plan digest before Amendment 1): see commit `250d3aeeb`.

## Test contract

```text
uv run pytest tests/unit/governance -q
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- ADRs (T-IGX-037-C), UX (T-IGX-037-D), IDE toolkit (T-IGX-037-E).
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

quality_gate

## Lease holder

Worktree `_wt/intergrax/sfx-w1-research`, branch `sfx-w1-research` (stacked on `sfx-p0-registration`).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `sfx-w1-research` (upstream `origin/development`), contains `origin/development` @ `0140bc6b5` and the W0 follow-up corrections |
| Implementers | Orchestrator session and implementer/research subagents. None of them wrote an `Approved:` line. Approval: delegate re-approval after plan Amendment 1 (`85ab15622`) |
| Test contract | `tests/unit/governance`: 164 passed, 1 skipped, 8 failed. All 8 failures are in `test_evol_x_real_repo_fresh_clone.py` with `Filename too long` (Windows MAX_PATH: the 174-character evidence path under the 16-character worktree name `sfx-w1-research`). The same file passes 15/15 in the shorter worktree `sfx-p0`. The file is unchanged by SFX and the long path exists on `development`; this is environmental. Log: `.tmp/session/sfx-w1/contract.log` |
| Gates | `igx_gov ticket` OK (push-fast PASS); `pdc_x_inventory.py --check` OK (354 items, 0 UNMAPPED); pre-commit `check_docs_domain_pairs` OK |
| Acceptance | Matrix: 15 products covered across 3 lanes, every cell cited or scored from vendor primary sources; 32 design-implication items. Inventory JSON: 37 rows x 10 fields, `inventory_version` `sfx-capability-inventory.v1+r1`; class counts REUSE_AS_IS 15, REUSE_WITH_CONFIGURATION 11, EXTEND_CONTRACT 2, NEW_SCENARIO_FACTORY_COMPONENT 2, REQUALIFICATION_REQUIRED 4 (CAP-08, 15, 30, 31), DEFERRED_WITH_RATIONALE 1, BLOCKED 2. IDE/MCP facts: SFX-GAP-07 resolved at documentation level. Evidence-quality caveats are stated in the matrix header (partial page reads, inferred zeros, beta features) |
| Product paths | none (`intergrax/`, `agents/`, `applications/`, `scripts/`, `agent-env/` unchanged); IIAC ACT-N11 unaffected |
| Review corrections | R-W1-01..11 handled in the follow-up commit; see the PR |

## Follow-up PR checkpoint (post-merge review corrections)

| Field | Value |
| --- | --- |
| Branch | `sfx-fix-w1w2` (upstream `origin/development`), cut from `development` @ `9d2f8bfc2` after PRs 486 and 487 were merged at 16:01, before their independent reviews finished |
| Test contract | Short-path worktree `sfx-fix`: `tests/unit/governance` + `tests/qualification/roadmap_replay_x`: 199 passed, 1 skipped, 5 failed. The 5 failures are the pre-existing `roadmap_replay_x` group C tests (identical on an unchanged roadmap; PDC_REG_DIAGNOSIS). The 8 `Filename too long` failures of the long-named worktree do not occur here (closes R-W1-10). `pdc_x_inventory --check`, `sync_agent_env --check`, `check_cursor_token_setup` and `igx_gov ticket` pass; `check_harness_adr.py` reports 24 pre-existing index gaps and none about ADR-SFX. Log: `.tmp/session/sfx-fix/contract.log` |
| Line endings | Roadmap and ADR index were LF at the base and had become CRLF (whole-file diffs); 21 SFX files converted back to LF. Roadmap diff against the pre-SFX base is again 88 added / 13 removed lines; all five plan digests are unchanged (the digest normalises line endings) |
| Approvals | No `Approved:` line or plan changed; plan files are untouched, so no re-approval is needed |
