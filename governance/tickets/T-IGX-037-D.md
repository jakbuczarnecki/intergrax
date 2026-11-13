# T-IGX-037-D — SCENARIO-FACTORY-X P3: UX design, mock-ups, usability protocol and generalization-set seal (docs only)

## Scope

Publish the SFX UX design with a prototype Artifact, the usability protocol with round-0 baseline, and the sealed generalization set digest (master plan R1, units SFX-P3-01..02, wave W2).

Documentation and governance-declarative files only; authorized before ARCH-FREEZE by OD-SFX-04 and OD-SFX-15. Execution STOPs before SFX-P5. Plan: `PLAN_T-IGX-037-D.md`.

## Plan digest

Plan digest: a50e0cd805076ba0cab5359d0dfabf894676d24a6fc1309882d8fb831a442452

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-037-D.md`](plans/PLAN_T-IGX-037-D.md).

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

- Studio frontend implementation (SFX-P10); revealing the GEN set content.
- Product code, `scripts/`, `agent-env/`, FRZ changes, merges and CLOSED claims.

## Lifecycle state

quality_gate

## Lease holder

Worktree `_wt/intergrax/sfx-w2-ux`, branch `sfx-w2-ux` (stacked on `sfx-p0-registration`).

## Checkpoint

| Field | Value |
| --- | --- |
| Branch | `sfx-w2-ux` (upstream `origin/development`). It contains `origin/development` @ `0140bc6b5` and the whole W1 branch `sfx-w1-research` (the skills reference the W1 schemas), so this PR stays stacked on PR 486 until that one merges |
| Implementers | Orchestrator session and implementer subagents; the GEN set was written by the independent session `sfx-gen-author`. None of them wrote an `Approved:` line. Approval: delegate re-approval after plan Amendment 1 (`85ab15622`) |
| Test contract | `sync_agent_env.py --check` OK (57 adapters + CLAUDE.md block); `test_agent_env_sync.py` 2 passed; `check_cursor_token_setup.py` OK; `igx_gov ticket` OK; `pdc_x_inventory.py --check` OK (354 items, 0 UNMAPPED). Log: `.tmp/session/sfx-w2/contract.log` |
| Acceptance | 17 screens SCR-01..17 and the journeys cover all 11 lifecycle steps; the clickable prototype Artifact is linked from `SFX_UX_DESIGN.md`; usability protocol has every result field `NOT_RUN` (round 0 not run); GEN set sealed by the independent session (6 problems, 6 domains, digest in `SFX_GEN_SET_SEAL.md`, content kept outside the repository). Open UX questions UX-Q-01..08 are carried into the gap analysis |
| Product paths | none changed: no `scripts/`, `intergrax/`, `agents/`, `applications/` or `governance/agent-control-plane/` file differs from `development`; IIAC ACT-N11 unaffected |

## Post-merge review corrections

PR 487 was merged (`c323dbd41`) before its independent review finished. The review (verdict `REVIEW_PASS_WITH_FINDINGS`, 0 BLOCKER/HIGH, 1 MEDIUM, 8 LOW, 3 INFO) was handled in the follow-up PR `sfx-fix-w1w2`:

| Finding | Correction |
| --- | --- |
| R-W2-01, R-W2-02 | Persona briefs aligned with the closed schemas (answer kinds, provenance enums, integer `confidence_bp`; open inventory questions go to `resolution_notes.md`) |
| R-W2-03 | Usability protocol row R0 points to `SFX_UX_DESIGN.md` section 10 |
| R-W2-04 | Prototype figures relabelled as illustrative example values (Artifact version 2); `SFX_UX_DESIGN.md` section 10 says so |
| R-W2-05 | The plan text "CLAUDE.md block does not change" is wrong (one table row was added). The plan is not edited because its digest is bound to the approval; the deviation is recorded in the Checkpoint above and here |
| R-W2-06, R-W2-07 | Ticket D now says UX-Q-01..08; ticket E lease holder names the real branch |
| R-W2-08 | SFX documents normalised to LF |
| R-W2-09 | GEN seal states the real protection (unencrypted shared scratch folder), the residual risk and the move to a restricted location before the P12 reveal |
| R-W2-10..12 | Informational; the CI of the PR remains the confirmation of the test results |

## Follow-up PR checkpoint (post-merge review corrections)

| Field | Value |
| --- | --- |
| Branch | `sfx-fix-w1w2` (upstream `origin/development`), cut from `development` @ `9d2f8bfc2` after PRs 486 and 487 were merged at 16:01, before their independent reviews finished |
| Test contract | Short-path worktree `sfx-fix`: `tests/unit/governance` + `tests/qualification/roadmap_replay_x`: 199 passed, 1 skipped, 5 failed. The 5 failures are the pre-existing `roadmap_replay_x` group C tests (identical on an unchanged roadmap; PDC_REG_DIAGNOSIS). The 8 `Filename too long` failures of the long-named worktree do not occur here (closes R-W1-10). `pdc_x_inventory --check`, `sync_agent_env --check`, `check_cursor_token_setup` and `igx_gov ticket` pass; `check_harness_adr.py` reports 24 pre-existing index gaps and none about ADR-SFX. Log: `.tmp/session/sfx-fix/contract.log` |
| Line endings | Roadmap and ADR index were LF at the base and had become CRLF (whole-file diffs); 21 SFX files converted back to LF. Roadmap diff against the pre-SFX base is again 88 added / 13 removed lines; all five plan digests are unchanged (the digest normalises line endings) |
| Approvals | No `Approved:` line or plan changed; plan files are unchanged in content (line endings normalised CRLF to LF; the plan digest is unchanged), so no re-approval is needed |
