# PLAN T-IGX-066-A — Wave 06 OD-REPLAYX-09 (ROADMAP-REPLAY-X prod-Δ wariant A)

Evidence pin (read): `da5ca5b53fbe510e655b12bc91ab4e6af1cef6e4`  
DUP_X_ACCEPTANCE_SHA (fixed): `a7d83925b375b5f08eb0f175ced2c17fa026bba9`  
Baseline scoped paths @ pin: **88**; unscoped prod paths to disposition: **418**.

## 1. Problem

Mechanical ROADMAP-REPLAY-X @ `da5ca5b53`: **3 FAIL / 13 PASS** in `test_roadmap_replay_x_mechanical_exit_gates.py`.

| Failure | Mechanical cause |
| --- | --- |
| `test_replay_x_prod_delta_empty_since_dup_x` | 418 paths in `git diff DUP_X..HEAD` outside `REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS` |
| `test_replay_x_closure_counters_computed` | `historical_evidence_invalidated_by_later_changes` = **3** (one tier-first unscoped per tier) |
| `test_replay_x_b6_r2_reconciliation_zero` | ECAP narrative counters **0** vs mechanical **3** and **1** |

Vendor neutrality: `vendor_neutrality_replay_failures` = **1** (`unexpected_prod_delta_for_vbn:agents/README.md`) — same scoped-set gap; PROHIBITED unresolved = 0.

## 2. Goal (variant A)

Classify every prod path since DUP-X through **wave/ticket-scoped batches** with explicit dispositions and regression evidence module per batch — not a single undifferentiated allowlist. Refresh ECAP counter table @ **post-PASS commit SHA** only. Do **not** re-pin DUP-X or weaken gate assertions.

## 3. Classification (scope by contract risk — implementation assigns each path to exactly one batch)

Prefix order is **longest-match-wins** when emitting paths (implementer sort). Counts @ `da5ca5b53` from `.tmp/session/w06/replay_prod_delta_buckets_da5ca5b.md`.

| Batch | Prefix / rule | ~paths | Disposition summary | Regression evidence (pytest module) |
| --- | --- | ---: | --- | --- |
| W06-B01 | `intergrax/platform/enterprise_audit/` | ENT | T-IGX-043-M Part E + impact map; gate-meaning unchanged | `tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py` |
| W06-B02 | `intergrax/agent_distribution/` | 14 | T-IGX-051-A/TENANT carrier + dist-scale wiring | `tests/qualification/tenant_x/test_tenant_x_replay_carrier_reclassification.py` |
| W06-B03 | `intergrax/applications/_shared/` | 71 | MCP/harness Tier-1 host composition (EQO 043 waves) | `tests/qualification/mcp_server_x/test_mcp_server_x_p7_l1_contract.py` |
| W06-B04 | `intergrax/runtime/` | 94 | RW execution/orchestration composition (043-G/J/B) | `tests/qualification/state_x/test_state_x_p0_baseline.py` |
| W06-B05 | `intergrax/integrations/` | 33 | Provider bundles + guardrail (CONFIG/VBN surface) | `tests/qualification/config_x/test_config_x_qualification_gates.py` |
| W06-B06 | `intergrax/rag/` | 21 | RAG bootstrap/lab opt-in | `tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py` |
| W06-B07 | `intergrax/contracts/` | 15 | Contract expansion (043-G neutrals) | `tests/qualification/compat_x/test_compat_x_inventory_gates.py` |
| W06-B08 | `intergrax/llm_adapters/` | 31 | Adapter wiring (non-vendor-lock) | `tests/qualification/config_x/test_config_x_r1_r1_provider_surface_gates.py` |
| W06-B09 | `intergrax/` (remaining buckets) | 64 | Residual Tier-0/1 (hosting, tools, skills, eval, etc.) | `tests/qualification/trace_x/test_trace_x_p6_closed_world_gates.py` |
| W06-B10 | `agents/` (all tiers) | 22 | Notebooks/qualifiers; non-prod or wave-local | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py` |
| W06-B11 | `applications/` (all tiers) | 39 | Tier-3 host/MCP drift (051, LKW, legal) | `tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py` |

Paths already in PRE-ENT L2/L3 scoped set remain unchanged; W06 batches add **only** paths present in `git diff DUP_X..HEAD` but not yet scoped.

**Qualification head SHA** for all W06 rows: `da5ca5b53fbe510e655b12bc91ab4e6af1cef6e4` (constant `W06_EQO_REPLAY_QUALIFICATION_HEAD`).

## 4. Exact write-set

```text
tests/qualification/roadmap_replay_x/_w06_eqo_prod_delta.py          # new — W06_EQO_PROD_DELTA_BATCHES SSOT
tests/qualification/roadmap_replay_x/_prod_delta_disposition.py      # merge W06 rows into REPLAY_X_L2_SCOPED_PROD_DELTA
docs/project/maintainers/qualification/pre_ent_enterprise_certification/W06_REPLAY_PROD_DELTA.md
docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md   # append Wave 06 entry @ evidence SHA
docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md  # six counters @ post-PASS SHA
governance/tickets/T-IGX-066-A.md
governance/tickets/plans/PLAN_T-IGX-066-A.md
```

**Forbidden:** edits to `test_roadmap_replay_x_mechanical_exit_gates.py` assertions; DUP-X pin change; FRZ promotion; roadmap parent CLOSED beyond OD-REPLAYX-08.

## 5. Dependencies

- AUTH-01 (`T-IGX-043-M`) published @ `da5ca5b53` (enterprise_audit paths classified in W06-B01).
- DUP-X pin `a7d83925` and existing PRE-ENT L3 batches (`T-IGX-034-E`) remain authoritative for historical rows.
- Operator `Approved:` referencing this plan digest and OD-REPLAYX-09.

Serialization: no concurrent edits to `_prod_delta_disposition.py` outside this ticket during implementation.

## 6. Acceptance tests

```text
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py -q
uv run pytest tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py -q
```

**PASS criteria:** 16/16 mechanical exit gates; all six closure counters **0**; `replay_r2_reconciliation_violations()` empty after ECAP refresh.

## 7. Negative / regression tests

Run batch regression modules (single pytest invocation):

```text
uv run pytest \
  tests/qualification/ent_audit_x/unit/test_changed_since_impact_map.py \
  tests/qualification/roadmap_replay_x/test_roadmap_replay_x_scoped_path_net_zero.py \
  tests/qualification/dup_x/test_dup_x_mechanical_exit_gates.py \
  -q --tb=line
```

If W06-B04 touches STATE-sensitive surfaces per batch table, add:

```text
uv run pytest tests/qualification/state_x/test_state_x_p0_baseline.py -q --tb=line
```

## 8. Compatibility

- Historical ROADMAP-REPLAY-X closure @ OD-REPLAYX-08 remains **scoped**; this ticket records prod-Δ dispositions only.
- CONFIG-X @ `fac9a43f` PASS artifacts unchanged (qualification SSOT update only).
- No product-path semantic changes.

## 9. Rollback

```text
git revert <merge-commit>  # disposition + docs only
```

Restore `_prod_delta_disposition.py` without W06 merge; remove `_w06_eqo_prod_delta.py`; revert ECAP/drift log append. Expect mechanical gates RED again (acceptable rollback state).

## 10. Independent review

- `igx-reviewer` on full diff; **REJECT** if gate assertions weakened, undischarged paths remain, or ECAP counters updated before mechanical PASS.
- Not G6 / not EQO global closure.

## 11. Publication

- Commit(s) with `Ref: governance/tickets/T-IGX-066-A.md`; no AI-tool attribution trailers.
- `uv run python -m scripts.governance.igx_gov ticket` on clean `development`.
- `git push origin development` (fast-forward only).