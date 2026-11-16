# PLAN T-IGX-042-C — EQO-A remediation batch DB-D: point-in-time notes on stale registers and matrices

Status: DRAFT for Operator review. Drafted 2026-11-16 from the WP-9 patch set for batch DB-D (`.tmp/session/eqo-a-remediation/WP-9/patches/DB-D/`). Nothing here is approved, applied beyond the draft PR, or audited.

## 1. Purpose

Label stale statuses in seven registers and matrices as point-in-time, citing the dossier baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (EQO-A Campaign A dossier, PR #589) and the record that overtook them. Documentation only. Registers are snapshots: no table row is rewritten, no status is changed, nothing is marked closed. This ticket certifies nothing, promotes no FRZ criterion, closes no stage and edits no product path.

## 2. Scope (8 hunks, 7 files)

| Hunk | File | Findings | Change |
|---|---|---|---|
| H01 | `docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` | A3-002 | Currency note: snapshot; 21 rows differ from the checklist at the baseline; the checklist decides |
| H02 | same | A2-021 | Fill the empty `ACCEPTED_EVIDENCE_SHA` cell of `FRZ-TRC-11` with the SHA the checklist row cites |
| H03 | `docs/project/maintainers/qualification/ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md` | A7-002 | Currency note for `FIND-FORMAL-B6` and `FIND-ACCEPTED-BASELINE`; rows unchanged |
| H05 | `docs/project/maintainers/qualification/EBH_5_OPERATOR_DECISION_REGISTER.md` | A2-033 | Note: three rows (OD-EBH5-03/-05/-06) keep their status; no disposition taken |
| H09 | `docs/project/maintainers/qualification/EBH_6_CONDITIONAL_REMEDIATION_REGISTER.md` | A7-006 | Historical note: register pinned at `202239f7...`; later EBH-6 records named |
| H10 | `docs/project/maintainers/qualification/EBH_6_EVIDENCE_FRESHNESS_LEDGER.md` | A7-006 | Point-in-time note; no CURRENT/Superseded mark changed |
| H16 | `docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md` | A7-002, A7-004 | Note: `P1-ARCH-01` Planned versus APP-EVOL-8 Done; nine Closed rows lack evidence and reopen columns; rows unchanged |
| H17 | `docs/project/maintainers/qualification/DIAGNOSTIC_GAP_LEDGER.md` | A7-002 | Note: DG-003 and DG-005 statuses are point-in-time; rows unchanged |

Every note is an inserted blockquote near the top of the file (H02 is one table cell). No line starting with `|` is added and no existing line is changed except the H02 cell, so the parsers named in the WP-9 notes (`verify_arch_freeze_plan_traceability.py`, `debt_burn_down.py`, the EBH-6 and DUP-X register tests) read the same rows.

## 3. Explicitly not included (9 of the 17 drafted hunks)

- Need an Operator or owner fact that is not in the decision file (7): H04 `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` (supersession of OD-EVOL-11, OD-EVOL-FORMAL, OD-ENT-00), H06 `EBH_6_OPERATOR_DECISION_REGISTER.md`, H07 `DUP_X_OPERATOR_DECISION_REGISTER.md`, H08 `EBH_6_APP_TYP_DEBT_DISPOSITION_REGISTER.md`, H11 `EBH_6_CORE_HARDCODING_FINDINGS_REGISTER.md`, H13 `ent_audit_x/ANNEX_F_GAP_REGISTER.md`, H14 `mcp_server_x/MCP_SERVER_X_GAP_REGISTER.md`.
- H12 `DUP_X_FINDINGS_REGISTER.md`: consisted only of the `EDER label` line.
- H15 `HISTORICAL_STAGE_SHA_AND_EVIDENCE_REGISTRY.md`: blocked on the open question W9:Q4 (closed tickets versus registry-only correction), which the merged decisions DP-01..DP-68 do not map.
- The `EDER label` line (`LIVE | SNAPSHOT_AS_OF_<sha> | SUPERSEDED_BY_EDER`) in all 15 registers: it is authorisation request AR-24 of the decision set (prerequisite AR-23 merged, window PDC) and needs the owner vocabulary of DP-47, which the Operator has not yet confirmed. Owner-action placeholders inside the drafted notes were dropped, not committed.

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md
docs/project/maintainers/qualification/ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md
docs/project/maintainers/qualification/DIAGNOSTIC_GAP_LEDGER.md
docs/project/maintainers/qualification/EBH_5_OPERATOR_DECISION_REGISTER.md
docs/project/maintainers/qualification/EBH_6_CONDITIONAL_REMEDIATION_REGISTER.md
docs/project/maintainers/qualification/EBH_6_EVIDENCE_FRESHNESS_LEDGER.md
docs/project/technical/guides/ARCHITECTURE_DEBT_REGISTER.md
```

Plus this plan and its ticket under `governance/`. None of the seven files is an F10 canonical input (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`), none is named in `ebh_7_p03_independent_replay/ARTIFACT_DIGESTS.json` or `MANIFEST.json`, and none is the roadmap, the FRZ checklist or an EBH-7 allowlisted test. Six sit under `docs/project/maintainers/qualification/`, which the EBH-7 identity script allows as documentation (`verify_ebh7_p03_immutable_evidence.py` allowlist prefixes).

## 5. Method

1. Worktree outside the repository from `origin/development`, branch `docs/eqo-a-db-d-registers`.
2. Notes re-written from the WP-9 patch with the placeholders removed; every fact re-read against the baseline blob (`git show 258bbf6a...:<path>`); line references are to the files at the baseline.
3. Commit without tool attribution; `Ref:` trailer to this ticket.

## 6. Test contract

One process at a time, under the shared pytest lock:

- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py`
- `uv run python scripts/maintenance/check_architecture_debt_register.py`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py tests/unit/runtime/observability/test_obs_coverage_1_certification.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/docs/test_check_docs_domain_pairs.py`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the seven files plus this ticket and plan.

## 7. Risks and ordering

- **EBH-7 pin interplay.** Changes under `docs/project/maintainers/qualification/` are inside the identity allowlist; the batch must land before the final EBH-7 pin batch (DP-10) or be confirmed by the PDC-X session as not requiring a re-pin.
- **Content risk.** Notes quote roadmap labels with attribution to the roadmap row and re-assert nothing; the roadmap row decides status. The H02 cell copies the SHA from the checklist row, it does not verify the evidence.
- **Reviewer.** A reviewer other than the author reads H01-H03 against the roadmap and checklist rows at the baseline.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review. No finding is marked closed or verified by this ticket; the findings listed in section 2 are only partly covered.
