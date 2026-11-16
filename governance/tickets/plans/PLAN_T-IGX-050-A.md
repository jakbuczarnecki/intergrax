# PLAN T-IGX-050-A — ARCH-FREEZE program W0: decision register, SSOT reconciliation (part A) and plan publication

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementing session for the orchestrator of MEP-AF2 v2.1. Nothing here is approved, applied or audited.

## 1. Purpose

Record the Operator's decisions in the repository, bring the stale ARCH-FREEZE package (written before IIAC closed EVOL-X and ENT-AUDIT-X, before PDC-X closed, and before the merge of the EQO-A phase-2 plan into MEP-AF2) up to date in part A, and publish the plan sources so they are tracked and digest-pinned. Documentation only. This ticket certifies nothing, promotes no FRZ criterion, changes no stage status and edits no product path.

## 2. Scope

| Node | Files | Change |
| --- | --- | --- |
| T-AF2-OD-RECORD | `ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` | Table form; original rows kept with statuses corrected (OD-ENT-08 COMPLETE, OD-ENT-01 executed, OD-ENT-02 compensating, OD-EVOL-11 / OD-EVOL-FORMAL / OD-ENT-00 superseded by IIAC); new rows for OD-AF-00, packets 1 and 2 (including OD-AF-01a), OD-REM-01..08, C1–C10, DS-1..DS-6, and the open M3–M5 decisions; OD-ARCH-FREEZE-01 OPEN |
| T-AF2-SSOT-RECONCILE-A | `ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md` | New last column `V21_QUALIFICATION` (54 rows); gate notes for MCP-SERVER-X and PDC-X marked SATISFIED with roadmap rows; pointer that MEP-AF2 v2.1 supersedes "task count remains 54" |
| | `ARCH_FREEZE_SHA_LADDER.md` | Rows `AUDIT_BASELINE_2`, G2, R, FCS, M3 (the last four `NOT SET`) |
| | `ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md` | Dated current-state addendum; entry still NOT AUTHORIZED; blockers attributed to plan section A |
| | `ARCH_FREEZE_INDEPENDENT_AUDIT_AND_CLOSURE_MATRIX.md` | OD-ENT-08 row corrected; staleness note for the other rows |
| | `ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md`, `ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md` | Header notes only (audit accepted at G6 `3d96691f1`, closure document `950097ea4`; R1–R6 superseded by MEP-AF2 v2.1; historical FRZ counts) |
| T-AF2-PLAN-PUBLISH | `arch_freeze_program/` (new) | Plan copy, four WP-12 records, README, `SOURCES.sha256`; secrets/personal-data scan; two machine-path redactions; LF normalization |

## 3. Explicitly out of scope

- `ARCH_FREEZE_FRZ_ACCEPTANCE_MATRIX.md` and `ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md` (PR #606; then T-AF2-SSOT-RECONCILE-B).
- `docs/project/maintainers/qualification/README.md` (PR #607).
- The roadmap, the FRZ checklist, IIAC records and scripts, PDC-X record directories, `scripts/`, `tests/`, `.github/`, product paths.
- The validator amendment (T-AF2-VALIDATOR-AMEND) and computed counters: the WBS file keeps 54 rows.
- Any certification statement: roadmap labels are quoted and attributed, not re-asserted; plan statements are attributed to the plan.

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md
docs/project/maintainers/qualification/ARCH_FREEZE_WBS_AND_DEPENDENCY_DAG.md
docs/project/maintainers/qualification/ARCH_FREEZE_SHA_LADDER.md
docs/project/maintainers/qualification/ARCH_FREEZE_PRE_ADMISSION_READINESS_REPORT.md
docs/project/maintainers/qualification/ARCH_FREEZE_INDEPENDENT_AUDIT_AND_CLOSURE_MATRIX.md
docs/project/maintainers/qualification/ARCH_FREEZE_EXECUTION_WAVE1_INDEPENDENT_AUDIT_HANDOFF.md
docs/project/maintainers/qualification/ARCH_FREEZE_COMPLETE_IMPLEMENTATION_AND_CERTIFICATION_MASTER_PLAN.md
docs/project/maintainers/qualification/arch_freeze_program/MEP-AF2_v2.1.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_DAG.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_TICKETS_DRAFT.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_AUTHORIZATION_REQUESTS.md
docs/project/maintainers/qualification/arch_freeze_program/WP12_OPEN_QUESTIONS.md
docs/project/maintainers/qualification/arch_freeze_program/README.md
docs/project/maintainers/qualification/arch_freeze_program/SOURCES.sha256
governance/tickets/T-IGX-050-A.md
governance/tickets/plans/PLAN_T-IGX-050-A.md
```

## 5. Method

1. Worktree outside the repository at `<parent>/_wt/intergrax/af2-w0-ssot-a`, branch `af2/w0-ssot-a` cut from a freshly fetched `origin/development` (`7960ae7f6c82da60ca9acdc0ceebf63810002a42`), as granted by OD-AF-02.
2. The WBS column is generated mechanically from plan section C; the script asserts the 54 TASK_IDs and the counts 6 / 12 / 1 / 1 / 7 / 1 / 26 before writing. The `V21_QUALIFICATION` column is appended after the existing 14 columns, which the validator indexes by position.
3. Every factual sentence added to a record is checked against `origin/development` and listed as `path@sha:line` in `.tmp/session/arch-freeze/W0-ssot-a/FACT_CHECK.md` (session record).
4. Plan-publish files are scanned for tokens, e-mail addresses, user-profile paths and credentials before copying; machine-specific paths are redacted and listed in the PR body.
5. Commit without any AI-tool attribution, with the trailer `Ref: governance/tickets/T-IGX-050-A.md`. Push with the upstream set first and an explicit refspec; open a draft PR against `development`.

## 6. Test contract

As in the ticket: one `uv run` process at a time under the lock; the traceability validator and the docs domain-pair check; reviewer checks of the diff stat and of the counts.

## 7. Risks and ordering

- **Conflicts with PR #606 and PR #607.** Neither file set is touched here. The orchestrator re-checks that #606's hunks do not overlap the files edited here.
- **Stale claims.** The pre-admission addendum attributes the blockers to MEP-AF2 v2.1 section A and says they are not re-verified; the roadmap decides every closure label.
- **Ticket numbering.** The id `T-IGX-050-A` follows the renumbering decision OD-AF-01a (packet 2): ARCH-FREEZE tickets use T-IGX-050/051/052 because T-IGX-043-A..F are EQO-A drafts. The first publication of this batch (PR #611, ticket id `T-IGX-043-A`) is superseded by this one.
- **EBH-7 / IIAC interplay.** Docs under `qualification/` are not product paths; whether the EBH-7 allowlist needs a re-pin for them is for the orchestrator to confirm before merge.

## 8. Rollback

Revert the single PR. No data, configuration or code path is affected.

## 9. Closure condition

PR merged after review; the ticket records the merge SHA. No finding, FRZ row, stage or WBS task is marked closed or verified by this ticket.
