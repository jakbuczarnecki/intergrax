# PLAN T-IGX-050-A2 — ARCH-FREEZE node T-AF2-SSOT-RECONCILE-B: program findings in the closed-world gap register

Status: DRAFT for Operator review. Drafted 2026-11-16 by an implementer session of the ARCH-FREEZE program (plan MEP-AF2 v2.1, node T-AF2-SSOT-RECONCILE-B, wave W2, sections C, F and appendix Z). Nothing here is approved, applied or audited.

## 1. Purpose

Add the program's own findings (FIND-AF2-01..23, MEP-AF2 v2.1 section F) to `docs/project/maintainers/qualification/ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md`, each with owner nodes, the EQO-A identifiers section F names, and a status. Record the disposition of the three historical findings (FIND-VBN-PARTIAL, FIND-FORMAL-B6, FIND-ACCEPTED-BASELINE) and record FIND-AF2-19 as CLOSED (FALSE) with the W0 probe evidence. The node also checks the FRZ acceptance matrix status cells against the checklist and changes them only where they are stale.

The register is a planning record. No row is promoted, closed or certified, except that FIND-AF2-19 states the closure of a hypothesis that the W0 probe showed to be false.

## 2. Scope

- Append to the register, after the existing table: a section with a short note, 23 rows `FIND-AF2-01..23` (the plan lists 09..12 as one row; they are four rows here, one per residual class), and a three-line table for the historical findings. The first column of that table is not a register row, so the validator counts each finding once.
- The wave-1 rows and the currency note stay as written (the note says the rows are parsed by `verify_arch_freeze_plan_traceability.py`).
- FRZ matrix: compare every `CURRENT_STATUS` cell with the checklist at the base. Edit a cell only if it differs. At the base commit no cell differs (225 rows, 0 differences; 108 `PASS`, 8 off-vocabulary `PASS` labels, 109 `OPEN`), so the matrix is not edited.

## 3. Explicitly out of scope

- The FRZ checklist, the roadmap, the IIAC files, every path under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`, `pyproject.toml`, `uv.lock`, the PDC-X record directories.
- Other stale rows of the independent-audit closure matrix (`ARCH_FREEZE_INDEPENDENT_AUDIT_AND_CLOSURE_MATRIX.md`, named in the T-IGX-050-A note); that file is not in this node's write set.
- Any PASS, closure, certification, FRZ promotion or independent-verification claim.

## 4. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_CLOSED_WORLD_GAP_REGISTER.md
governance/tickets/T-IGX-050-A2.md
governance/tickets/plans/PLAN_T-IGX-050-A2.md
```

## 5. Method

1. Worktree outside the repository on branch `af2/w2-ssot-b`, cut from `origin/development` at `332c5217c390202a911ad5592e88a8f366845587`. The register at that commit has no FIND-AF2 row (checked with `git grep`); the PR #606 edit to it is the currency note only.
2. A script in the session output folder appends the section; it refuses to run if a FIND-AF2 row already exists, and it rejects `|` inside a cell.
3. A `FACT_CHECK.md` in the session output folder records each repo fact as `path@sha:line` or as a GitHub run or PR reference.
4. Each cell quotes or compresses plan section F or appendix Z; nothing is added that the plan does not state, except the W0 probe facts for FIND-AF2-19, which were re-read with `gh run view` and `gh pr view`.

## 6. Test contract

One process at a time, taken under the pytest lock, no product test:

- `python scripts/governance/verify_arch_freeze_plan_traceability.py` (expected: exit 0, `findings=41`, which is 18 wave-1 rows plus 23 new rows).
- `python scripts/docs/check_docs_domain_pairs.py`.
- `python -I check_matrix_status.py <worktree>` (session output folder): every matrix `CURRENT_STATUS` cell equals the checklist status (expected: 0 differences).
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths of section 4.

## 7. Risks

- **Validator coupling.** The validator maps every register finding to a `T-...` token on its row. Every new row carries owner-node tokens; the check at the base passes with 41 findings.
- **Plan inconsistency on FIND-ACCEPTED-BASELINE.** Section F names `T-B8-ECAP-16`; appendix Z and section C assign the post-freeze baseline promotion to `T-AF2-POST-BASELINE`. The register follows appendix Z and states the difference in the cell.
- **Ticket letters.** `T-IGX-051-*` letters in rows 09..12 follow the orchestrator dashboard and may be reassigned by T-AF2-RW-PLAN.
- **Pin validity.** The files live under `docs/` and `governance/`; they do not change `intergrax/`, `agents/` or `applications/`. Whether a docs-only change needs an EBH-7 re-pin is decided by the orchestrator under DP-61.
