# PLAN T-IGX-050-H — ARCH-FREEZE W2 node T-AF2-VBN-DOSSIER (draft): VBN freeze SHA dossier and SQLite annex

Status: DRAFT for Operator review. Drafted by the implementing session. Nothing here is approved, applied or audited.

## 1. Purpose

The freeze needs an independently accepted vendor/backend neutrality dossier (sections 2.1 to 2.7 and the SQLite annex of the VBN evidence specification). Plan MEP-AF2 v2.1 section H step 1 asks for a W2 draft that the W5 final generation (ticket T-IGX-052-C) can start from. This batch brings sections 2.1 to 2.6 of `ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md` up to date at the current head of `origin/development`, adds the SQLite annex (section 3, eight questions, question 8 tied to the DIST-SCALE-X envelope per DP-62), names every fixed substrate of section 2.4 as needing independent ratification, and makes no acceptance claim. FREEZE_CERTIFICATION_SHA stays TBD. It certifies nothing, promotes no FRZ criterion and edits no product path.

## 2. Scope

| Item | Path | Change |
|---|---|---|
| Dossier | `docs/project/maintainers/qualification/ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md` | rewritten in place: draft status block, sections 2.1 to 2.7 updated, new section 3, fixed-substrate ratification table, draft limits |
| Ticket and plan | `governance/tickets/T-IGX-050-H.md`, this file | new |

## 3. Method

1. Paths cited in the dossier are checked to exist in the tree; links are checked to resolve.
2. Counts come from one read-only call of the pure accessor functions of the EBH-5 register modules (inventory, matrix, fixed-substrate registry, SQLite touchpoint registry, closed-world validation) at the head commit, run once under the shared lock with the main checkout's interpreter and the worktree on the import path. It is not a pytest run and not a certifying run; the log stays under `.tmp/`.
3. Figures from the closed-world result and the S1 diagnostic are quoted with their (older) baseline and marked as not re-measured. Dispositions are quoted from `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`.
4. Question 8 of the annex uses only the recorded DIST-SCALE-X records and the DP-62 text; it claims no multi-instance validity.
5. Every added factual sentence is checked and recorded as `path@sha:line` or as a quoted source in `FACT_CHECK.md` (under `.tmp/`, not committed).

## 4. Out of scope

Any change under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`; `pyproject.toml`, `uv.lock`; the roadmap and the FRZ checklist; every `IIAC_*` file and script; the PDC-X record directories; any other qualification document; classifying A4-018 or PDC-D3-R019; writing a decision record for the SQLite single writer or the in-memory graph defaults; running a test suite; any acceptance, ratification or certification claim.

## 5. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_VBN_FREEZE_SHA_DOSSIER.md
governance/tickets/T-IGX-050-H.md
governance/tickets/plans/PLAN_T-IGX-050-H.md
```

## 6. Verification plan

One process at a time under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`:

- `scripts/governance/verify_arch_freeze_plan_traceability.py` (plan validator)
- `scripts/docs/check_docs_domain_pairs.py` (docs pairs)
- link resolution check of the dossier (plain Python)
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths above.

## 7. Risks

- The counts describe the head commit at drafting time; later commits change them. The W5 final replaces them.
- The accessor call imports test-tree modules; a divergence from what pytest-collected gates assert is possible. The dossier says the call is not a certifying run.
- The quoted closed-world and S1 figures are from an older baseline and a non-certifying diagnostic; they are labelled as such.
- The 56 behavioral-registry entries against 48 qualified matrix rows are left unexplained on purpose and flagged for the W5 reconciliation.
