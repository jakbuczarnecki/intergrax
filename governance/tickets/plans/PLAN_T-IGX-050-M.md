# PLAN T-IGX-050-M — ARCH-FREEZE node T-AF2-REGISTER-UPDATE-1: record the decisions and facts of 2026-11-16/17 in the tracked registers

Status: DRAFT for review. Drafted by the implementing session. Nothing here is approved, applied or audited.

## 1. Purpose

Many decisions and facts of 2026-11-16/17 exist only in untracked session records of the ARCH-FREEZE orchestrator. This batch writes them into the tracked ARCH-FREEZE registers as a documentation-only change. It records what the sources state; it certifies nothing, promotes no FRZ criterion, closes no finding, signs nothing and claims no independent verification. Operator wording is restated as a labelled English rendering, never as a verbatim quote.

## 2. Scope

| Item | Path | Change |
|---|---|---|
| OD register | `docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` | new rows: OD-AF-20, OD-AF-21, OD-AF-21a, OD-AF-DELEG-050B, OD-AF-DELEG-050L, Q-RW-19, Q-WA-05, OD-AF-22, OD-AF-PKT0, OD-AF-PKT0-A1, Q-WB-09, Q-WB-13, Q-WB-14, A4-040 (W-B Part G), OD-AF-PKT1, OD-AF-PKT2 (wave packet 2: five tickets and its lane 2 rulings), DP-54 (assigned). Corrected rows: OD-AF-11 (was OPEN; now the packet 3 decision) and DS-5 (DP-54 no longer open). Header line names this ticket |
| Known-red register | `docs/project/maintainers/qualification/arch_freeze_program/KNOWN_RED_REGISTER.md`, `known_red_register.v1.json` | new row KR-28 (class FIX, owner PT-G01) with the two `tests/acceptance/llm_routing` nodes; totals line updated; the JSON stays valid and consistent with the markdown |
| Debt and exceptions register (EDER) | `docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl`, `reconciliation.json` | four new post-ingest rows: EDER-N-0001 (Q-WA-03), EDER-N-0002 (the DUP-X item for Q-WA-07), EDER-N-0003 (A4-007 seams b1/b2, residual of PT-WB), EDER-N-0004 (DP-54, ticket T-IGX-052-E). `reconciliation.json` gets a `post_ingest_rows` note; the ingest totals are unchanged; its stale `not_stated` text about DP-54 is rewritten |
| Ticket and plan | `governance/tickets/T-IGX-050-M.md`, this file | new |

## 3. Method

1. Each fact comes from a session record named in the row (`.tmp/session/arch-freeze/...`, untracked) or from a tracked file at `origin/development`. Facts about the `arch-freeze/rw-train` branch are cited with the train SHA.
2. Every added factual sentence is checked and recorded as `path@sha:line` in `FACT_CHECK.md` (kept under `.tmp/session/arch-freeze/register-1/`, not committed). Where only a session record supports a sentence, the check says so and the sentence is labelled as quoted.
3. EDER rows follow `register_schema.json` (checked by script; the schema library is not installed, so required fields, enums and patterns are checked by hand). Rows use lifecycle `INGESTED`, severity `UNRATED` and impact `UNKNOWN`: no classification is invented.
4. No existing row is edited except where a new fact contradicts it (OD-AF-11, DS-5, the `not_stated` field of `reconciliation.json`); each such fix is listed in the PR body.

## 4. Out of scope

Everything under `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`; `pyproject.toml`, `uv.lock`; the roadmap and the FRZ checklist; IIAC files; the PDC-X record directories; the DUP-X register files (`DUP_X_*.md`, signed historical records) and the closed-world gap register (its rows are parsed by a script and kept as written); `docs/project/maintainers/qualification/README.md` (it still says DP-54 is open); the Q-RW-10 owner changes for KR-17/18/19; the AR-45/AR-50 trial outcome; the DP-47 data flip (done by PR #656 before this branch was cut from `origin/development`, so the new EDER rows carry `owner_vocabulary_status` `CONFIRMED` like the ingested rows); merging, approval and certification of any kind.

## 5. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md
docs/project/maintainers/qualification/arch_freeze_program/KNOWN_RED_REGISTER.md
docs/project/maintainers/qualification/arch_freeze_program/known_red_register.v1.json
docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl
docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json
governance/tickets/T-IGX-050-M.md
governance/tickets/plans/PLAN_T-IGX-050-M.md
```

## 6. Verification plan

One process at a time under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`:

- JSON and JSONL parse check of the three data files; EDER rows against `register_schema.json` (by script).
- `python -m scripts.governance.lint_iiac_closure_regex` once on its default targets, reported as measured.
- `python -m scripts.governance.verify_ticket_lifecycle`, `verify_plan_approval` and `verify_ticket_approvals` on the ticket, if they run without extra dependencies.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the seven paths above.

No pytest suite covers these registers (a `git grep` of `scripts/`, `tests/` and `.github/` finds no reader of them).

## 7. Risks

- Several facts rest on untracked session records. A reader of the repository cannot open them; the rows say so, and nothing is presented as repository evidence.
- The two KR-28 nodes are registered from a static reading and from the PT-WB review, not from a run. The orchestrator note says they are red at the train base; the static reading shows the patched name still present at the post-PT-WA train SHA and absent after PT-WB. The row records both and claims no SHA of first failure.
- The EDER file is a draft register that is not adopted and whose owner vocabulary is not confirmed in data; new rows carry the same `owner_vocabulary_status` (`CONFIRMED`, after PR #656) and the same row-level `owner_status` `SINGLE_PROPOSED` as the ingested rows; no owner is confirmed at row level.
- The owner of the A4-007 seams b1/b2 residual (EDER-N-0003) is the ticket T-IGX-051-I (node T-AF2-A4007-CALLERS, after PT-FC2), per an orchestrator correction of 2026-11-17; that ticket is not drafted yet, so the row names it as assigned, not as existing.
- `qualification/README.md` still says DP-54 is not decided until a separate docs change fixes it.
