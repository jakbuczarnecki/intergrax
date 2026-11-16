# T-IGX-042-B — EQO-A remediation batch DB-F2: decision-dependent owner-document hunks

## Scope

Documentation-only changes to 23 domain owner documents (`docs/project/architecture/` and `docs/project/capabilities/`), 36 hunks drafted from the EQO-A Campaign A dossier findings A4-013, A4-022, A4-023, A4-029, A8-002, A8-004, A8-005, A8-007, A8-008, A8-009, A8-011, A8-013, A8-014, A8-015, A8-016, A8-019 and A8-023. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Each included hunk was blocked on an open question that the Operator has since decided (blocker-to-decision table in the plan, section 2). Out of scope: hunk H34 (`EVOL_X_CHANGE_SAFETY.md`, blocked on W9:Q5, not decided), every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, qualification records, `governance/` apart from this ticket and its plan, and the files owned by batch DB-F1 (ticket T-IGX-042-A). No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 90a0f35d7d8690063df67eea3edd983c9f46cba0471b56b38095829c23f013cb

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-B.md`](plans/PLAN_T-IGX-042-B.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T13:43:12+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@c98a0d7d67a0b52b8e7e0e279baa0d906bb9c45d

## Test contract

One process at a time, under the shared pytest lock:

- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/unit/docs/ tests/qualification/pdc_x/p6_repair/test_p6r_feature_docs.py tests/unit/runtime/architecture/test_gov_final_1_documentation_regression_gates.py -p no:cacheprovider`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 23 files plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, qualification records, `EVOL_X_CHANGE_SAFETY.md` (W9:Q5), concern IDs (Operator concern list pending), plan stubs for COMPAT_X / PROOF_DATA / TOKEN cache-prefix, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
