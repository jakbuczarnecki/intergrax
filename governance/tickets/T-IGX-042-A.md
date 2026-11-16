# T-IGX-042-A — EQO-A remediation batch DB-F1: decision-free owner-document corrections

## Scope

Documentation-only correction of eleven domain owner documents (`docs/project/architecture/` and `docs/project/capabilities/`), 18 hunks, drafted from the EQO-A Campaign A dossier findings A4-010, A4-014, A4-021 (with A8-018), A4-023, A4-033 (part), A8-007, A7-002, A8-010 and A8-011. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Out of scope: hunks H18-H21 (A4-038, A4-047), every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist and `governance/`. No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 8cc0b1a63b0c7b85c22bdb2dcf334ebf44f23aa43a31a16c708f319cb8a2613a

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-A.md`](plans/PLAN_T-IGX-042-A.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T13:10:10+01:00
Delegation: D:/Projekty/intergrax/.tmp/session/eqo-a-remediation/FC/OPERATOR_ANSWERS_TRANSCRIPTION.md@sha256:cbb86371ec01828fff5e84683d1cf225e7bacdd7bca5d7692a3e7603dc2cdc85

## Test contract

One process at a time, under the shared pytest lock; results are recorded in the pull request body:

- `uv run pytest tests/unit/runtime/architecture/test_gov_final_1_documentation_regression_gates.py tests/unit/runtime/architecture/test_gov_final_3_*.py tests/unit/runtime/architecture/test_gov_final_4_*.py tests/unit/docs/ -p no:cacheprovider`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- B6-F10 / `RF-GOV-01` binding of `GOVERNED_EXECUTION.md`: static read only, no qualification suite run by the implementing session (details in the plan, sections 6 and 7).
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the 11 files plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, qualification records, `DOCUMENTATION_MAP.md` (batch DB-G), `TOKEN_OPTIMIZATION_CACHE_PREFIX_STABILIZATION.md` and `ADR_PDC_05_EVALUATION_CONTRACTS.md` (blocked on Operator question W9:Q5), `SCENARIO_FACTORY.md` and the new `SFX_SCENARIO_MECHANISM_MAP.md` (needs one Operator fact), FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
