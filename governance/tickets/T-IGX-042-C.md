# T-IGX-042-C — EQO-A remediation batch DB-D: point-in-time notes on stale registers and matrices

## Scope

Documentation-only point-in-time notes on seven registers and matrices (six under `docs/project/maintainers/qualification/`, one under `docs/project/technical/guides/`), 8 hunks, drafted from the EQO-A Campaign A dossier findings A3-002, A2-021, A2-033, A7-002, A7-004 and A7-006. Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Registers are snapshots: no table row is rewritten, no status is changed, nothing is marked closed. Out of scope: nine drafted hunks that need an Operator or owner fact or an undecided question (listed in the plan), the `EDER label` lines (decision request AR-24), every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist and the rest of `governance/`. No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 16f69b0a43c17c57d351852a735c591a24105b74b91f842dc421fe0c4cd16b50

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-C.md`](plans/PLAN_T-IGX-042-C.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T13:43:41+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@c98a0d7d67a0b52b8e7e0e279baa0d906bb9c45d

## Test contract

One process at a time, under the shared pytest lock:

- `uv run python scripts/governance/verify_arch_freeze_plan_traceability.py`
- `uv run python scripts/maintenance/check_architecture_debt_register.py`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/qualification/ebh_6/test_ebh_6_mechanical_exit_gates.py tests/unit/runtime/observability/test_obs_coverage_1_certification.py tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py tests/unit/docs/test_check_docs_domain_pairs.py`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the seven files plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, PDC-X and IIAC records, the `EDER label` lines (AR-24), registers whose notes need an Operator or owner fact, `HISTORICAL_STAGE_SHA_AND_EVIDENCE_REGISTRY.md` (open question W9:Q4), FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
