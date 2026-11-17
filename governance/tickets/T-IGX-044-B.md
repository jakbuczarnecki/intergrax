# T-IGX-044-B — EQO-A remediation node PT-E01 (T-ENF-1): workflow prerequisites

## Scope

Pin the runner and the uv Python in the two ENT-AUDIT-X workflows (`.github/workflows/enterprise-audit.yml`, `.github/workflows/enterprise-audit-nightly.yml`: `runs-on: ubuntu-24.04`, `astral-sh/setup-uv@v5` with `python-version: "3.12"`) and, per Operator decision DP-65 and the stability data in the plan, remove `continue-on-error: true` from the job `governance-policy-pin` in `.github/workflows/unit-tests.yml` (one line, no other change in that file). The workflows `unit-tests.yml`, `llm-adapters-guard.yml`, `llm-network-smoke.yml` and `rag-guard.yml` were already pinned by T-IGX-040-C (`9fe9c9944`); this ticket verifies that and does not change them otherwise. Out of scope: every GitHub setting, branch protection and ruleset (the disposable trial ruleset AR-45/AR-50 is a later step on a separate authorisation), every other workflow, `scripts/`, `tests/`, `intergrax/`, `agents/`, `applications/`, the IIAC protocol and scripts. No PASS, FRZ promotion, stage closure or independent-verification claim. An independent G6 audit is required and is performed by node PT-AUD-ENF1 (implementer excluded).

## Plan digest

Plan digest: 0435eafed6f97cbe2fa7120266b45f5a4999051cb29ba62c0ae93baf1010008d

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-B.md`](plans/PLAN_T-IGX-044-B.md). The digest was computed with `scripts/governance/plan_digest.py` on the plan text; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-17T08:53:50+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@fdeec3a8abce83dc55f287b056b03dcb7e7e6fa3

## Test contract

No workflow is run by the implementing session. One process at a time for anything that runs:

- Static parse of the three workflow files and review of `git diff` (13 changed lines in total).
- `python -I scripts/governance/plan_digest.py governance/tickets/plans/PLAN_T-IGX-044-B.md` equals the digest above.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three workflows, this ticket and its plan.
- Execution evidence for the pinned runner is the CI of the pull request itself (`ent-audit-fast`, `ent-audit-pr-gate`, `governance-policy-pin`); it is read, not claimed in advance.

## Out of scope

Code, tests, scripts, GitHub settings, rulesets, branch protection, auto-merge, `harness-release.yml` and `sync-repository-metadata.yml` (still on `ubuntu-latest`, question left to the Operator), the step label of the policy-pin job, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
