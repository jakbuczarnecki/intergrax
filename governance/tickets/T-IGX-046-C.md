# T-IGX-046-C — EQO-A remediation node PT-D12: seam canon documents

## Scope

Documentation-only dated notes in five canon and owner documents (`docs/project/architecture/` and `docs/project/technical/guides/`), recording the Operator decisions DP-13 (lower-layer import canon kept, no allowlist for the twelve `intergrax/runtime` modules, fixed by inversion in the code wave), DP-16 (canonical host LLM resolver), DP-18 (closed named list of provider composition roots, the rule only), DP-19 (direct `uvicorn.run` entrypoints declared non-production, production through `run_hosted_application`, default bind 127.0.0.1) and the scope relabel of the CTRL-X, GOV-X2 and HARNESS-FINAL claims about the registry-bypass gate (finding EQOA-A3-016). Baseline of the findings: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589); facts re-checked at `origin/development` `2ab57e36d`. The notes describe the target state decided by the Operator and state that the code wave implements it. Out of scope: every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, qualification records and `governance/` except this ticket and its plan. No PASS, FRZ promotion, stage closure, IIAC closure, re-certification or independent-verification claim.

## Plan digest

Plan digest: d1711f3d34f318bccabe09ba3983c9c3dc59446550547b4baaead3b0856af3ba

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-C.md`](plans/PLAN_T-IGX-046-C.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T16:42:12+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@0685852d44a64f8c2f523136515df25cb63bd0ca

## Test contract

One process at a time, under the shared pytest lock; results are recorded in the pull request body:

- `uv run pytest tests/unit/runtime/architecture/test_gov_final_1_documentation_regression_gates.py tests/unit/runtime/architecture/test_gov_final_3_*.py tests/unit/runtime/architecture/test_gov_final_4_*.py tests/unit/docs/ -o cache_dir=<scratch directory outside the repository>` (the ticket-A form with `-p no:cacheprovider` fails under `--strict-config`).
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the five documents plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, qualification records, the decisions DP-14, DP-15, DP-17, DP-20, DP-21 and DP-22 (other nodes), the named list of provider composition roots (needs an Operator list), FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
