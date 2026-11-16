# T-IGX-042-D — EQO-A remediation batch DB-G: Enterprise Debt & Exceptions Register structure, qualification index, documentation map link

## Scope

Documentation-only implementation of the structure part of Operator decision OD-REM-04 / condition C4 (EQO-A Campaign A remediation, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589): the governing document and row schema of the Enterprise Debt & Exceptions Register, an index of `docs/project/maintainers/qualification/`, and three routing rows plus one index pointer row in `docs/project/technical/DOCUMENTATION_MAP.md` (6 hunks, 4 files). Findings: EQOA-A0-001, -002, -003, -004, -006, EQOA-A2-006, EQOA-A4-028, -033, -046, EQOA-A7-034, -035, -037. Out of scope: register data rows and the owner vocabulary (the Operator confirms the vocabulary once, DP-47), the consistency script and tests, every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, other `governance/` files, the three superseding-manifest names (needs an Operator or owner fact) and any position on a standalone ECAP executable (DP-54 is open). No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 5fbd963a4f2a2da5660e6a7956884df2b0e555c0f418c1f6f10995346b9b7027

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-D.md`](plans/PLAN_T-IGX-042-D.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/README.md
docs/project/maintainers/qualification/enterprise_debt_register/README.md
docs/project/maintainers/qualification/enterprise_debt_register/register_schema.json
docs/project/technical/DOCUMENTATION_MAP.md
```

## Approved

Approved: operator-delegate @ 2026-11-16T13:44:11+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@c98a0d7d67a0b52b8e7e0e279baa0d906bb9c45d

## Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`: that flag conflicts with the repo pytest temp plugin, which sets `cache_dir`, and the run aborts with exit 4)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Register checker `check_register.py` (standard library, read-only, kept outside the repository), run once against the draft data; informative only because no data rows are committed.
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the four Write set paths plus this ticket and its plan.

## Out of scope

Data ingest (`register.jsonl`, `reconciliation.json`, owner vocabulary), the consistency script and CI step, owner assignments, severity ratings, the PDC-P9-01 reference amendment of DP-52, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
