# T-IGX-042-D — EQO-A remediation batch DB-G: Enterprise Debt & Exceptions Register structure, qualification index, documentation map link

## Scope

Documentation-only implementation of the structure part of Operator decision OD-REM-04 / condition C4 (EQO-A Campaign A remediation, baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589): the governing document and row schema of the Enterprise Debt & Exceptions Register, an index of `docs/project/maintainers/qualification/`, and three routing rows plus one index pointer row in `docs/project/technical/DOCUMENTATION_MAP.md` (6 hunks, 4 files). Findings: EQOA-A0-001, -002, -003, -004, -006, EQOA-A2-006, EQOA-A4-028, -033, -046, EQOA-A7-034, -035, -037. Out of scope: register data rows and the owner vocabulary (the Operator confirms the vocabulary once, DP-47), the consistency script and tests, every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, other `governance/` files, the three superseding-manifest names (needs an Operator or owner fact) and any position on a standalone ECAP executable (DP-54 is open). No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 87fc5aab2eda91072ccc091ed5cc8be1c6ac6151f50992f045c1f4e4231f8976

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-D.md`](plans/PLAN_T-IGX-042-D.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/README.md
docs/project/maintainers/qualification/enterprise_debt_register/README.md
docs/project/maintainers/qualification/enterprise_debt_register/register_schema.json
docs/project/technical/DOCUMENTATION_MAP.md
```

## Approved

PENDING. <<OPERATOR: the approval line goes here after plan review, written by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).>>

## Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/ -p no:cacheprovider`
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Register checker `check_register.py` (standard library, read-only, kept outside the repository), run once against the draft data; informative only because no data rows are committed.
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the four Write set paths plus this ticket and its plan.

## Out of scope

Data ingest (`register.jsonl`, `reconciliation.json`, owner vocabulary), the consistency script and CI step, owner assignments, severity ratings, the PDC-P9-01 reference amendment of DP-52, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

`plan_pending`
