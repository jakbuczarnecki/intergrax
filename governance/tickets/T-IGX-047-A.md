# T-IGX-047-A — EQO-A remediation node PT-R01 (EDER batch B): register data load, reconciliation, exclusions, proposed owner vocabulary

## Scope

Documentation-only data load of the Enterprise Debt & Exceptions Register whose structure was merged by T-IGX-042-D (PR #607): 1069 rows at `INGESTED` (239 EQO-A findings, 605 OPEN/UNKNOWN and 26 DEFERRED dossier entries, 155 accepted A2/register entries as EXCEPTION rows, 39 IIAC per-manifest pointer rows, 4 skip category rows and 1 N/A category row), the reconciliation (605 + 155 = 760, and the exact partition of the 1480 dossier entries), the exclusions list of the 155 non-entries, and the proposed owner vocabulary (55 IDs, unconfirmed, DP-47). Baseline of the dossier: `258bbf6abb6fdbb92ef0185d058527fbb1e47f04` (PR #589). Decisions applied: DP-02, DP-46..DP-53. Out of scope: owner confirmation, severity ratings, impact confirmation, triage, any lifecycle state beyond INGESTED, the consistency script and CI step (PT-R02), every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, other `governance/` files, and any position on a standalone ECAP executable (DP-54 is open). No PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 6021649da78e68b34dec52e4e603b10b4929b8a36da7bcbd56cde8aabd9ca680

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-047-A.md`](plans/PLAN_T-IGX-047-A.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/enterprise_debt_register/register.jsonl
docs/project/maintainers/qualification/enterprise_debt_register/reconciliation.json
docs/project/maintainers/qualification/enterprise_debt_register/exclusions.jsonl
docs/project/maintainers/qualification/enterprise_debt_register/owner_vocabulary.proposed.json
```

## Approved

Approved: operator-delegate @ 2026-11-16T17:45:11+01:00
Delegation: docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md@743087d57c674100dc9119959dc875271f70e72c

## Test contract

One process at a time, under the shared pytest lock:

- Register checker `check_register.py` (standard library, read-only, kept outside the repository) over the committed data; result reported as is.
- `uv run pytest tests/unit/docs/` (without `-p no:cacheprovider`)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: `git diff --name-only origin/development...HEAD` lists exactly the four Write set paths plus this ticket and its plan.

## Out of scope

Owner confirmation (DP-47), severity ratings, triage, the consistency script and CI step, the PDC-P9-01 reference amendment of DP-52, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
