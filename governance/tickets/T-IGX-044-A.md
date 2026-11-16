# T-IGX-044-A - EQO-A remediation node PT-E00 (T-ENF-0): IIAC draft-manifest point-in-time record, ECAP open-question index entry, FRZ-AUD-10 live-enforcement amendment draft

## Scope

Documentation-only. Findings EQOA-A2-006 (three of 40 IIAC manifests fail the repository validators) and EQOA-A0-006 (no standalone ECAP executable at the baseline; whether one is required is open, DP-54) of the EQO-A Campaign A dossier (baseline `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`, PR #589), plus a clearly labelled DRAFT of the FRZ-AUD-10 live-enforcement amendment (Operator decisions OD-REM-07, C9, DP-64). Two new records next to the ENT-AUDIT-X records and two pointer edits in the qualification index and the debt register README (write set in the plan). No manifest is edited, moved or deleted. No PASS, closure, FRZ promotion, stage status or independent-verification claim; the FRZ-AUD-10 disposition, the freeze checklist and the roadmap are unchanged.

## Plan digest

Plan digest: f1233e47fa7fe8e07784fe544a9ec6bbbe70ccab427623fc9e5857eea5d5f544

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-044-A.md`](plans/PLAN_T-IGX-044-A.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Write set

```text
docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_IIAC_MANIFEST_DRAFT_POINT_IN_TIME_RECORD.md
docs/project/maintainers/qualification/ent_audit_x/FRZ_AUD_10_LIVE_ENFORCEMENT_AMENDMENT_DRAFT.md
docs/project/maintainers/qualification/README.md
docs/project/maintainers/qualification/enterprise_debt_register/README.md
```

## Approved

PENDING. The approval line goes here after plan review, written by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time, under the shared pytest lock:

- `uv run pytest tests/unit/docs/ -o cache_dir=<scratch dir outside the repository>` (`-p no:cacheprovider` is not usable here: the repository sets the `cache_dir` ini option, which then fails with a usage error)
- `uv run python scripts/docs/check_docs_domain_pairs.py`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the four documents above plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, GitHub rulesets and settings, validator or allowlist changes, the roadmap, the FRZ checklist (both Operator-only, changed in the later PT-D09/PT-D10 batch), the ECAP protocol, any manifest, evidence index or `IIAC_*` file, the IIAC protocol and activation pin, EBH-7 F10 canonical inputs and digest-pinned artefacts, FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

`plan_pending`
