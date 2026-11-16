# T-IGX-042-E — EQO-A remediation: publication of the Operator decision record

## Scope

Documentation-only publication of the orchestrator's transcription of the Operator's EQO-A remediation decisions of 2026-11-16 (decision record and code-wave answers, with an index), three new files under `docs/project/maintainers/qualification/eqo_a_remediation/`. Reason: finding X-8 of the independent delegate review (approvals must not rely on a gitignored record); the Operator authorised publication as a docs-only pull request. The Operator's chat is the original; the files hold only labelled English renderings. Out of scope: any change to the decisions, any approval line, every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, the PDC-X records. No PASS, FRZ promotion, stage closure, signature or independent-verification claim.

## Plan digest

Plan digest: e84ef302e0f7cab4e90be6fd440cee943babf48cbf2894ece3e17ea94765e0e1

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-042-E.md`](plans/PLAN_T-IGX-042-E.md). The digest was computed with `scripts/governance/plan_digest.py` on the draft; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line goes here after plan review, written by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time, under the shared pytest lock; results are recorded in the pull request body:

- `uv run python scripts/docs/check_docs_domain_pairs.py`
- EBH-7 / B6-F10 classification of the new paths: static read only, no qualification suite run by the implementing session (details in the plan, section 6).
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three new documents plus this ticket and its plan.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, PDC-X records, `PDC_INDEX.md`, the Enterprise Debt & Exceptions Register (batch DB-G), approval lines of any ticket, GitHub settings, Docker, vendor integration, production configuration.

## Lifecycle state

plan_pending
