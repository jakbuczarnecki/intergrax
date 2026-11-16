# T-IGX-046-A — EQO-A remediation node PT-D03 (batch DB-J): byte-preserving publication of locally held audit artefacts

## Scope

Documentation-only addition of new files under `docs/project/maintainers/qualification/iiac_evidence/`: byte-preserving copies of locally held (gitignored) audit reports, findings, logs and signing records that tracked records cite or rely on (EQO-A findings A2-026 and A7-026), each with sha256, a secret/PII scan result and a custody statement in `iiac_evidence/EQO-A-PT-D03/EVIDENCE_MANIFEST.md`. Decisions applied: DP-07, DP-44; authority AR-20. Files with a scan hit are not published and are listed by path, line and kind only. Out of scope: any edit of an existing tracked file, every product path, `scripts/`, `tests/`, `.github/`, the roadmap, the FRZ checklist, the Operator re-affirmation of the delegation, settling G-SKILL. Publication is not certification: no PASS, FRZ promotion, stage closure, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: e6e9b2624c0992a3c9460e25feccd017e14610103c72843ac5c6d498c45b4325

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-046-A.md`](plans/PLAN_T-IGX-046-A.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

PENDING. The approval line is written after plan review by the Operator or a commissioned session; the implementer never writes it. This placeholder deliberately avoids the approval keyword because the ticket validators accept any line that starts with it (finding EQOA-A6-002).

## Test contract

One process at a time under the shared pytest lock, documentation checks only:

- `uv run python scripts/docs/check_docs_domain_pairs.py`
- `uv run pytest tests/unit/docs/ -p no:cacheprovider`
- Reviewer check: `git diff --stat origin/development...HEAD` lists only new files under `iiac_evidence/` plus this ticket and its plan; after the commit the git blob of each published file hashes to the sha256 in the manifest.

## Out of scope

Code, tests, scripts, workflows, roadmap, FRZ checklist, existing qualification records (including the citing record `CERT_CAMPAIGN_DELEGATED_DECISIONS_2026-11-07.md`), the optional ARCH-FREEZE G6 wave-1 report (finding A2-015), FRZ promotion, Docker, vendor integration, production configuration.

## Lifecycle state

`plan_pending`
