# T-IGX-050-I — ARCH-FREEZE W2 node T-AF2-ECAP-REVISION (draft): ECAP revision proposal for new contracts and regression classes, with FRZ recount

## Scope

Documentation only. Adds one proposal document, `docs/project/maintainers/qualification/arch_freeze_program/ECAP_REVISION_DRAFT.md`, for node T-AF2-ECAP-REVISION of the merged ARCH-FREEZE plan (MEP-AF2 v2.1): where the ECAP text is, which ECAP §16 update triggers apply, the new and changed contract surfaces and what ECAP must say about each, the known-red classes mapped to ECAP wording, a recount of the FRZ criteria with the target shape at M3 (split rows PRD-03/08 and SEC-04/05/06, Path B rows), and the gaps for the W5 final (T-IGX-052-D). It does not edit ECAP, the roadmap or the FRZ checklist. It claims no PASS, no certification and no independent verification. Facts were read at `origin/development` `a7f02361540752426f0015b628f8a40b6cbdbb2a`.

## Plan digest

Plan digest: 5f03699b87e3d87fe46455f527b04281f88914453bcabf425d33d7258bee832f

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-I.md`](plans/PLAN_T-IGX-050-I.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T21:11:14+01:00
Delegation: docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md@24b9c9e72b68de61c36523738e02677d318b1f40

## Test contract

No pytest suite is part of this docs-only batch. One process at a time:

- the two read-only recount commands in section 8 of the draft;
- `python -I scripts/governance/verify_arch_freeze_plan_traceability.py`
- `python -I scripts/docs/check_docs_domain_pairs.py`
- `python -I scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-050-I.md`
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths of the plan's write set.

## Out of scope

Editing ECAP, the roadmap or the FRZ checklist; code under `intergrax/`, `agents/`, `applications/`; `scripts/`, `tests/`, `.github/`; dependency files; IIAC scripts and records; the T-B8-ECAP-16 handoff record; the W5 final version; deciding how the A/B split is written into the checklist; Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
