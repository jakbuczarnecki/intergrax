# T-IGX-050-G — ARCH-FREEZE W2 node T-AF2-MANIFESTS (draft): contract, layer, ownership and composition manifests

## Scope

Documentation only. Adds a DRAFT folder `docs/project/maintainers/qualification/arch_freeze_program/manifests/` with four generated manifests (contract, layer, ownership, composition; FRZ-FRZ-05..08), a generated summary, a README (DRAFT label, mark vocabulary, not-included list) and a byte-identical reference listing of the stdlib-only generator (as a `.txt` file under `docs/`, not under `scripts/`). Entries are marked `NOT_COMPOSED` / `NOT_CLAIMED` per plan section E (the dispositions quoted in `PDC_TO_ARCH_FREEZE_RESIDUAL_QUALIFICATION.md`) and OD-AF-18, and the neutral contracts of BND-CLOSURE are marked `PLANNED_MOVE (PT-WA)`. Generated from `origin/development` at `332c5217c390202a911ad5592e88a8f366845587`. The final manifests are generated from the G2 tree in W5 under T-IGX-052-B. Out of scope: every product path, `scripts/`, `tests/`, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, every `IIAC_*` file and script, the PDC-X record directories, a drift gate. No PASS, FRZ promotion, stage closure, freeze, IIAC closure or independent-verification claim.

## Plan digest

Plan digest: 88554bf507ca5d0c99edefd716b7c4ccd41244c99732b54485a7127f871a9fda

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-050-G.md`](plans/PLAN_T-IGX-050-G.md). The digest was computed with `scripts/governance/plan_digest.py`; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-16T18:44:20+01:00
Delegation: docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md@24b9c9e72b68de61c36523738e02677d318b1f40

## Test contract

One process at a time, under the shared lock `D:/Projekty/_wt/_locks/pytest.lock`; no pytest suite is part of this docs-only batch:

- `python -I scripts/governance/verify_arch_freeze_plan_traceability.py`
- `python -I scripts/docs/check_docs_domain_pairs.py`
- `python -I scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-050-G.md`
- Generator rerun from the committed listing against a checkout of the baseline commit; the output must equal the committed JSON and `SUMMARY.md` byte for byte.
- Reviewer check: `git diff --stat origin/development...HEAD` lists exactly the nine paths of the plan's write set.

## Out of scope

Code under `intergrax/`, `agents/`, `applications/`; `scripts/`, `tests/`, `.github/`; dependency files; IIAC scripts and records; a drift gate; non-composition gates; owners the plan does not name; the reviewed public/internal contract split; Docker, vendor integration, production configuration.

## Lifecycle state

plan_approved
