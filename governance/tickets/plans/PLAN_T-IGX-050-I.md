# PLAN T-IGX-050-I — ARCH-FREEZE W2 node T-AF2-ECAP-REVISION (draft): ECAP revision proposal for new contracts and regression classes, with FRZ recount

Status: DRAFT for Operator review. Drafted by the implementing session. Nothing here is approved, applied or audited.

## 1. Purpose

The merged plan MEP-AF2 v2.1 (Appendix Z) names the node T-AF2-ECAP-REVISION: a revision of the Enterprise Continuous Architecture Assurance Protocol (ECAP) for the new contracts and the new regression classes of the ARCH-FREEZE program, with a recount of the FRZ criteria. The final version is the W5 task T-IGX-052-D, and the lock `T-B7-LOCK-FREEZE-SHA` waits for it. This batch writes the W2 draft as a **proposal document**. It does not edit ECAP, the roadmap or the FRZ checklist.

## 2. Scope

| Item | Path | Change |
|---|---|---|
| Draft proposal | `docs/project/maintainers/qualification/arch_freeze_program/ECAP_REVISION_DRAFT.md` | new; where the ECAP text is, why the draft is a proposal, which §16 triggers apply, the contract surfaces and what ECAP must say, the regression classes mapped to ECAP wording, the FRZ recount and the M3 expectation, the gaps for T-IGX-052-D |
| Ticket and plan | `governance/tickets/T-IGX-050-I.md`, this file | new |

## 3. Method

1. Locate ECAP and its §2.C, §15 and §16 with `git grep` at `origin/development` and record `path@sha:line`.
2. Read the inputs from `origin/development`: the draft manifests and their summary, the known-red register, the FRZ acceptance matrix, the FRZ checklist, the VBN dossier draft, the plan sections named by the brief.
3. Recompute the FRZ inventory and status counts with a stdlib script (no pytest) over the checklist and ECAP text, and compare the IDs listed in ECAP §3 with the checklist rows. Counts are not copied from any document.
4. Every added factual sentence is checked against `origin/development` and recorded in `FACT_CHECK.md` under `.tmp/session/arch-freeze/ecap-rev/` (not committed).
5. The draft states plainly that it claims no PASS and no certification, and that planned surfaces are planned.

## 4. Out of scope

Any change to ECAP itself, the roadmap or the FRZ checklist; `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `.github/`; `pyproject.toml`, `uv.lock`; every `IIAC_*` file and script; the PDC-X record directories; the post-closure handoff record of T-B8-ECAP-16; the final version (T-IGX-052-D); choosing the form of the A/B split in the checklist; any FRZ promotion, certification or independent verification; merging, approval and settings.

## 5. Write set (the only paths the PR may touch)

```text
docs/project/maintainers/qualification/arch_freeze_program/ECAP_REVISION_DRAFT.md
governance/tickets/T-IGX-050-I.md
governance/tickets/plans/PLAN_T-IGX-050-I.md
```

## 6. Verification plan

No pytest suite is part of this docs-only batch. One process at a time:

- the two recount commands printed in section 8 of the draft (read-only);
- `python -I scripts/governance/verify_arch_freeze_plan_traceability.py`
- `python -I scripts/docs/check_docs_domain_pairs.py`
- `python -I scripts/governance/verify_ticket_lifecycle.py governance/tickets/T-IGX-050-I.md`
- reviewer check: `git diff --stat origin/development...HEAD` lists exactly the three paths above.

## 7. Risks

- The draft describes the tree at one commit; later commits change counts and surfaces. The W5 final recomputes them.
- Several surfaces are planned and not built (PT-WA ports, typed error contract, trust classes). The draft marks them as planned and proposes no port names.
- ECAP is a read input of the EBH-7 F10 harness. Applying the revision at the wrong time stales the generated F10 matrix; the draft lists the ordering as a gap instead of deciding it.
- The expected FRZ counts at M3 depend on an undecided checklist representation of the A/B split; the draft gives bounds and says so.
