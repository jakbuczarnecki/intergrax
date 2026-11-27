# AF-CLOSE OP-B: record the Operator decisions of 2026-11-27 and the FRZ-FRZ-10 ownership

Ticket: T-IGX-078-F. Lane kind: DOC (bookkeeping). No product path, no test file, no script is edited.

## Context

The Operator answered the OP-B packet in chat on 2026-11-27 with "Accept all (Recommended)". The packet holds five items (B-1..B-5) taken from `RED_FIRST_GA2R_OPERATOR_ITEMS.md` and the K1-F ENV and ECF notes. The decisions are not yet written into the repository. Independent of the Operator, the orchestrator allocates the ownership ticket for FRZ-FRZ-10 (T-IGX-052-I) so that the RW plan names a ticket that exists.

## Read scope

- `docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md`
- `docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_V5_RED_CLASSES_DECISION.md`
- `docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_OPERATOR_ITEMS.md`
- `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json`, `scripts/governance/build_rw_plan.py` (read only)
- `governance/tickets/T-IGX-078-B.md`, `T-IGX-078-C.md` (models)

## Write set (exact)

- `docs/project/maintainers/qualification/ARCH_FREEZE_OPERATOR_DECISION_REGISTER.md` (rows OD-AF-CLOSE-B1..B5 and one signature-record section)
- `docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_V5_RED_CLASSES_DECISION.md` (append-only dated entry)
- `docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_OPERATOR_ITEMS.md` (status line and decided option per item)
- `governance/tickets/T-IGX-052-I.md` (new, plan_pending, no Approved line)
- `docs/project/maintainers/qualification/arch_freeze_program/rw_plan_bindings.v1.json` (FRZ-FRZ-10 owner and the ticket entry for T-IGX-052-I)
- `docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json`, `RW_PLAN.md` (generated only, no hand edits)
- `docs/project/maintainers/qualification/arch_freeze_program/SOURCES.sha256` (only if the repository convention requires it)
- `governance/tickets/T-IGX-078-F.md`, `governance/tickets/plans/PLAN_T-IGX-078-F.md`

Not written: `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `pyproject.toml`, `uv.lock`, `.github/`, the two EBH-6 Q2-D2 test files, the 2026-11-26 text of the RED-CLASSES decision record, any other plan file.

## Approach

1. Register: one row per decision B1..B5 in the style of OD-AF-CLOSE-D1, English rendering, date 2026-11-27, status DECIDED. B1 states that the confirmation is later than the application of the dispositions and covers the checker change in `a61eeaedf`. B3 is a record only; the test files are not edited here. B5 records E-14, E-15 and E-16 as duplicates and E-17 as dropped unless restated.
2. RED-CLASSES record: append a dated 2026-11-27 entry for B-2 (twelve rows, owner tickets, no evidence claimed, `RED_FIRST_EVIDENCE` stays INCOMPLETE, FRZ-DEBT-02 and -03 stay EVIDENCE_MISSING until a seeded red case is shown) and a pointer to the B1 register row. Existing text is unchanged.
3. GA.2R items file: status line becomes decided 2026-11-27 with pointers; each item gets the decided option appended.
4. Ownership: T-IGX-052-I is a plan_pending stub for FRZ-FRZ-10 (T-B7-PFP-ACTIVATE held-path guard). It grants no write authority.
5. Binding: set the FRZ-FRZ-10 owner ticket to T-IGX-052-I in the bindings file, add the ticket entry, regenerate with `scripts/governance/build_rw_plan.py`. If the generator needs a code change, item 5 stops and is reported BLOCKED.

## Acceptance and verification (one command per step)

- `uv run pytest tests/unit/governance/test_build_rw_plan.py tests/unit/governance/test_verify_rw_plan_phase_check3.py -q`
- `uv run python scripts/governance/verify_rw_plan_phase_check3.py` (result recorded, not repaired by editing records)
- `uv run python -m scripts.governance.igx_gov ticket`

## Risks

The register rows quote the Operator's button label only; the packet text is an untracked session record and is cited as such.
