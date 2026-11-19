# T-IGX-043-E — EQO-A remediation code wave, gate slice of FC-2: production capacity maintenance gate script

## Scope

Gate-meaning change, split out of `T-IGX-043-B` (FC-2) after the independent delegate review. Master-plan node `PT-FC2-E`, lane 3 of the RW train, merging after `PT-FC2`. It edits one maintenance gate script, `scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4), so that it asserts the fail-closed behaviour that the FC-2 product part introduces: capacity wiring without a cluster is reported unqualified and an in-memory backend never yields a passed probe (DP-29). The script is run by `scripts/gates/check_audit_ideal_gates.py`. Plan evidence was read at `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42` and re-checked at the RW train head `05699b633cdb1318c215f6529e449420efc1382c` (contains `PT-WA`): both scripts and every product file the plan cites are unchanged. By reading, the capacity script is already red independently of FC-2 (it calls the wiring with a boundary-less governance on a PRODUCT profile, which raises, and an existing test pins that raise); part 1 covers repairing it. Write set (`RW_PLAN.md` appendix A1 block `PT-FC2-E` lists the two scripts, firm, counts 0 product / 2 firm / 0 conditional; this ticket holds the sandbox script out, so its write set has counts 0 product / 1 firm / 0 conditional, see `.tmp/session/arch-freeze/w3-packet-2/043-E/INPUT_043-E_amend1.md`): the capacity script and the ticket and plan files; no product path, no test file.

**Authority:** this is a gate-meaning change. Master-plan node `PT-G01` owns `scripts/maintenance/check_*` and states that a gate meaning change is not delegable; the approval line of this ticket follows the Operator's decision recorded in the published transcription (item 5 of "What message 3 covers" and addendum 4, OD-AF-22 wave-packet route) on a wave packet that names this ticket as a gate-meaning item. If the owner of `PT-G01` absorbs this edit, this ticket is withdrawn.

**Ruling recorded:** `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` (an orchestrator transcription, not an Operator signature): the Operator's personal ruling accepts both parts as drafted and allows the sandbox part to be ruled out separately without affecting `T-IGX-043-B`; the merge order B, then A, then C, then F is confirmed. Addendum 4 (2026-11-17) keeps that ruling and routes the approval line through the wave packet. Milestone `M1` is treated as confirmed for the program by the same record.

**Part 1 in, part 2 held:** part 1 repairs and strengthens the capacity gate, which is red at the current head independently of FC-2 (by reading; not executed). Part 2, the sandbox gate script `scripts/maintenance/check_sandbox_policy_wiring.py`, is withdrawn from this ticket and held: by reading it is green at the train head (not a base red) and turns red only under a sandbox guard that `T-IGX-043-B` (amendment 3) does not add. Its future owner is the fixer ticket of the sandbox residual, decided before the wave audit `PT-Q02` (plan section 1). **Dependency:** the product part of `T-IGX-043-B` has no dependency on this ticket. This ticket uses names defined by `T-IGX-043-B` (`CapacityBackendPosture`, `CapacityQualification`) and is re-aligned if they change. Whether the two scripts are IIAC-material (the impact matrix lists them as category 2 by the default enumeration pattern) and whether `check_audit_ideal_gates.py` runs in CI (no `.github` reference found) are UNVERIFIED. Nothing in this ticket claims closure, PASS, FRZ promotion or independent verification.

## Plan digest

Plan digest: 5c1a3d11cf7da01b3b5ff27d2e444ee74ad1b61b2bda7a2e8eba06ec43258acd

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-043-E.md`](plans/PLAN_T-IGX-043-E.md). The digest was computed with `scripts/governance/plan_digest.py` on the plan text; it must be recomputed after any edit of the plan.

## Approved

Approved: operator-delegate @ 2026-11-18T09:48:29+01:00
Delegation: docs/project/maintainers/qualification/arch_freeze_program/RW_TRAIN_STANDING_DELEGATION_RECORD.md@881428fb231f34215a67364aa5788a4f2f986e83
Delegate session: local_64ab068c-841e-4021-9874-5d5ba65f95ff

## Test contract

One process at a time under the shared test lock; none run when this ticket was drafted:

- `uv run python scripts/maintenance/check_production_capacity_adapters.py` (first on the unedited script for the red-first record, then edited)
- `uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity -p no:cacheprovider`
- `uv run python -m scripts.governance.igx_gov ticket` before push
- Reviewer check: `git diff --stat` lists exactly the capacity script plus this ticket and its plan.
- The full runner `scripts/gates/check_audit_ideal_gates.py` is not part of this contract (broad); it needs a separate Operator authorisation. The independent mutation check (plan section 5, E-3) belongs to the wave audit at the exact wave SHA by a session that did not implement or review it.

## Out of scope

Every product path (`intergrax/`, `agents/`, `applications/`), every test file, `scripts/gates/check_audit_ideal_gates.py`, all other `check_*` scripts, `.github/`, `pyproject.toml`, `uv.lock`, the roadmap, the FRZ checklist, IIAC records, GitHub settings, production configuration, Docker, FRZ promotion, and the product guards themselves (`T-IGX-043-B`).

## Lifecycle state

plan_approved

## Implementer session

Implementer session: local_76769e42-cc2e-41d7-9cd1-274e79847690
