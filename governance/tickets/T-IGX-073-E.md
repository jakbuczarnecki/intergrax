# T-IGX-073-E — W2-RECOVERY-01: RW-PLAN, phase-check-3, PT-V04, Campaign B alpha

## Owner

AF-CLOSE (D-2, 2026-11-26)

## Scope

Architecture mandate **W2-RECOVERY-01** on `development` @ `8aaa178a5d45736ed51efe7c03b638ce5e253e37`:

- Canonical **T-AF2-RW-PLAN** (`RW_PLAN.md` + machine `rw_plan.v1.json`).
- Strict **phase-check-3** validator (`verify_rw_plan_phase_check3.py`).
- **PT-V04** S1 baseline runner; repair pytest collection (`pytest-asyncio` vs pytest 9).
- **Campaign B** B-1/B-2 alpha evidence runner + manifest (diagnostic, non-certifying).

Does **not** set `SCOPE`, `G2`, `RW-PLAN-AUDIT` PASS, or `REV-MASTER` formal PASS.

## Approved

Approved: W2-RECOVERY-01 architecture mandate @ 2026-11-24

## Test contract

```text
python scripts/governance/build_rw_plan.py
python scripts/governance/verify_rw_plan_phase_check3.py
python scripts/governance/verify_arch_freeze_plan_traceability.py
python -m pytest tests/unit/governance/test_verify_rw_plan_phase_check3.py --collect-only -q
python scripts/qualification/run_pt_v04_s1_baseline.py
python scripts/qualification/run_campaign_b_w2_alpha.py --phase b1
python scripts/qualification/run_campaign_b_w2_alpha.py --phase b2
```

Evidence logs: `.tmp/session/w2-recovery-03/` (W2-RECOVERY-03); `.tmp/session/w2-recovery-02/` (W2-RECOVERY-02); `.tmp/session/w2-recovery-01/` (W2-RECOVERY-01)

## W2-RECOVERY-03 corrective scope (AF-W2-RECOVERY-03)

- Six source-backed CODE_FIX ownership mappings (TYP/RPL/TEN/FRZ-10) in `build_rw_plan.py`.
- Phase-check-3 accepts node-based FRZ-FRZ-10 ownership; emits `CODE_FIX_OWNERSHIP=39/39`.
- Campaign B runner: pytest node paths, output-arg handling, historical B-2 worktree, `HEAD_DIAGNOSTIC`, coverage metadata.

## AF-W2W3-SEMANTIC-CONSOLIDATION-01 (2026-11-24)

- PT-T01 canonical ownership for FRZ-TEN-02, FRZ-TEN-07, and audit conflict A4-006 (`owner_ticket` null; no T-IGX-044-A).
- Phase-check-3: explicit six-row ownership contracts, full SCOPE hard-predecessor set, PT-WA/PT-WB lane guards.
- PT-T01 matrix registry: explicit 18-NT inventory (non-tautological); source-file collectability naming corrected.
- Mechanical validation @ HEAD: `PLAN_STRUCTURE=PASS`, `CODE_FIX_OWNERSHIP=39/39`, `RED_FIRST_EVIDENCE=INCOMPLETE`, `RW_PLAN_AUDIT=NOT_PERFORMED`.
- Evidence: `.tmp/session/af-w2w3-semantic-consolidation-01/` (diagnostic execution only; not formal qualification).

## AF-W2-CLOSURE-01 (2026-11-27)

- AF-W2-CLOSURE-01 batch (docs, bindings, generator).

## AF-W2-CLOSURE-02 (2026-11-27)

- OP-B progressive evidence validation (
erify_rw_plan_phase_check3.py, uild_rw_plan.py, unit tests).
- Campaign B diagnostic matrix 28/28; targeted R04/R24/S03 @ baseline B; HEAD diagnostic R01a/R02/R05/R20.
- Indices: RW_PLAN_AUDIT_EVIDENCE_INDEX.md, SCOPE_PREDECESSOR_INDEX.md, CAMPAIGN_B_MISSING_CASE_MAP.md.

## AF-W2-CLOSURE-03 (2026-11-27)

- Campaign B evidence integrity (R04/R24/S03 log digests; HEAD diagnostic supplements).
- OP-B progressive red-first completeness for future `RED_AT_PREFIX` transitions (12 OP-B obligations).
- Owner ticket write-set: NT-BND-06-1, NT-DEBT-04-1, NT-TYP-05-2 test paths in plans + RW generator.
- Closure qualification pack; `development` publication.


## AF-W2-CLOSURE-04 (2026-11-27)

- Evidence reference binds OP-B rows to the cited GA2R JSON file (tracked, authorized path; duplicate FRZ rejection).
- Progressive OP-B execution contract: pin_sha, exit codes, log digests, git ancestry; seeded-red record enforcement (no test-name bypass).
- Campaign B diagnostic index reconciliation; SCOPE hard-predecessor count 12 (declarative fix).
- Targeted regression pack test_verify_rw_plan_phase_check3_op_b_closure03.py; development publication.

## AF-W2-FINAL-01 (2026-11-27)

- Final OP-B evidence contract corrections (`verify_rw_plan_phase_check3.py`): pin_sha=fix_commit, test blob identity, overlay, GA2R schema, seeded-red binding, pre-G2 OP-B gate.
- Operator decision **OD-AF-CLOSE-B6** (variant A) recorded; MEP-AF2 V5.2.1; Campaign B diagnostic acceptance record; RW-PLAN audit readiness indices.
- Targeted regression extension `test_verify_rw_plan_phase_check3_op_b_closure03.py`; `development` publication.


## AF-W2-GATE-BIND-01 (2026-11-27)

- Mandatory pre-G2 settlement: `verify_rw_plan_phase_check3.py --pre-g2` (`PRE_G2_OP_B_SETTLEMENT`) independent of plan `pre_g2_op_b_gate_enforce`.
- G2 qualification binding in `rw_plan_bindings.v1.json` `g2_entry_conditions` (MANDATORY_QUALIFICATION_COMMAND).
- MEP-AF2 V5.2.1 B6 wording; SCOPE predecessor index; audit prep dossier; `SOURCES.sha256` refresh; targeted regression `test_verify_rw_plan_phase_check3_pre_g2_gate.py`.

## AF-W2-SCOPE-CLOSE-01 (2026-11-27)

- Publish independent RW-PLAN-AUDIT round 2 PASS (AF-W2-INDEPENDENT-AUDIT-02 @ `73835a889fcebe4393113abd25d9cf9aeceda7c1`, AR-12-07).
- Settle PT-B01/PT-B02 as diagnostic `PREDECESSOR_CONTRACT_SATISFIED`; formal SCOPE milestone closure (`SCOPE_FORMAL_CLOSEOUT_RECORD.md`); 12/12 hard predecessors.
- **SCOPE=CLOSED**; **W2 scope milestone complete**; **W3_ENTRY=AUTHORIZED**; OP-B / red-first obligations preserved (`OP_B_PENDING=12`, `RED_FIRST_EVIDENCE=INCOMPLETE`, `G2=OPEN`).

## AF-W2W3-RESIDUAL-CLOSURE-01 (2026-11-24)

- Test environment: isolated writable UV env (`.tmp/session/af-w2w3-residual-closure-01/.venv`); main `.venv` tiktoken install ACL-denied; `uv sync --extra dev --extra dev-unit-cert --frozen` recovered FastMCP.
- FC-1: `test_reject_and_escalate_verdicts_are_recorded_through_materialization` injects `LoggingNotificationAdapter` on `HostOrchestrationLoopInitSpec`.
- Phase-check-3: strict matrix ownership (no arbitrary ticket fallback); full explicit metadata including `owner_ticket_family`; malformed SCOPE dependency entries rejected without `TypeError`.
- Regression: governance 297 passed; mechanical `CODE_FIX_OWNERSHIP=39/39`; FC-1 10 passed; TENANT+SUPPLY+MCP identity 98 passed; ruff pass on write-set.
- Evidence: `.tmp/session/af-w2w3-residual-closure-01/` (session artifacts only).

## Lifecycle state

in_progress — AF-W2-SCOPE-CLOSE-01 published independent RW-PLAN-AUDIT PASS and SCOPE formal closure; OP-B product evidence and G2 remain open on downstream tickets.
