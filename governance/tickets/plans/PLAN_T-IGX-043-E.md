# PLAN T-IGX-043-E — EQO-A remediation code wave, gate slice of FC-2: the production capacity maintenance gate script

Scope statement: this plan certifies nothing, promotes no FRZ criterion and states no independent verification. No test or script was run while preparing it; every fact below is "by reading" unless a line says otherwise. Plan text revised 2026-11-17 against the RW train head (see Notation).

**Classification: gate-meaning change.** The script is a gate; part 1 changes what it asserts (the capacity gate stops asserting "adapters built and probe passed" for a PRODUCT environment). Master-plan row `PT-G01` owns `scripts/maintenance/check_*` and says "gate meaning change is not delegable" (`WP-12/TICKETS_DRAFT.md`, row `PT-G01`), and the Operator's rulings treat gate-meaning changes as personal. The approval line of this ticket therefore is not written by a delegate on its own authority: it follows the Operator's decision recorded in section 0 (the wave-packet route, OD-AF-22), taken on a packet that names this ticket as a gate-meaning item. If the owner of `PT-G01` absorbs this edit, this ticket is withdrawn and its plan text moves there unchanged. At the train head the `PT-G01` entries of the RW plan (section (e) regression table) name other scripts and not this one.

Notation: `H` = `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42` (the evidence head of the earlier text of this plan); `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`; `T` = `origin/arch-freeze/rw-train` `05699b633cdb1318c215f6529e449420efc1382c` (contains `PT-WA`; `PT-WB` and `PT-FC2` are not merged); `D` = `origin/development` `2727a2c2364a97492202b931375772a400279a9b`. Citations are `path@<full 40-hex SHA>:line`. Both scripts, `production_capacity_wiring.py`, `production_capacity_governance_wiring.py`, `control_plane_composition.py`, `sandbox_wiring.py`, `runtime/sandbox/manager.py`, `scripts/gates/check_audit_ideal_gates.py` and both gate tests are `UNCHANGED` between `H`, `D` and `T` (git diff of each path is empty).

## 0. Ruling recorded

Recorded in the published transcription `docs/project/maintainers/qualification/eqo_a_remediation/OPERATOR_ANSWERS_CODE_WAVE_TRANSCRIPTION.md` (an orchestrator rendering under IIAC Amendment 01 section 1.2, not an Operator signature; the Operator may correct any row):

- Section "What message 3 covers", item 5 (chat of 2026-11-16): the Operator's personal ruling accepts both parts as drafted (part 1 capacity gate repair, part 2 sandbox gate); the sandbox part may be ruled out separately without affecting `T-IGX-043-B`. The merge order B, then A, then C, then F is confirmed (this ticket rides the same integration train as `T-IGX-043-B`).
- Addendum 4 (chat of 2026-11-17, "recommended options"): the content ruling above stands as drafted; because the ticket changes what a gate checks, its approval line is written through the train approval packet process (OD-AF-22) of the wave that carries `PT-FC2-E`, which the Operator approves once, or by a commission the Operator states separately; the recorded message does not itself commission the delegate for this ticket.
- Milestone `M1`: the same record treats `M1` as confirmed for the program (section "How the orchestrator applies message 4"; an orchestrator interpretation). This plan does not depend on `M1` beyond train start.

## 1. Purpose, the two parts and the dependency

The ticket has two independent parts. Part 1 is the content of this ticket. Part 2 is withdrawn from it and held (row 2 and the paragraph "Part 2 held" below).

| Part | Script | Relation to `T-IGX-043-B` | Why it is needed |
|---|---|---|---|
| 1 (capacity) | `scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4) | **pre-existing, not caused by B** | By reading the script is red at `T` (and at `H`): it calls `resolve_production_capacity_wiring` (`:20`) with boundary-less governance on a PRODUCT profile, which raises `ControlPlaneCompositionError` (`intergrax/applications/_shared/production_capacity_wiring.py@05699b633cdb1318c215f6529e449420efc1382c:47-55`), and `test_audit_ideal_30_4_production_capacity` pins that raise (`tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py@05699b633cdb1318c215f6529e449420efc1382c:674-692`). Not executed, so UNVERIFIED by execution. Part 1 repairs the stale gate and adds the new DP-29 assertions |
| 2 (sandbox) | `scripts/maintenance/check_sandbox_policy_wiring.py` (AUDIT-IDEAL-11.1) | **held; not caused by B** | Not edited by this ticket. By reading the script is green at `T` (it is not a base red): `wire_sandbox_sessions` returns a manager for the root-less positive case (`:27-30`; `sandbox_wiring.py@05699b633cdb1318c215f6529e449420efc1382c:12-19`). It would turn red only under a sandbox guard that refuses a root-less profile, and `T-IGX-043-B` (amendment 3) does not edit `sandbox_wiring.py`. Not executed, so UNVERIFIED by execution |

Both scripts are run by `scripts/gates/check_audit_ideal_gates.py` (`@05699b633cdb1318c215f6529e449420efc1382c:39` and `:92`). Whether that runner executes in CI is UNVERIFIED: `scripts/ci/script_paths.py:19` maps its name and a search of `.github` at `T` found no reference to it.

**Dependency, stated for the approver of both tickets:** the product part of `T-IGX-043-B` has no dependency on this ticket: it assumes neither part 1 nor part 2. `T-IGX-043-B` does not edit these scripts; this ticket edits no product path. Names used below are defined by plan section 4 of `T-IGX-043-B` (plan digest `5da771854184756977c1ed3a3effee72c0a5b21ea90fcf18d7db4558086956a7`, amendment 3 of that plan): `CapacityBackendPosture` (`PRODUCTION` default, `CI_DETERMINISTIC`) in `intergrax/runtime/capacity/production_adapters.py`; `CapacityQualification` (`UNQUALIFIED_NO_CLUSTER`, `CI_DETERMINISTIC_ONLY`) and the new keyword `backend_posture` of `resolve_production_capacity_wiring(env, *, governance=None, backend_posture=PRODUCTION)` in `production_capacity_wiring.py`. If that design changes, this plan is re-aligned and its digest recomputed.

**Part 2 held:** the sandbox script is not edited by this ticket and is outside its write set. Its future owner is the fixer ticket of the sandbox residual, the ticket that brings a sandbox guard back into the product code and decides the sandbox root and fixture the script needs; that decision is taken before the wave audit `PT-Q02`. `T-IGX-043-B` is unaffected (its contract V1..V7 does not run the scripts; its verification V9 runs the sandbox script once at the base and once at the end as an unedited control, and does not depend on this ticket). If part 1 is ruled out, the capacity script stays red exactly as it is at `H` (pre-existing), with the owner of master-plan node `PT-G01` together with the Operator.

## 2. Findings and observations

| Item | Fact at `T` | Verdict |
|---|---|---|
| Capacity gate asserts built adapters and a passed probe | `scripts/maintenance/check_production_capacity_adapters.py@05699b633cdb1318c215f6529e449420efc1382c:18-40`: `product_defaults()`, `build_production_capacity_governance(env)` (boundary None), then `resolve_production_capacity_wiring(env, governance=governance)` at `:20`, then branches on `governance.mutation_authorization_boundary is None` (`:24`) and on `wiring.adapters is None or not wiring.probe_passed` (`:36`) | VERIFIED by reading |
| Pre-existing failure (red-first on this script shows a failure that is independent of FC-2): by reading, the capacity gate already raises at `T` before it reaches those branches | the PRODUCT preset sets `production_adapters_enabled=True` (`build_product_defaults` in `presets.py@05699b633cdb1318c215f6529e449420efc1382c:258-331`; the gate test asserts the same for `product_defaults()` at `test_audit_ideal_depth_gate.py:686`) and `product_consequential_capacity_mutations_enabled(env)` is true for PRODUCT with that flag (`control_plane_composition.py@05699b633cdb1318c215f6529e449420efc1382c:26-32`), so `resolve_production_capacity_wiring` raises `ControlPlaneCompositionError` for a missing boundary (`production_capacity_wiring.py@05699b633cdb1318c215f6529e449420efc1382c:47-55`); the existing tests assert exactly that (`tests/unit/runtime/capacity/test_ecp_control_plane_governance.py@05699b633cdb1318c215f6529e449420efc1382c:446-452`, `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py@05699b633cdb1318c215f6529e449420efc1382c:674-692`) | UNVERIFIED by execution; part 1 covers repairing it |
| Sandbox gate expects a manager for a root-less sandbox profile (part 2, held) | `scripts/maintenance/check_sandbox_policy_wiring.py@05699b633cdb1318c215f6529e449420efc1382c:20-30`: `product_defaults(tool_ids=["sandbox.exec", "workspace.write_file"])`, `SandboxProfile(enable_exec_tool=True)` (no root), `wire_sandbox_sessions(env)` must not be None; `wire_sandbox_sessions` returns a manager for any non-None sandbox profile (`sandbox_wiring.py@05699b633cdb1318c215f6529e449420efc1382c:16-19`) | VERIFIED by reading; the script is green at `T`, not a base red; the script is not edited (section 1) |
| Runner | `scripts/gates/check_audit_ideal_gates.py@05699b633cdb1318c215f6529e449420efc1382c:39,92` lists both scripts; the runner then runs `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` | VERIFIED |
| Both scripts are listed in the IIAC impact matrix | `docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json@05699b633cdb1318c215f6529e449420efc1382c:10238,10251,89899,89990`, category 2, pattern `g0-enumeration-default` ("no authority keyword match", owner hint `inventory/neutral`) | listing VERIFIED; whether the edit is IIAC-material is UNVERIFIED |
| Effect of `PT-WA` on this plan | `PT-WA` (merged at `T`) edits `intergrax/applications/contracts/environment_profile/root.py` (imports made lazy, KR-18); `product_defaults`, `ApplicationEnvironmentProfile` and the `SandboxProfile` import path used by both scripts are unchanged in behaviour (diff read) | VERIFIED by reading |
| File overlap with other train tickets | none: the row of this ticket in RW-PLAN section (b) is empty; neither `PT-WB` nor `PT-FC2` lists either script | VERIFIED by reading RW-PLAN |

## 3. Design (assertion changes, the reason this is a gate-meaning change)

The script keeps the exit-code contract (0 and the `OK` line on success, 1 and a message on stderr on failure). Environment hygiene: every environment variable a check depends on (`INTERGRAX_KUBERNETES_URL`) is removed for the check that needs it and restored in a `finally` block, so a developer's own environment cannot change the result and the script leaves no change behind.

`scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4):

1. Missing boundary: expected refusal is `ControlPlaneCompositionError` with `blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"` (as the tests assert); the script catches it and passes this check, and fails (exit 1) if no refusal occurs.
2. With a supplied boundary (`ControlPlaneMutationAuthorizationBoundary(evaluator=<inline allowing evaluator>)`, passed through `build_production_capacity_governance(env, mutation_authorization_boundary=...)`; the evaluator is a small stand-in defined inline in the script: `evaluate(request)` returns `PolicyDecision(action=PolicyAction.ALLOW, reason="ok")` with `PolicyDecision` and `PolicyAction` from `intergrax.contracts.runtime_policy`; the script does not import test code, because `tests/` is not a dependency of `scripts/` and a test helper can change without the gate noticing; `_RecordingEvaluator` at `tests/unit/runtime/capacity/test_ecp_control_plane_governance.py@05699b633cdb1318c215f6529e449420efc1382c:82` is only the reference for its shape) and `INTERGRAX_KUBERNETES_URL` removed: default posture gives `wiring.enabled`, `wiring.adapters is None`, `wiring.probe_passed is False`, `wiring.qualification is CapacityQualification.UNQUALIFIED_NO_CLUSTER` (this replaces the old positive assertion).
3. Same input (URL removed) with `backend_posture=CapacityBackendPosture.CI_DETERMINISTIC`: adapters built, `wiring.probe_passed is False`, `qualification is CapacityQualification.CI_DETERMINISTIC_ONLY` (an in-memory backend is never reported as a passed probe).

`scripts/maintenance/check_sandbox_policy_wiring.py` (AUDIT-IDEAL-11.1): held, not edited (section 1).

## Negative tests

The refusal cases that the edited gate script must assert (each is a configuration the gate must see refused, taken from section 3; none is run by this plan). They are assertions inside the script, not pytest files; RW-PLAN section (d) assigns no negative-test row to this ticket, so the red-first rule of that section is not applied here and E-1 and E-3 of section 5 serve as the equivalent evidence.

1. Capacity gate, missing boundary: a PRODUCT environment with adapters enabled and no mutation-authorization boundary must raise `ControlPlaneCompositionError` with `blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"`; the script exits 1 if no refusal occurs.
2. Capacity gate, default posture with a supplied boundary and `INTERGRAX_KUBERNETES_URL` unset: adapters are not built and the probe is not reported as passed (`wiring.adapters is None`, `wiring.probe_passed is False`, `UNQUALIFIED_NO_CLUSTER`); a built adapter or a passed probe is a failure of the gate.
3. Capacity gate, `CI_DETERMINISTIC` posture: an in-memory backend is never reported as a passed probe (`wiring.probe_passed is False`, `CI_DETERMINISTIC_ONLY`).
The expected-red check E-1 (section 5) shows the unedited script failing and E-3 is the audit-only mutation check.

## 4. Exact write set

```text
scripts/maintenance/check_production_capacity_adapters.py
governance/tickets/T-IGX-043-E.md
governance/tickets/plans/PLAN_T-IGX-043-E.md
```

This write set holds one script path (class `firm`, `EXISTS`; counts 0 product, 1 firm total, 0 conditional). The sandbox script path that RW-PLAN appendix A1 block `PT-FC2-E` lists is held (section 1) and is not in it; the ticket and plan files are governance files and are not counted there. Product paths: none (`scripts/` is outside `PRODUCT_PATH_PREFIXES`; a `Ref:` trailer is still required). Forbidden: every other path, in particular `scripts/gates/check_audit_ideal_gates.py` (not edited), all `intergrax/`, `agents/` and `applications/` files, `tests/`, `.github/`, roadmap, FRZ checklist, IIAC records, the other `check_*` scripts.

## 5. Tests (script-level; no new pytest file)

| ID | What | Expected |
|---|---|---|
| E-1 (red-first) | the **unedited** capacity script run at the tree | exits non-zero (it raises, see section 2), a pre-existing failure that is independent of FC-2 and is recorded as such; it can be recorded at any train head |
| E-2 | the edited capacity script at the same tree | exits 0 and prints its `OK` line |
| E-3 (mutation, audit only) | in a scratch worktree at the audited SHA, remove the capacity default-posture guard | the edited script exits 1: the gate detects the regression it now guards |
| E-4 (regression) | `test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity` | stays green |

RW-PLAN section (e) lists gate scripts as this ticket's regression scope ("the two gate scripts run once each after PT-FC2"); E-2 is the run of the one script in this write set, and the held sandbox script is not run by this ticket.

## 6. Test commands (one process at a time under the shared lock; none run when drafted)

1. `uv run python scripts/maintenance/check_production_capacity_adapters.py`
2. `uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity -p no:cacheprovider`
3. `uv run python -m scripts.governance.igx_gov ticket`

The full runner `scripts/gates/check_audit_ideal_gates.py` (about 70 scripts and one test file) is not part of this contract; running it needs a separate Operator authorisation.

## 7. Order of work

Train position: lane 3 of the RW schedule, after `PT-WA` and `PT-WB` and after `PT-FC2` (RW-PLAN section (c): "gate scripts merge after the FC-2 fix (gate would be red before the fix)"). No other ticket of the train edits these paths.

0. Preflight: read scope = the capacity script, the `T-IGX-043-B` plan section 4, `_RecordingEvaluator`; edit scope = the capacity script; tests = section 6. Confirm that the `T-IGX-043-B` names used by part 1 (section 1) exist in the integration tree; if `PT-FC2` has not merged into the train, stop.
1. Run E-1 on the unedited script and record the output.
2. Edit the capacity script.
3. Run E-2 and E-4; ticket to `quality_gate` by the implementer; never to a PASS state.

## 8. ARCH-FREEZE impact, ACT-N11, audit, risks, rollback, effort

- ARCH-FREEZE: none of its own; RW-PLAN section (a) lists this ticket as contributing to `FRZ-PRD-03`, `FRZ-REL-07`, `FRZ-REL-09`, `FRZ-REL-10` (owned by `PT-FC2`) and as owning nothing. It keeps one existing gate meaningful after FC-2 (it is an evidence surface for an AUDIT-IDEAL item, not an FRZ row). No FRZ status is touched or claimed.
- ACT-N11: no product path, so this ticket alone does not break the IIAC activation; whether the script is IIAC-material is UNVERIFIED (the matrix lists it as category 2 by the default enumeration pattern, section 2; the `iiac_common.py` rule was not read for it) and is checked by reading before the edit; whether the gate runner runs in CI is likewise UNVERIFIED (section 1).
- Independent audit: part of the wave audit (`PT-Q02`) at the exact wave SHA by a session that did not implement or review it: replays E-1..E-4 including the mutation check E-3 and confirms the assertion changes match section 3. The Operator's ruling on the gate meaning is not replaced by the audit.
- Risks: (R-E1) the capacity script is by reading already red at `T` (section 2), so its "red-first" evidence shows a pre-existing failure; record it as such and do not attribute it to FC-2. (R-E2) The allowing evaluator is an inline stand-in written in the script (no test import); its shape must match the evaluator protocol (`evaluate(request) -> PolicyDecision`, `ControlPlaneMutationAuthorizationBoundary(evaluator=...)`), read at implementation. (R-E3) The `T-IGX-043-B` names change; digest and text are re-aligned. (R-E4) Environment variables left set by a developer would change a gate result; section 3 removes and restores them.
- Rollback: revert the script edit; the gate returns to its old assertions and to its pre-existing failure.
- Effort: S (one small script, one ruling).
