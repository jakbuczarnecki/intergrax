# PLAN T-IGX-043-E — EQO-A remediation code wave, gate slice of FC-2: two maintenance gate scripts that the FC-2 product guards invalidate

Status: DRAFT. Drafted 2026-11-16 after the independent delegate review of `T-IGX-043-B` required the gate edits to be split out. Nothing here is approved, implemented, executed or audited; no test or script was run while drafting. This plan certifies nothing, promotes no FRZ criterion and states no independent verification.

**Classification: gate-meaning change.** Both scripts are gates; the two parts change what they assert (the capacity gate stops asserting "adapters built and probe passed" for a PRODUCT environment, the sandbox gate stops accepting a root-less sandbox profile for side-effect tools). Master-plan row `PT-G01` owns `scripts/maintenance/check_*` and says "gate meaning change is not delegable" (`WP-12/TICKETS_DRAFT.md`, row `PT-G01`), and the master plan's authorisation list treats gate-meaning changes as personal to the Operator. Therefore: this ticket needs the Operator's personal ruling and is **not approvable by a delegate or a commissioned session**. If the owner of `PT-G01` prefers to absorb these two edits, this ticket is withdrawn and its plan text moves there unchanged.

Notation: `H` = `origin/development` `7960ae7f6c82da60ca9acdc0ceebf63810002a42`; `B` = `258bbf6abb6fdbb92ef0185d058527fbb1e47f04`. Citations are `path@<full 40-hex SHA>:line`. Both scripts are `UNCHANGED` between `B` and `H`.

## 0. Ruling recorded

Recorded from the Operator's chat answers as transcribed by the orchestrating session on 2026-11-16 in `.tmp/session/eqo-a-remediation/FC/OPERATOR_ANSWERS_TRANSCRIPTION.md` (a session artefact under `.tmp/`, a draft transcription that is not yet a repository record and not an Operator signature): the Operator's personal ruling on this ticket accepts both parts as drafted (part 1 capacity gate repair, part 2 sandbox gate); the sandbox part may be ruled out separately without affecting `T-IGX-043-B`; the merge order B, then A, then C, then F is confirmed (this ticket rides the wave train with `T-IGX-043-B`). The ruling is on the content; the approval line of this ticket remains the Operator's own and is not written here or by any delegate. Still open: milestone `M1` (not confirmed by the transcription) and that approval line.

## 1. Purpose, the two parts and the dependency

The ticket has two independent parts. The Operator can rule on them separately.

| Part | Script | Relation to `T-IGX-043-B` | Why it is needed |
|---|---|---|---|
| 1 (capacity) | `scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4) | **pre-existing, not caused by B** | By reading the script is red at `H`: it calls `resolve_production_capacity_wiring` (`:20`) with boundary-less governance on a PRODUCT profile, which raises `ControlPlaneCompositionError` (`intergrax/applications/_shared/production_capacity_wiring.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:47-55`), and `test_audit_ideal_30_4_production_capacity` pins that raise (`tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:674-692`). Not executed, so UNVERIFIED by execution. Part 1 repairs the stale gate and adds the new DP-29 assertions |
| 2 (sandbox) | `scripts/maintenance/check_sandbox_policy_wiring.py` (AUDIT-IDEAL-11.1) | **caused by B** | The root-less positive case (`:27-30`) raises `SandboxCompositionError` once the FC-2 sandbox guard exists |

Both scripts are run by `scripts/gates/check_audit_ideal_gates.py` (`@7960ae7f6c82da60ca9acdc0ceebf63810002a42:39` and `:92`). Whether that runner executes in CI is UNVERIFIED (`scripts/ci/script_paths.py` references it; no workflow reference was found).

**Dependency, stated for the approver of both tickets:** the product part of `T-IGX-043-B` assumes that **part 2** of this ticket lands in the same wave (same integration train, `RW`). It has no dependency on part 1. `T-IGX-043-B` does not edit these scripts; this ticket edits no product path. Names used below (`CapacityBackendPosture.CI_DETERMINISTIC`, `CapacityQualification.UNQUALIFIED_NO_CLUSTER` and `CI_DETERMINISTIC_ONLY`, `SandboxCompositionError`) are defined by plan section 4 of `T-IGX-043-B`; if that design changes, this plan is re-aligned and its digest recomputed.

**Fallback if the Operator rules part 2 out (or it is delayed):** `T-IGX-043-B` stays approvable (its contract V1..V7 does not run the scripts); the sandbox script stays red in the gate runner at the wave head; the owner of that red gate is the owner of master-plan node `PT-G01` together with the Operator; the wave audit lists it as a known residual; no text of `T-IGX-043-B` claims that gate green. If only part 1 is ruled out, the capacity script stays red exactly as it is at `H` (pre-existing), with the same owner. Ruling both out leaves the gate runner red for the two reasons above.

## 2. Findings and observations

| Item | Fact at `H` | Verdict |
|---|---|---|
| Capacity gate asserts built adapters and a passed probe | `scripts/maintenance/check_production_capacity_adapters.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:18-40`: `product_defaults()`, `build_production_capacity_governance(env)` (boundary None), then `resolve_production_capacity_wiring(env, governance=governance)` at `:20`, then branches on `governance.mutation_authorization_boundary is None` (`:24`) and on `wiring.adapters is None or not wiring.probe_passed` (`:36`) | VERIFIED by reading |
| Pre-existing failure (label kept: red-first on this script shows a failure that is independent of FC-2): by reading, the capacity gate already raises at `H` before it reaches those branches | `resolve_production_capacity_wiring` raises `ControlPlaneCompositionError` for a PRODUCT environment with adapters enabled and no boundary (`intergrax/applications/_shared/production_capacity_wiring.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:47-55`); the existing tests assert exactly that (`tests/unit/runtime/capacity/test_ecp_control_plane_governance.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:446-452`, `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:674-692`) | UNVERIFIED by execution; part 1 covers repairing it |
| Sandbox gate expects a manager for a root-less sandbox profile | `scripts/maintenance/check_sandbox_policy_wiring.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:20-30`: `product_defaults(tool_ids=["sandbox.exec", "workspace.write_file"])`, `SandboxProfile(enable_exec_tool=True)` (no root), `wire_sandbox_sessions(env)` must not be None | VERIFIED; after the FC-2 guard this call raises `SandboxCompositionError` |
| Runner | `scripts/gates/check_audit_ideal_gates.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:39,92` lists both scripts; the runner then runs `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` | VERIFIED |
| Both scripts are listed in the IIAC impact matrix | `docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json@7960ae7f6c82da60ca9acdc0ceebf63810002a42:10238,10251,89899,89990` | listing VERIFIED; whether that makes them IIAC-material is UNVERIFIED |

## 3. Design (assertion changes, the reason this is a gate-meaning change)

`scripts/maintenance/check_production_capacity_adapters.py` (AUDIT-IDEAL-30.4):

1. Missing boundary: expected refusal is `ControlPlaneCompositionError` with `blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"` (as the tests assert); the script catches it and exits 0 for that case, 1 if no refusal occurs.
2. With a supplied boundary (an allowing evaluator defined inline in the script as a small stand-in that satisfies the evaluator protocol used by `ControlPlaneMutationAuthorizationBoundary`; the script does not import test code, because `tests/` is not a dependency of `scripts/` and a test helper can change without the gate noticing; the protocol shape is read at implementation, `_RecordingEvaluator` at `tests/unit/runtime/capacity/test_ecp_control_plane_governance.py@7960ae7f6c82da60ca9acdc0ceebf63810002a42:82` is only the reference for its shape) and `INTERGRAX_KUBERNETES_URL` unset: default posture gives `wiring.enabled`, `wiring.adapters is None`, `wiring.probe_passed is False`, `wiring.qualification is CapacityQualification.UNQUALIFIED_NO_CLUSTER` (this replaces the old positive assertion).
3. Same input with `backend_posture=CapacityBackendPosture.CI_DETERMINISTIC`: adapters built, `wiring.probe_passed is False`, `qualification is CI_DETERMINISTIC_ONLY` (an in-memory backend is never reported as a passed probe).

`scripts/maintenance/check_sandbox_policy_wiring.py` (AUDIT-IDEAL-11.1):

1. Keep the `product_requires_sandbox` true/false assertions (`:23-25`, `:37-40`) and `tool_profile_with_sandbox` (`:32-35`).
2. Positive case: `SandboxProfile(enable_exec_tool=True, root=<tempfile.TemporaryDirectory path>)` returns a manager whose `root` equals that path (replaces the root-less positive case).
3. Negative cases: root None with `INTERGRAX_SANDBOX_ROOT` removed from the environment raises `SandboxCompositionError`; `sandbox=None` with side-effect tools raises it; a PRODUCT environment without side-effect tools still returns None or a manager without raising.

## Negative tests

The refusal cases that the edited gate scripts must assert (each is a configuration the gates must see refused, taken from section 3; none is run by this plan):

1. Capacity gate, missing boundary: a PRODUCT environment with adapters enabled and no mutation-authorization boundary must raise `ControlPlaneCompositionError` with `blocker_code == "ECP_BLOCKED_MISSING_BOUNDARY"`; the script exits 1 if no refusal occurs.
2. Capacity gate, default posture with a supplied boundary and `INTERGRAX_KUBERNETES_URL` unset: adapters are not built and the probe is not reported as passed (`wiring.adapters is None`, `wiring.probe_passed is False`, `UNQUALIFIED_NO_CLUSTER`); a built adapter or a passed probe is a failure of the gate.
3. Capacity gate, `CI_DETERMINISTIC` posture: an in-memory backend is never reported as a passed probe (`wiring.probe_passed is False`, `CI_DETERMINISTIC_ONLY`).
4. Sandbox gate, root-less profile: a `SandboxProfile` with `enable_exec_tool=True`, no root and `INTERGRAX_SANDBOX_ROOT` removed from the environment must raise `SandboxCompositionError`.
5. Sandbox gate, `sandbox=None` with side-effect tools must raise `SandboxCompositionError`.

The expected-red check E-1 (section 5) shows the unedited scripts failing and E-3 is the audit-only mutation check.

## 4. Exact write set

```text
scripts/maintenance/check_production_capacity_adapters.py
scripts/maintenance/check_sandbox_policy_wiring.py
governance/tickets/T-IGX-043-E.md
governance/tickets/plans/PLAN_T-IGX-043-E.md
```

Product paths: none (`scripts/` is outside `PRODUCT_PATH_PREFIXES`; a `Ref:` trailer is still required). Forbidden: every other path, in particular `scripts/gates/check_audit_ideal_gates.py` (not edited), all `intergrax/` and `applications/` files, `tests/`, `.github/`, roadmap, FRZ checklist, IIAC records, the other `check_*` scripts.

## 5. Tests (script-level; no new pytest file)

| ID | What | Expected |
|---|---|---|
| E-1 (red-first) | the two **unedited** scripts run at a tree that has the `T-IGX-043-B` product part | part 2: the sandbox script exits 1 (`SandboxCompositionError`), a failure caused by B. Part 1: the capacity script exits non-zero (it raises, see section 2), a pre-existing failure that is independent of FC-2 and is recorded as such |
| E-2 | the two edited scripts at the same tree | both exit 0 and print their `OK` line |
| E-3 (mutation, audit only) | in a scratch worktree at the audited SHA, remove the capacity default-posture guard and the sandbox root guard | the edited scripts exit 1: the gates detect the regression they now guard |
| E-4 (regression) | `test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity`, `::test_audit_ideal_11_1_sandbox_product_requirement` | stay green |

## 6. Test commands (one process at a time under the shared lock; none run when drafted)

1. `uv run python scripts/maintenance/check_production_capacity_adapters.py`
2. `uv run python scripts/maintenance/check_sandbox_policy_wiring.py`
3. `uv run pytest tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_30_4_production_capacity tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py::test_audit_ideal_11_1_sandbox_product_requirement -p no:cacheprovider`
4. `uv run python -m scripts.governance.igx_gov ticket`

The full runner `scripts/gates/check_audit_ideal_gates.py` (about 70 scripts and one test file) is not part of this contract; running it needs a separate Operator authorisation.

## 7. Order of work

0. Preflight: read scope = the two scripts, the `T-IGX-043-B` plan section 4, `_RecordingEvaluator`; edit scope = the two scripts; tests = section 6. Confirm that the `T-IGX-043-B` names exist in the integration tree.
1. Run E-1 on the unedited scripts and record the output.
2. Edit the capacity script, then the sandbox script.
3. Run E-2 and E-4; ticket to `quality_gate` by the implementer; never to a PASS state.

## 8. ARCH-FREEZE impact, ACT-N11, audit, risks, rollback, effort

- ARCH-FREEZE: none of its own; it keeps two existing gates meaningful after FC-2 (they are evidence surfaces for the AUDIT-IDEAL items, not FRZ rows). No FRZ status is touched or claimed.
- ACT-N11: no product path, so this ticket alone does not break the IIAC activation; whether the two scripts are IIAC-material through the impact matrix is UNVERIFIED (the matrix lists them; the `iiac_common.py` rule was not read for it) and is checked by reading before the edit; whether the gate runner runs in CI is likewise UNVERIFIED.
- Independent audit: part of the wave audit (`PT-Q02`) at the exact wave SHA by a session that did not implement or review it: replays E-1..E-4 including the mutation check E-3 and confirms the assertion changes match section 3. The Operator's ruling on the gate meaning is not replaced by the audit.
- Risks: (R-E1) the capacity script is by reading already red at `H` (section 2), so its "red-first" evidence shows a pre-existing failure; record it as such and do not attribute it to FC-2. (R-E2) The allowing evaluator is an inline stand-in written in the script (no test import); its shape must match the evaluator protocol, read at implementation. (R-E3) The `T-IGX-043-B` names change; digest and text are re-aligned.
- Rollback: revert the script edits; the gates return to their old assertions. Part 1 returns to its pre-existing failure; part 2 fails against the FC-2 product part, so a rollback of part 2 is not independent of `T-IGX-043-B`.
- Effort: S (two small scripts, one ruling).
