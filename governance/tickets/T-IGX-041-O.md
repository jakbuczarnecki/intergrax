# T-IGX-041-O — PDC-X batch B-P8-R3: test-only refresh before the EBH-7 pin (B6-F10 classifier entries and one COMPAT-X evidence-fixture id)

## Scope

Test-infrastructure ticket, preparatory to the final EBH-7 re-pin (T-IGX-041-J), by `DELEGATION-PDC-CLOSE.md` decision 2 (P8 repair loop) and the precedents T-IGX-041-K and T-IGX-041-L of `DELEGATION-PDC-P7-REPIN.md` part (a). At `258bbf6ab` (merge of PR #584) two test-side facts remain after K and L: (1) `certify_f10()` classifies 3 paths of `git diff --name-only 5a3a05347 HEAD` (3437 paths) and of `git diff --name-only dc18e36d9 HEAD` (3223 paths) as UNKNOWN, which blocks the C4 gate (`C4-UNCLASSIFIED`): `intergrax/integrations/_shared/p3/clients.py`, `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py` and `platform_proofs/scenarios/indirect_prompt_injection/proof.json`; (2) one test of `tests/qualification/compat_x` fails (`test_every_added_surface_and_exclusion_is_present_and_every_removed_one_is_gone`) because the line number inside one recorded exclusion id of `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json` (`scheduler.py::module_level.model_validate:337:1`) moved to 362 with the P8 scheduler repair. The ticket adds three exact-path classifier entries (the default classification never changes) and corrects that one line number. The write set is test-tree only and contains no product path. It is not a certification verdict.

## Plan digest

Plan digest: b2399a1ceb7dd8271e093812724edd07e4a4b744db40fc3ce8810912a10697f7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-O.md`](plans/PLAN_T-IGX-041-O.md).

## Approved

Approved: operator-delegate @ 2026-11-15T16:33:41+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-CLOSE.md@7857632dd064e9fd6583d975d781cca516c5bca5

Delegate checks (scope of `DELEGATION-PDC-CLOSE.md` decision 2, P8 repair loop, delegate-signed under the limits of `DELEGATION-PDC-P7-REPAIR.md`, with the K and L precedents of `DELEGATION-PDC-P7-REPIN.md`; all passed before this line was written; read at the batch branch `pdc-x-041o` at `a557e6808`, base `258bbf6ab` = `origin/development`; delegation file read from `origin/development`, last changed by `7857632dd`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`b2399a1ceb7dd8271e093812724edd07e4a4b744db40fc3ce8810912a10697f7`).
- Write set: the three lines of the plan's Write set block (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py`, `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py`, `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`) equal, as lists, the `writes` tuple of `B-P8-R3` in `scripts/qualification/pdc_x_wbs.py` and the `Write set` of the `B-P8-R3` section in `PDC_IMPLEMENTATION_WBS.md`. `pdc_x_wbs.py --check`: OK (107 rows, 37 batches, no write conflicts).
- Forbidden paths: nothing under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/` or `.github/` in the write set; all three files are under `tests/`.
- Classifier probe reproduced by the delegate (`certify_f10` on identity-neutral canonical inputs, live `git diff --name-only <accepted> origin/development`): for `5a3a05347..258bbf6ab` (3437 paths) and `dc18e36d9..258bbf6ab` (3223 paths) exactly `intergrax/integrations/_shared/p3/clients.py`, `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py` and `platform_proofs/scenarios/indirect_prompt_injection/proof.json` are UNKNOWN and C4 FAILS (`C4-UNCLASSIFIED`); with the three exact-path rules appended 0 paths are UNKNOWN and C4 PASSES. The five sibling paths of the plan's negative tests (including the real `intergrax/integrations/_shared/p2/clients.py`) stay UNKNOWN.
- COMPAT-X fixture: `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json:734` holds `...scheduler.py::module_level.model_validate:337:1`; the call `task = Task.model_validate(checkpoint.task_snapshot)` is on line 337 at `63005e290` and on line 362 now (`scheduler.py:362`; `584c84168` hunks at old lines 207, 231, 240). With the one-token edit `337` to `362` (applied temporarily, then reverted, nothing of it is in this commit) `tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py` passes (25 passed); the implementer's full-directory run before the edit shows exactly that one failure (1 failed, 352 passed). The removed row (`:900`, id `...:222:1`) is unaffected.
- Factual sentences and file:line references re-read against the code (about 60 spot reads, no discrepancy): `classify_change` (`_ebh7_f10_ecap_normative_matrix.py:654-660`), `evaluate_c4` (`:671`, `:702-712`), `_ebh7_f10_canonical_inputs.py:961, 1010, 606-625, 628`, `_constants.py:15`, the existing exact-path and class precedents (`invoker.py`, `providers/`, `testing_support/`, `_APP_HOST`), `pyproject.toml:6`; `clients.py` (`:4, 38, 71, 112, 162, 207, 214, 318, 323`, `_path_for` used by `put_object`, `get_object`, `delete_object`; imported only by the seven `_shared` modules the plan names); the importers of `runtime_composition.py` (`scenario.py:63`, `agent_factory.py:25`, `runtime_bundle.py:24`, three more modules of that directory, the five `testing_support/decision_e2e/` lines, `pdc_x_wbs.py:930` as a string; nothing under `intergrax/`, `agents/` or `applications/` except the `obs_diag_x3_ast_gates.py:185` directory scan); the `proof.json` descriptor (`safety_class` `LOCAL_NETWORK` to `LOCAL_MUTATING` in `116f1c0bc`, no reader under `intergrax/`, `agents/`, `applications/`); `--stat` numbers (`584c84168`: 12 insertions and 221 deletions in `runtime_composition.py`); `PLAN_T-IGX-041-M.md` line 24 and `PLAN_T-IGX-041-N.md` (6 + 2 tests).
- No gate is weakened or deleted: entries only, appended after K's; the default classification, every existing rule, every regression map and every existing test stay byte-identical (the plan names `test_ebh_7_b6_f10_c4_041k_entries_gates.py`, the closure gates and all of `tests/qualification/compat_x/` as not edited); the fixture edit moves one recorded line number and the gate still demands the exact `(id, reason)` pair; each classifier entry has negative tests (siblings stay UNKNOWN and keep `C4-UNCLASSIFIED` failing; rule-removal negative; default unchanged).
- No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict (certifies nothing, does not close B6 or promote any FRZ criterion). The implementer (planner lane) and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state.
- Judgment calls the plan flags, confirmed by the delegate: (1) exact-path entries as in K; (2) the pair `VENDOR_BACKEND` + `SECURITY_GOVERNANCE_TENANT` for `clients.py` (the stronger binding, as for `invoker.py`); (3) the one-token fixture edit is covered by the L precedent (L's own evidence file, not a gate and not a pin).
- Ordering and merge gate: the pull request merges after `B-P8-R1` and `B-P8-R2` and before the pin `P` of T-IGX-041-J, under authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md` and decision 8 of `DELEGATION-PDC-WAVE2.md`: green CI, only the known red classes (IIAC activation until the P8-04 rebind, EBH-7 identity until the final re-pin), `headRefOid` equal to the verified SHA. Any later change of the plan changes the digest and voids this signature; write-set amendments are not delegated; later product paths or a later scheduler edit above line 362 need a new ticket.

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py -q
uv run pytest tests/qualification/compat_x -q
uv run ruff check tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any default change of the classifier, directory entries for the three paths, a change of a class definition or regression map, any COMPAT-X pin, counter, gate or test, the re-pin (T-IGX-041-J), the IIAC audit, activation and rebind, FRZ promotion, OD signatures, certification.
- Everything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`, and every existing test (byte-identical).

## Lifecycle state

plan_approved
