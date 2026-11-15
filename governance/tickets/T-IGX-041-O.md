# T-IGX-041-O — PDC-X batch B-P8-R3: test-only refresh before the EBH-7 pin (B6-F10 classifier entries and one COMPAT-X evidence-fixture id)

## Scope

Test-infrastructure ticket, preparatory to the final EBH-7 re-pin (T-IGX-041-J), by `DELEGATION-PDC-CLOSE.md` decision 2 (P8 repair loop) and the precedents T-IGX-041-K and T-IGX-041-L of `DELEGATION-PDC-P7-REPIN.md` part (a). At `258bbf6ab` (merge of PR #584) two test-side facts remain after K and L: (1) `certify_f10()` classifies 3 paths of `git diff --name-only 5a3a05347 HEAD` (3437 paths) and of `git diff --name-only dc18e36d9 HEAD` (3223 paths) as UNKNOWN, which blocks the C4 gate (`C4-UNCLASSIFIED`): `intergrax/integrations/_shared/p3/clients.py`, `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py` and `platform_proofs/scenarios/indirect_prompt_injection/proof.json`; (2) one test of `tests/qualification/compat_x` fails (`test_every_added_surface_and_exclusion_is_present_and_every_removed_one_is_gone`) because the line number inside one recorded exclusion id of `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json` (`scheduler.py::module_level.model_validate:337:1`) moved to 362 with the P8 scheduler repair. The ticket adds three exact-path classifier entries (the default classification never changes) and corrects that one line number. The write set is test-tree only and contains no product path. It is not a certification verdict.

## Plan digest

Plan digest: b2399a1ceb7dd8271e093812724edd07e4a4b744db40fc3ce8810912a10697f7

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-O.md`](plans/PLAN_T-IGX-041-O.md).

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

plan_pending
