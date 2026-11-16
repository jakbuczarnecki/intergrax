# T-IGX-041-Q — PDC-X batch B-P8-R5: test hygiene of the two B6-F10 C4 entry gates (no duplicate changed paths in the "removing the entries reopens exactly the three paths" tests)

## Scope

Test-only ticket of the second P8 repair loop, by `DELEGATION-PDC-LOOP2.md` (Commission table row "Second loop", ticket T-IGX-041-Q). At the EBH-7 pin `d39a520dd` the three paths of the K entries and the three paths of the O entries are already in `git diff --name-only 5a3a05347 d39a520dd` (3440 paths), while `test_f10_c4_041k_removing_the_entries_reopens_exactly_the_three_paths` and `test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths` append them to the canonical changed paths unconditionally, so each reason occurs twice and the one-reason-per-path assertions fail (`assert 2 == 1`, CI run 43738431866). The ticket appends a path only when the range does not already hold it (one private helper per file) and adds negative tests that prove the tests still fail, with exactly one reason, when an entry is really missing and that the reopen check also holds on a range without the paths. The classifier, the K and O entries, the replay inputs and every existing assertion stay; no gate is weakened. After the implementation only the two files run; no EBH-7 replay, no broad suite, no new pin. The write set is two test files and contains no product path. It is not a certification verdict.

## Plan digest

Plan digest: a3a8f5ac133b6b4f32e242a0dee69daf4908579f1582ed33113ddc71f26ffa52

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-Q.md`](plans/PLAN_T-IGX-041-Q.md).

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py -q
uv run ruff check tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any change of the classifier, of `C4_RULES`, of a regression map, of a class definition or of the replay inputs; the EBH-7 L3 replay, a new EBH-7 pin, the broad or full suites, the `compat_x` directory.
- Everything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`, and every other test (byte-identical); T-IGX-041-P.

## Lifecycle state

plan_pending
