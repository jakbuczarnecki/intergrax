# T-IGX-041-Q — PDC-X batch B-P8-R5: test hygiene of the two B6-F10 C4 entry gates (no duplicate changed paths in the "removing the entries reopens exactly the three paths" tests)

## Scope

Test-only ticket of the second P8 repair loop, by `DELEGATION-PDC-LOOP2.md` (Commission table row "Second loop", ticket T-IGX-041-Q). At the EBH-7 pin `d39a520dd` the three paths of the K entries and the three paths of the O entries are already in `git diff --name-only 5a3a05347 d39a520dd` (3440 paths), while `test_f10_c4_041k_removing_the_entries_reopens_exactly_the_three_paths` and `test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths` append them to the canonical changed paths unconditionally, so each reason occurs twice and the one-reason-per-path assertions fail (`assert 2 == 1`, CI run 43738431866). The ticket appends a path only when the range does not already hold it (one private helper per file) and adds negative tests that prove the tests still fail, with exactly one reason, when an entry is really missing and that the reopen check also holds on a range without the paths. The classifier, the K and O entries, the replay inputs and every existing assertion stay; no gate is weakened. After the implementation only the two files run; no EBH-7 replay, no broad suite, no new pin. The write set is two test files and contains no product path. It is not a certification verdict.

## Plan digest

Plan digest: a3a8f5ac133b6b4f32e242a0dee69daf4908579f1582ed33113ddc71f26ffa52

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-Q.md`](plans/PLAN_T-IGX-041-Q.md).

## Approved

Approved: operator-delegate @ 2026-11-16T06:55:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-LOOP2.md@04a741d47aa10c69cb19cf344338fb2fff09d918

Delegate checks (scope of `DELEGATION-PDC-LOOP2.md`; the record is on `origin/development`, merge commit `04a741d47`, file last changed by `1e68151af`; plan read at commit `951f03f5e` of the branch `pdc-x-plan-041pq`, whose ancestor is `76516c4dd`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`a3a8f5ac133b6b4f32e242a0dee69daf4908579f1582ed33113ddc71f26ffa52`).
- The write set of the plan (2 entries) equals the `writes` tuple of batch `B-P8-R5` in `pdc_x_wbs.py` (compared as lists, equal; `pdc_x_wbs.py --check` OK). It is exactly the two named test files; no product path, nothing under `scripts/governance/` or `.github/`.
- Measurements and file:line re-run by the delegate: `git diff --name-only 5a3a05347 d39a520dd` lists 3440 paths and each of the six `_EXPECTED` paths occurs exactly once; `git diff --name-only 5a3a05347 146e7acf1` lists 2488 paths and none of the six; `_ebh7_f10_canonical_inputs.py:29, 973, 980, 1029`, `_constants.py:15`, `MANIFEST.json:2` (`d39a520dd...`), `_ebh7_f10_ecap_normative_matrix.py:663-668, 699, 712`; K test `:92, :95, :103, :104`, `_without_entries` `:62-63`; O test `:93, :96, :104, :105`, `_without_entries` `:63-64`; the CI run 43738431866 (`gh run view --log-failed`) shows `2 failed, 720 passed, 9 skipped`, the assertion lines `041k...:103` and `041o...:104`, and `assert 2 == 1`. All true.
- No gate is weakened or deleted: every assertion of the two tests stays, only the input line changes (append a path only when absent); the classifier, `C4_RULES`, the K and O entries and the replay inputs are not touched; the three new negative tests only add strictness. No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict and states the targeted-test condition (the two files only; no `ebh_7` directory run, no `compat_x`, no EBH-7 replay, no new pin).
- The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Not checked by the delegate: the behaviour of the new tests (not written yet) and no pytest was run for Q (it would run the EBH-7 canonical-input build, which the targeted-test condition leaves to the implementer's single run).
- Merge gate (as the record states): authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md`, green CI with the known red classes only (IIAC activation staleness, EBH-7 identity), head equal to the verified SHA. Write-set amendments are not delegated.

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

plan_approved
