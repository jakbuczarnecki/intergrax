# PLAN T-IGX-041-Q — PDC-X batch B-P8-R5: test hygiene of the two B6-F10 C4 entry gates (no duplicate changed paths in the "removing the entries reopens exactly the three paths" tests)

Parent: PDC-X P8, repair loop 2 (one loop, approved by the Operator). Authority: `operator_decisions/DELEGATION-PDC-LOOP2.md` (Commission table, row "Second loop": ticket **T-IGX-041-Q**, deduplicate the appended paths in the K and O gate tests, no gate is weakened; row "Condition": targeted tests only). That record is carried by PR #590 (branch `pdc-x-loop2-record`) and is merged on `development` before this plan is signed. This plan is written by the planner lane and is **unsigned** (`plan_pending`); it writes no `Approved:` line. It is not a certification verdict: it certifies nothing, does not close B6 or promote any FRZ criterion, and introduces no registry, scheduler, engine, trust, policy or audit authority. The implementer and the signer are different sessions. The write set contains no product path.

**Targeted-test condition (normative, from `DELEGATION-PDC-LOOP2.md`).** After the implementation only the tests that cover the repair run: the new or changed tests of the two files in the Write set. The EBH-7 L3 replay is NOT run, no broad or full suite is run (no `tests/qualification/ebh_7` directory run, no `tests/qualification/compat_x` run, no global audit pytest), and no new EBH-7 pin is taken.

Every file:line reference and factual sentence below was re-checked against `origin/development` at `76516c4dd` (the merge of PR #589) in the planning session. Measurements: `git diff --name-only 5a3a05347 d39a520dd` (read-only git, no pytest) and the failed CI run 43738431866 read with `gh run view --log-failed`.

## Problem

1. The two tests `test_f10_c4_041k_removing_the_entries_reopens_exactly_the_three_paths` (`tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py:92`) and `test_f10_c4_041o_removing_the_entries_reopens_exactly_the_three_paths` (`tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py:93`) build their input with `closed = replace(base, changed_paths=(*base.changed_paths, *paths))` (K `:95`, O `:96`): the three paths of the file's `_EXPECTED` map are appended to the canonical changed paths unconditionally.
2. `base.changed_paths` is `git diff --name-only <accepted> <TEST_EXECUTION_SHA>` (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:1029`), with `accepted` = `EBH7_PARENT_REPLAY_ACCEPTANCE_SHA` (`:980`, `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` per `tests/qualification/ebh_7/_constants.py:15`) and `TEST_EXECUTION_SHA` read from `docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json` (`:2`, `d39a520dda965981a38f8e1a90ec5135ea5d508e`; read at `_ebh7_f10_canonical_inputs.py:973`, manifest path `:29`).
3. **Measured at `76516c4dd`:** `git diff --name-only 5a3a05347 d39a520dd` lists 3440 paths, and each of the six paths of the two `_EXPECTED` maps occurs exactly once in it: `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py`, `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py` (K), `intergrax/integrations/_shared/p3/clients.py`, `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py` and `platform_proofs/scenarios/indirect_prompt_injection/proof.json` (O). At the previous pin `146e7acf1` (`git diff --name-only 5a3a05347 146e7acf1`, 2488 paths) none of the six paths is in the range, so the unconditional append produced no duplicate there; the pin move to `d39a520dd` is what made the paths duplicates.
4. `c4_classification` (`tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py:663-668`) iterates `inputs.changed_paths` without removing duplicates, and `evaluate_c4` builds one reason per entry of the UNKNOWN bucket (`:699`, `:712`: `[f"unclassified change {p}" for p in unknown]`). With the entries removed, a path that occurs twice in `changed_paths` therefore yields the reason twice.
5. The tests assert exactly one reason per path (K `:103`, O `:104`: `sum(1 for reason in result.reasons if reason == f"unclassified change {path}") == 1`) and exactly three reasons (K `:104`, O `:105`). Both assertions fail at the new pin. The EBH-7 L3 replay at the pin `d39a520dd` (CI run 43738431866) reported exactly these two failures; the log lines are `test_ebh_7_b6_f10_c4_041k_entries_gates.py:103: AssertionError` and `test_ebh_7_b6_f10_c4_041o_entries_gates.py:104: AssertionError`, each followed by `assert 2 == 1`.
6. This is a test defect, not a product defect and not a classifier defect: the classifier, the K and O entries and the replay inputs are correct, and the two tests assume a range that does not yet contain the paths. `DELEGATION-PDC-LOOP2.md` records the same diagnosis. The replay run stays recorded as failed and "not B6 PASS".

## Design

The only change is how the test builds its input: append a path to `changed_paths` only when the range does not already hold it. Each of the two files gets one private helper (the files already repeat their small helpers `_neutral`, `_result`, `_without_entries` instead of importing from each other; K's and O's stay independent):

```python
def _with_paths(base: F10Inputs, paths: tuple[str, ...]) -> tuple[str, ...]:
    """Changed paths of ``base`` plus the given paths that the range does not already hold (never a duplicate)."""
    return (*base.changed_paths, *(path for path in paths if path not in base.changed_paths))
```

and `closed = replace(base, changed_paths=_with_paths(base, paths))` in the changed test. The classifier, `c4_classification` and `evaluate_c4` are not changed: a duplicate in `changed_paths` must keep producing a reason per occurrence (that is the classifier's behaviour, not the test's concern). A dict-based `tuple(dict.fromkeys(...))` of the whole sequence would also work; the helper form is chosen because it leaves `base.changed_paths` byte-identical and states the intent.

No assertion of the changed tests is touched: the PASS assertions on `closed`, the FAIL status on `reopened`, exactly one reason per path, exactly `len(paths)` reasons and the BLOCKED C4 verdict stay as they are.

## Deliverables

1. **Changed test (both files):** the line `closed = replace(base, changed_paths=(*base.changed_paths, *paths))` becomes `closed = replace(base, changed_paths=_with_paths(base, paths))`; the helper is added after `_without_entries` (K `:62-63`, O `:63-64`). The changed test is red-to-green at the pin: red before (2 failed, as in run 43738431866) and green after.
2. **New negative tests (both files, same names with the `041k` / `041o` infix):**
   - `test_f10_c4_<k|o>_removing_one_entry_reopens_exactly_that_path`, parametrized over `sorted(_EXPECTED)`: from `closed` remove only the one rule whose prefix is the parametrized path; `C4-UNCLASSIFIED` must FAIL with `list(result.reasons) == [f"unclassified change {path}"]` (exactly one reason, exactly that path) and the C4 verdict must be BLOCKED. This is the check that the tests still fail when an entry is really missing: the dedupe does not hide a missing entry.
   - `test_f10_c4_<k|o>_reopen_check_does_not_depend_on_the_paths_being_in_the_range`: on `replace(base, changed_paths=tuple(p for p in base.changed_paths if p not in _EXPECTED))` add the three paths with `_with_paths`, remove the entries, and expect exactly three reasons, one per path. It proves that the test also holds at a pin that does not contain the paths yet (the K situation at `146e7acf1`).
   - `test_f10_c4_<k|o>_with_paths_appends_only_missing_paths`: on a synthetic base `replace(base, changed_paths=("zz_q/a.py", first))`, where `first` is the first path of `_EXPECTED`, `_with_paths(synthetic, tuple(_EXPECTED))` equals `("zz_q/a.py", first, *the other two paths in _EXPECTED order)` (the present path is not repeated, the absent ones are appended once); and on the unchanged `base` with paths that are all in the range it returns `base.changed_paths` unchanged (no assertion on the length of the range).
3. **Evidence for the checkpoint (phase 2, recorded in the ticket, not committed code):** the red-before / green-after of the two changed tests on the unchanged and the changed file, and the single pytest command of the Test contract.

## Write set

The block below is the complete code and test write set, one path per line. It is equal, line for line, to the `writes` tuple of batch `B-P8-R5` in `scripts/qualification/pdc_x_wbs.py` and to the "Write set" line of the `B-P8-R5` section in `PDC_IMPLEMENTATION_WBS.md`. Anything else is a write-set amendment, which is not delegated.

```text
tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
```

Governance and documents (not part of the row): `governance/tickets/T-IGX-041-Q.md`, this plan, `scripts/qualification/pdc_x_wbs.py` and the generated `PDC_IMPLEMENTATION_WBS.md` / `PDC_DEPENDENCY_DAG.md`.

Not edited: anything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`; `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` (the C4 rules and the canonical inputs), `_ebh7_f10_ecap_normative_matrix.py` (the classifier), `_ebh7_f10_types.py`, `_constants.py`, `test_ebh_7_b6_f10_c1_c4_closure_gates.py`, every other test and `tests/qualification/compat_x/`, `MANIFEST.json`, the pins and the allowlist.

## Gate-semantics note

No gate is weakened or deleted. Every existing assertion of the two files stays byte-identical except the one input line of each changed test; the classifier default (a path under no rule is UNKNOWN and blocks `C4-UNCLASSIFIED`), the K and O entries and the regression maps are not touched; the new tests only add strictness (one entry missing gives exactly one reason; the helper never drops a path that is not in the range). The change lies under `tests/` and is outside the executable-identity allowlist: the EBH-7 identity gate stays red until a later re-pin, which this ticket does not take (condition of `DELEGATION-PDC-LOOP2.md`).

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before the step; stop and ask when more files than this plan names are needed.
2. Tests first: the two changed tests are shown red on the unchanged files at the manifest pin (2 failed, `assert 2 == 1`) and recorded; then the helper, the changed line and the new tests are added and shown green.
3. One `uv run pytest` over the two files of the Write set (the Test contract); `uv run ruff check` and `ruff format --check` on the two files and `uv run pyright` on them. No other pytest run, no EBH-7 replay, no `compat_x`, no full suite. Logs under `.tmp/proof/pdc-x-041q/`.
4. `uv run python scripts/qualification/pdc_x_wbs.py --check` and the equality of the Write set block with the `writes` tuple, before the commit and before the PR.
5. Commits carry exactly one `Ref: governance/tickets/T-IGX-041-Q.md` trailer, no AI attribution, no `Approved:` line (the implementer never writes one). One pull request, not merged by the implementer.

## Acceptance

- The two changed tests are green at the manifest pin and were red before the change; each of the other tests of the two files that was green before stays green.
- The new negatives hold: removing exactly one entry yields exactly one reason for exactly that path; the reopen check also holds on a range that does not contain the paths.
- `git diff --stat` lists only the two test files and the governance documents.
- No other test was run (condition of `DELEGATION-PDC-LOOP2.md`).

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py -q
uv run ruff check tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any change of the classifier, of `C4_RULES`, of a regression map or of a class definition; any change of the replay inputs; the EBH-7 L3 replay, a new EBH-7 pin, the broad or full suites, the `compat_x` directory; the IIAC activation and rebind; FRZ promotion; OD signatures; certification.
- Ticket T-IGX-041-P (the G-23 product repair) and every other finding of the global audit.

## Operator decisions needed

None beyond `DELEGATION-PDC-LOOP2.md`. Judgment call for the delegate to confirm: the helper form `_with_paths` (append only the absent paths) instead of de-duplicating the whole sequence.
