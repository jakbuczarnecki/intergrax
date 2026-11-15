# PLAN T-IGX-041-O — PDC-X batch B-P8-R3: test-only refresh before the EBH-7 pin (three B6-F10 C4 classifier entries and one COMPAT-X evidence-fixture id)

Parent: PDC-X closing wave, P8 repair loop; preparatory ticket for the final EBH-7 re-pin (T-IGX-041-J, batch B-P7-RP). Authority: `operator_decisions/DELEGATION-PDC-CLOSE.md` decision 2 (P8 repair loop, delegate-signed under the limits of `DELEGATION-PDC-P7-REPAIR.md`) and `operator_decisions/DELEGATION-PDC-P7-REPIN.md` part (a), whose precedents are T-IGX-041-K (classifier entries) and T-IGX-041-L (COMPAT-X counters). This plan is written by the implementer lane and is **unsigned** (`plan_pending`); it writes no `Approved:` line. It is not a certification verdict: it certifies nothing, does not close B6 or promote any FRZ criterion, and introduces no registry, scheduler, engine, trust, policy or audit authority. The write set contains no product path.

Every file:line reference and factual sentence below was re-checked against `origin/development` at `258bbf6ab` (the merge of PR #584) in the planning session. Measurements come from `.tmp/proof/pdc-x-041o/` (not committed): `probe_o.py`, `probe_o.log`, `compat_all.log`.

## Problem

K and L are merged, but the P8 repair loop (T-IGX-041-M, PR #584, and T-IGX-041-N, PR #583) changed paths after the measurements of those two plans. Two test-side facts follow, both measured at `258bbf6ab`.

### Part 1 — three more paths are UNKNOWN to the B6-F10 classifier

1. `classify_change` (`tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py:654-660`) returns `(ChangeClass.UNKNOWN,)` for a path that no `C4Rule` matches (exact path, or a `/`-terminated prefix; longest wins), and `evaluate_c4` (`:671`) turns every UNKNOWN changed path into a failure of `C4-UNCLASSIFIED`, "unknown = BLOCKER" (`:702-712`).
2. `build_canonical_f10_inputs` sets `changed_paths` to `git diff --name-only <accepted> <TEST_EXECUTION_SHA>` (`tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py:1010`), `accepted` being `EBH7_PARENT_REPLAY_ACCEPTANCE_SHA` (`:961`, `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92` per `tests/qualification/ebh_7/_constants.py:15`).
3. **Measured at `258bbf6ab`** (probe `probe_o.py`: canonical inputs with `identity_violations` emptied and `changed_paths` replaced by the live diff to `HEAD`; the three K entries are already in `C4_RULES`, appended at `:606-624`, the tuple closing at `:625`):
   - `git diff --name-only 5a3a05347 HEAD`: 3437 paths, 3 UNKNOWN, C4 verdict BLOCKED (`C4-UNCLASSIFIED`).
   - `git diff --name-only dc18e36d9 HEAD`: 3223 paths, the same 3 UNKNOWN, C4 BLOCKED. (`dc18e36d9` is `INDEPENDENT_ACCEPTANCE_BASELINE_SHA`; `5a3a05347`, the F10 baseline, is its ancestor.)
   - The three paths: `intergrax/integrations/_shared/p3/clients.py` and `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py`, both last changed by `584c84168` (T-IGX-041-M, PR #584), and `platform_proofs/scenarios/indirect_prompt_injection/proof.json`, last changed by `116f1c0bc` (T-IGX-041-N, PR #583).
   - With one exact-path entry per path (below), both ranges end with 0 UNKNOWN and C4 PASS (`probe_o.log`).
4. The K tests are still green at `258bbf6ab` (36 passed in one run): their closure gate checks only the current pin range `5a3a05347..146e7acf1`, which does not contain these paths. The paths enter the F10 range only when the pin moves to `P`, so the final re-pin would find them and report C4 BLOCKED.

### Part 2 — one COMPAT-X evidence-fixture id is stale

1. At `258bbf6ab` the whole `tests/qualification/compat_x` directory gives **1 failed, 352 passed** (one pytest run, 1628 s, `compat_all.log`). The failing test is `tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py::test_every_added_surface_and_exclusion_is_present_and_every_removed_one_is_gone` (`:136`), assertion `:143`: the set of `(id, reason)` pairs of `added_exclusion_ids` in the fixture is not a subset of the `(exclusion_id, reason)` pairs of `build_closed_world_report().exclusions`.
2. The one missing pair is `exclusion.wire.intergrax.runtime.long_running.scheduler.py::module_level.model_validate:337:1` with reason `internal_wire_call_not_public_compat_surface`. It is recorded at `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json:734`. A wire exclusion id is `exclusion.` plus the candidate id (`_wire_exclusion`, `tests/qualification/compat_x/_compat_x_closed_world.py:452-459`); the candidate id carries the line number of the call.
3. The call is `task = Task.model_validate(checkpoint.task_snapshot)` in `intergrax/runtime/long_running/scheduler.py`. It was on line 337 at `63005e290` (the measurement base of the L plan, `git show 63005e290:intergrax/runtime/long_running/scheduler.py`) and is on line 362 now (`scheduler.py:362`), because `584c84168` (T-IGX-041-M) inserted code before it (the commit changes the file by 28 insertions and 3 deletions, three hunks at the old lines 207, 231 and 240). The current inventory contains the same exclusion at `...scheduler.py::module_level.model_validate:362:1`, same reason (probe over `build_closed_world_report`).
4. The fixture has exactly two rows naming `long_running.scheduler.py` (`:734` in `added_exclusion_ids`, `:900` in `removed_exclusion_ids`, id `...:222:1`). The removed row must stay absent from the inventory and is not affected. No other test of the directory fails, so no other recorded id moved.
5. The fixture row is id-exact evidence for the L re-pin ("counts never stand alone", module docstring of the test, `:5-10`). The counters, the pins and the meaning of the gate are unchanged; only the recorded line number of one exclusion id follows the product file.

## Classification decisions

Exact-path entries, appended at the end of `C4_RULES`, in the manner of K (`invoker.py`, `scripts/__init__.py`, `testing_support/`; and `_APP_HOST`, "enumerated per application so a new application stays unknown"): a new sibling file stays UNKNOWN. The class decides which regression nodes `C4_CLASS_REGRESSIONS` (`:628`) binds to the change; this ticket changes no regression map and no class definition (`ChangeClass`, `_ebh7_f10_types.py:41-51`).

| Path | Classes | Reason (evidence) |
| --- | --- | --- |
| `intergrax/integrations/_shared/p3/clients.py` | `VENDOR_BACKEND`, `SECURITY_GOVERNANCE_TENANT` | The module is the "Phase M.7 catalog adapters over duck-typed backends" (`:4`): `VaultSecretsStore` (`:38`), `Neo4jGraphStore` (`:71`), `HttpObservabilityBackend` (`:112`), `SentryObservabilityBackend` (`:162`), `RestVectorStoreIntegration` (`:207`), `HttpNotificationChannel` (`:214`), two browser-automation clients and `FilesystemBlobClient` (`:318`). It is imported by the provider factories under `intergrax/integrations/_shared/` (`p3/factories.py:20`, `p4/factories.py:14`, `p5/clients.py:27`, `p5/factories.py:15`, `p6/factories.py:13`, `p7/factories.py:17`, `p8/factories.py:17`). The existing class of provider integrations is `VENDOR_BACKEND` (`intergrax/integrations/providers/`, rule at `:288`). The last change (`584c84168`) added `FilesystemBlobClient._path_for` (`:323`), which refuses absolute, drive-qualified and `..` keys and keys that resolve outside the store root, and routes `put_object`, `get_object` and `delete_object` through it; `PLAN_T-IGX-041-M.md` (line 24) describes the defect it closes (variants of a governed storage key were reachable and an absolute key wrote outside the root). A secret store (`VaultSecretsStore`) is in the same file, and the precedents for secret and key-confinement enforcement carry `SECURITY_GOVERNANCE_TENANT` (`intergrax/core/security/`, `intergrax/runtime/security/`, K's entry for `secrets_store_resolver.py`). The pair binds both the vendor-backend regressions and the cross-tenant and tenant-scope regressions, the stronger choice, as for `invoker.py` (`:308`). |
| `platform_proofs/scenarios/ai_incident_investigation/application/runtime_composition.py` | `QUALIFICATION_INFRASTRUCTURE` | `platform_proofs/` is the proof library, not part of the distribution (`pyproject.toml:6` builds `packages = ["intergrax"]`; K's entry for `investigator_agent.py` rests on the same fact). The module composes the incident scenario's runtime from `intergrax.applications._shared` (imports at `:12-31`). It is imported only inside `platform_proofs/scenarios/ai_incident_investigation/` (`application/scenario.py:63`, `integration/agent_factory.py:25`, `fixtures/runtime_bundle.py:24`, and three more modules of that directory), by `testing_support/decision_e2e/` (`ai_incident_qualification_run.py:18`, `completion_alignment_producer_reachability.py:17`, `controlled_alignment/runner.py:9`, `provider_binding.py:86`, `scenario_qualification.py:18`) and by tests; `scripts/qualification/pdc_x_wbs.py:930` only names it as a string. No file under `intergrax/`, `agents/` or `applications/` imports it (`git grep platform_proofs` over those trees finds only `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py:185`, which scans `platform_proofs/scenarios` as a directory). The class of `testing_support/` is `QUALIFICATION_INFRASTRUCTURE` (rule at `:400`). The last change (`584c84168`) is 12 insertions and 221 deletions in this file (`git show --stat`). |
| `platform_proofs/scenarios/indirect_prompt_injection/proof.json` | `QUALIFICATION_INFRASTRUCTURE` | A static scenario proof descriptor (`schema_version` `intergrax.platform_proof_descriptor.v3`, `library_class` `SCENARIO`), "the canonical package-owned discovery source" of the proof suite (`scripts/proof/intergrax_platform_proof_descriptor.py:5`, file name `:29`). No file under `intergrax/`, `agents/` or `applications/` references `PROOF_DESCRIPTOR_FILENAME` (0 hits) or reads a descriptor. The last change (`116f1c0bc`, T-IGX-041-N) set `safety_class` from `LOCAL_NETWORK` to `LOCAL_MUTATING` so that the descriptor validates (`PLAN_T-IGX-041-N.md`: 6 + 2 tests failed with `ManifestLoadError: invalid platform proof descriptor`). |

## Deliverables

1. **Tests first** — new file `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py`, shown red for the right reason on the unchanged classifier (the positive cases, the exact-path checks and the rule-removal negative fail because the three paths classify as UNKNOWN; the other negatives are green before and after). Same inputs and helpers as `test_ebh_7_b6_f10_c4_041k_entries_gates.py` (`dataclasses.replace(cached_canonical_f10_inputs(), identity_violations=())`). Cases:
   - each of the three paths classifies exactly as in the table (positive, parametrized);
   - each of the three entries is an exact path (the prefix does not end with `/`), has a non-empty reason, carries the table's classes and occurs exactly once in `C4_RULES`;
   - **negatives (siblings stay UNKNOWN):** `intergrax/integrations/_shared/p3/zz_unclassified.py`, the real unchanged sibling `intergrax/integrations/_shared/p2/clients.py`, `platform_proofs/scenarios/ai_incident_investigation/application/zz_unclassified.py`, `platform_proofs/scenarios/indirect_prompt_injection/zz_unclassified.json` and `platform_proofs/scenarios/zz_unclassified/proof.json` classify as `(ChangeClass.UNKNOWN,)`, and appending each to `changed_paths` makes `C4-UNCLASSIFIED` FAIL with `unclassified change <path>` and the C4 verdict BLOCKED;
   - **negative on the rule set:** with the three entries removed from `c4_rules`, appending the three paths to `changed_paths` makes `C4-UNCLASSIFIED` FAIL with exactly one reason per path (the K entries stay present and add no reason); with the entries present the same input passes and C4 is PASS;
   - the default is unchanged: a path under no rule is `(ChangeClass.UNKNOWN,)`, and no rule carries `ChangeClass.UNKNOWN`;
   - every class used by the three entries has regression nodes in `c4_class_regressions` that the replay profile covers (`argv_covers_pytest_node`).
2. **Classifier entries** — three `C4Rule` entries appended at the end of `C4_RULES` (`_ebh7_f10_canonical_inputs.py`, after the K entries, before the closing `)` at `:625`), with the classes and reasons of the table. Nothing else in the file changes: no default, no existing rule (K's included), no regression map.
3. **Fixture id** — in `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json:734`, `...scheduler.py::module_level.model_validate:337:1` becomes `...:362:1` (one token, the line number). Nothing else in the fixture changes. The existing test of Part 2 is the red-to-green pin (red before: 1 failed in `test_compat_x_pdc_x_p7_repin_delta.py`; green after); no test is edited.
4. **Evidence for the checkpoint (phase 2, recorded in the ticket, not committed code):** the probe of Part 1 re-run on the implemented tree for both ranges (0 UNKNOWN, C4 PASS), the whole `tests/qualification/compat_x` directory green, and `git diff --name-only 5a3a05347 146e7acf1` showing none of the three paths is in the current pinned range.

## Write set

The block below is the complete code and test write set, one path per line. It is equal, line for line, to the `writes` tuple of batch `B-P8-R3` in `scripts/qualification/pdc_x_wbs.py` and to the "Write set" line of the `B-P8-R3` section in `PDC_IMPLEMENTATION_WBS.md`. Anything else is a write-set amendment, which is not delegated.

```text
tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py
tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
tests/fixtures/compat_x/repin_pdc_x_p7_delta.json
```

Governance and documents (not part of the row): `governance/tickets/T-IGX-041-O.md`, this plan, `scripts/qualification/pdc_x_wbs.py` and the generated `PDC_IMPLEMENTATION_WBS.md` / `PDC_DEPENDENCY_DAG.md`.

Not edited: anything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`; `_ebh7_f10_ecap_normative_matrix.py` (the classifier code), `_ebh7_f10_types.py`, `test_ebh_7_b6_f10_c4_041k_entries_gates.py`, `test_ebh_7_b6_f10_c1_c4_closure_gates.py`, `tests/qualification/compat_x/` (every test and pin, byte-identical), `MANIFEST.json`, the pins and the allowlist; the F10 matrix and report documents; the re-pin itself (T-IGX-041-J).

## Gate-semantics note

Entries are added to a closed-world classifier; the unknown-is-a-blocker default and every existing rule stay. The fixture edit moves one recorded line number to the measured one; the gate still demands that the exact `(id, reason)` pair exists in the inventory and still fails on a missing, extra or substituted id (the L file's own negative tests stay unchanged). No gate is weakened or deleted. The change lies under `tests/` and is outside the executable-identity allowlist: after it merges the EBH-7 identity gate stays red until the final re-pin (accepted in `DELEGATION-PDC-WAVE2.md` decision 6 and `DELEGATION-PDC-P7-REPIN.md`, "Ordering"). It touches no path the IIAC activation covers.

**Fragility and ordering.** The evidence-fixture id carries a line number, so any later edit of `intergrax/runtime/long_running/scheduler.py` above line 362 would break the test again; likewise any later product path outside the K and O entries would enter the F10 range as UNKNOWN. The pull request therefore merges after the P8 repair batches (`B-P8-R1`, `B-P8-R2`) and before the pin `P` of T-IGX-041-J, and the re-pin session re-runs both measurements at `P` as its first check (`certify_f10()` and the COMPAT-X directory) and reports any new path instead of adding entries silently.

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before the step; stop and ask when more files than this plan names are needed.
2. The new test file is committed first and recorded red (`.tmp/proof/pdc-x-041o/`); then the three entries; then the fixture token.
3. One `uv run pytest` over `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py`, `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py` and `tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py`; one `uv run pytest tests/qualification/compat_x -q` in the background (about 27 minutes); `uv run ruff check` and `ruff format --check` on the touched Python files and `uv run pyright` on the new test file. Logs under `.tmp/proof/pdc-x-041o/`.
4. `uv run python scripts/qualification/pdc_x_wbs.py --check` and the equality of the Write set block with the `writes` tuple, before the commit and before the PR.
5. Commits carry exactly one `Ref: governance/tickets/T-IGX-041-O.md` trailer, no AI attribution, no `Approved:` line (the implementer never writes one). One pull request, not merged by the implementer.

## Acceptance

- The three paths classify as in the table; the new tests are red before the entries and green after; the K tests and the existing C1/C4 tests stay green byte-identical.
- The negatives hold: the sibling paths and any unlisted path stay UNKNOWN and keep `C4-UNCLASSIFIED` failing.
- The probe over `5a3a05347..HEAD` and `dc18e36d9..HEAD` reports 0 UNKNOWN and C4 PASS on the implemented tree (including this ticket's own governance files in the diff).
- `uv run pytest tests/qualification/compat_x -q` passes completely (the one red test is green).
- `git diff --stat` lists only the write set and the governance documents.

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py -q
uv run pytest tests/qualification/compat_x -q
uv run ruff check tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041o_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any default change of the classifier, any directory entry for the three paths, any change of a class definition or regression map, any COMPAT-X pin, counter, gate or test, any product path, the re-pin (T-IGX-041-J), the IIAC audit, activation and rebind, FRZ promotion, OD signatures, certification.
- A line-independent id scheme for the COMPAT-X evidence fixture: a change of the evidence design, not a refresh; it would be a separate ticket.

## Operator decisions needed

None beyond `DELEGATION-PDC-CLOSE.md` decision 2 and `DELEGATION-PDC-P7-REPIN.md` part (a). Judgment calls for the delegate to confirm: (1) exact-path entries (as in K); (2) the pair `VENDOR_BACKEND` + `SECURITY_GOVERNANCE_TENANT` for `intergrax/integrations/_shared/p3/clients.py` (the stronger choice) instead of `VENDOR_BACKEND` alone; (3) that the one-token fixture edit in `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json` is covered by the L precedent (the fixture is L's own evidence file, it is not a gate and not a pin).
