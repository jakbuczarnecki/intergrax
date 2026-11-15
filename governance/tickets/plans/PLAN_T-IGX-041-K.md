# PLAN T-IGX-041-K — PDC-X batch B-P7-F10: three B6-F10 C4 classifier entries for the paths the classifier reports as UNKNOWN

Parent: PDC-X closing wave; preparatory ticket for the final EBH-7 re-pin (T-IGX-041-J, batch B-P7-RP). Authority: `operator_decisions/DELEGATION-PDC-P7-REPIN.md` (Operator answer "Yes, I approve (a)-(c)", original "Tak, zatwierdzam (a)-(c)", 2026-11-15), part (a): add classifier entries to the B6-F10 canonical inputs for the three paths the classifier reports as UNKNOWN; entries are added, the default classification never changes. This plan is written by the implementer lane and is **unsigned** (`plan_pending`); it writes no `Approved:` line. It is not a certification verdict: it certifies nothing, does not close B6 or promote any FRZ criterion, and introduces no registry, scheduler, engine, trust, policy or audit authority.

Every file:line reference and factual sentence below was re-checked against `origin/development` at `63005e290` (the merge of PR 576) in the planning session. Measurements come from probe scripts under `.tmp/session/041k/` (not committed).

## Problem

1. **The F10 gate C4 reports three UNKNOWN paths and blocks.** `classify_change` (`tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py:654-660`) picks the longest `C4Rule` whose prefix equals the path or is a `/`-terminated prefix of it (`:657-659`) and returns `(ChangeClass.UNKNOWN,)` when no rule matches (`:660`). `evaluate_c4` turns every UNKNOWN changed path into a failure of the obligation `C4-UNCLASSIFIED` with classification "unknown = BLOCKER" (`:702-712`).
2. **Which range the classifier sees.** `build_canonical_f10_inputs` sets `changed_paths` to `git diff --name-only <accepted> <TEST_EXECUTION_SHA>` (`_ebh7_f10_canonical_inputs.py:991`), where `accepted` is `EBH7_PARENT_REPLAY_ACCEPTANCE_SHA` (`:942`, defined at `tests/qualification/ebh_7/_constants.py:15` as `5a3a05347aaea93a8e4086fc90bc08e6d5eeaf92`). That baseline is an ancestor of `INDEPENDENT_ACCEPTANCE_BASELINE_SHA` `dc18e36d971c93c8f631fbc86f73149d1b81b4d7` (`docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json:8`), so the F10 range is the larger one. With the current pin `146e7acf1` none of the three paths below lies in the range (`git diff --name-only 5a3a05347 146e7acf1` does not list them), so the current F10 matrix render is unaffected by this ticket. After the final re-pin the range ends at the new pin `P` and contains them.
3. **Measured at `63005e290` (probe `.tmp/session/041k/probe_certify.py`, `identity_violations` emptied, `changed_paths` replaced by the live diff):**
   - `git diff --name-only 5a3a05347 HEAD`: 3338 paths, 3 UNKNOWN, C4 verdict BLOCKED (`C4-UNCLASSIFIED`).
   - `git diff --name-only dc18e36d9 HEAD`: 3124 paths, the same 3 UNKNOWN, C4 BLOCKED.
   - The three paths are `intergrax/experiments/workflow.py`, `intergrax/integrations/credentials/secrets_store_resolver.py` and `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py`. Each was last changed after `5a3a05347` by exactly one commit: `c076c7d0b` (D5 repair, T-IGX-041-E), `9ed0ee4f5` ("fix(security): ... no provider text in the credential error chain") and `864fe6adb` ("add Agent.run to the incident fixture agent", D3 repair).
   - With one exact-path entry per path (below) both ranges end with 0 UNKNOWN and the C4 verdict is PASS.
4. **Why the entries must merge before the pin.** `governance/tickets/` and `tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py` are outside the executable-tree identity allowlist (`PLAN_T-IGX-036-Y.md`, "Sequencing facts", first bullet), so this ticket, its plan and the classifier change merge before the pin `P` is taken (`DELEGATION-PDC-P7-REPIN.md`, "Ordering"). `PLAN_T-IGX-036-Y.md` ("Sequencing facts", last bullet) already states that closing an unknown path takes "one more directory entry in the classifier (a documentation-and-test-only follow-up), never a default change"; the three entries here are exact-path rules, the narrower form of the same follow-up (a `C4Rule` prefix that is an exact path matches only that file, `_ebh7_f10_ecap_normative_matrix.py:654-660`), so no sibling path is classified by them.

## Classification decisions

Exact-path entries (like the existing exact entries `intergrax/runtime/nexus/tools/invoker.py`, `_ebh7_f10_canonical_inputs.py:308`, and `scripts/__init__.py`, `:432`), not directory entries: a new sibling file in the same directory stays UNKNOWN, as the file's own comment demands for applications (`_APP_HOST`, `:227-230`: "enumerated per application so a new application stays unknown"). The class decides which regression nodes `C4_CLASS_REGRESSIONS` (`:609`) binds to the change; this ticket changes no regression map and no class definition (`ChangeClass`, `_ebh7_f10_types.py:41-51`).

| Path | Classes | Reason (evidence) |
| --- | --- | --- |
| `intergrax/experiments/workflow.py` | `RUNTIME_BEHAVIOR`, `SECURITY_GOVERNANCE_TENANT` | The module runs a root task through `UnifiedTaskRunner(...).run_task(task, run_id=mint_run_id())` (`workflow.py:164-169`), passing `admit_root_governance_identity=admit_certified_internal_harness_root_governance_identity` (`:167`; imported at `:29-31`). It is therefore an execution path that admits a governance identity, the same shape as the existing pair for the tool invoker. It is listed in the closed-world `LEGACY_UNIFIED_TASK_RUNNER_IMPORT_ALLOWLIST` of the modules that may import `UnifiedTaskRunner` (harness, scheduler and eval orchestration; `tests/unit/runtime/architecture/gr2_r3_model_c1_gate_policy.py:60`). The last change (`c076c7d0b`) added exactly that identity admission and `mint_run_id`. A change here must re-run both regression sets; the single class `RUNTIME_BEHAVIOR` (the class of `intergrax/eval/`) was rejected because it would not bind the tenant and governance regressions. |
| `intergrax/integrations/credentials/secrets_store_resolver.py` | `SECURITY_GOVERNANCE_TENANT` | `SecretsStoreCredentialResolver` (`:20`) resolves a `CredentialRef` through the `SecretsStore` seam and enforces the tenant scope before it reads (`_assert_tenant_scope`, `:62`, called at `:32`); it maps failures to sanitized credential errors. It is imported by `intergrax/integrations/credentials/__init__.py:11`, `credentials/google_workspace.py:17` and `intergrax/integrations/providers/collaboration_suite/google_workspace/tenant_connection_factory.py:22`. The existing precedents for secret and tenant-scope enforcement carry the single class `SECURITY_GOVERNANCE_TENANT` (`intergrax/core/security/`, `intergrax/runtime/security/`); the last change (`9ed0ee4f5`) removed provider text from the credential error chain. `RUNTIME_BEHAVIOR` is not added: its regression nodes exercise the tool invoker, not credential resolution, while the security class binds the cross-tenant and tenant-scope nodes. |
| `platform_proofs/scenarios/ai_incident_investigation/application/investigator_agent.py` | `QUALIFICATION_INFRASTRUCTURE` | `platform_proofs/` is the proof library of executable scenario claims (`platform_proofs/README.md`), not part of the distribution: `pyproject.toml:6` builds `packages = ["intergrax"]`. No file under `intergrax/`, `agents/` or `applications/` imports it: `git grep platform_proofs` over those trees finds only `intergrax/runtime/architecture/obs_diag_x3_ast_gates.py:185`, which scans `platform_proofs/scenarios` as a directory (an AST gate), not as an import. The agent class `IncidentInvestigatorAgent` (`:170`) is imported only inside `platform_proofs/scenarios/ai_incident_investigation/` (`application/scenario.py:44`, `integration/agent_factory.py:22`, `proof/evidence_builder.py:20`) and by the unit tests under `tests/unit/platform_proofs/scenarios/ai_incident_investigation/` (for example `test_architecture_scaffold_gate.py:17`, `test_autonomous_evidence_gathering.py:40`, `test_baseline_evidence_gathering.py:29`); `testing_support/decision_e2e/` only names its path as a string. The same class holds the other test-support code (`testing_support/`, rule at `:400`, reason "imported only by tests and */tests/ packages"). The last change (`864fe6adb`) added `Agent.run` so the fixture agent can be instantiated. |

## Deliverables

1. **Tests first** — new file `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py`, shown red for the right reason on the unchanged classifier (the positive cases fail because the three paths classify as UNKNOWN; the negatives are green before and after). The replay profile runs the whole directory (`docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json` `pytest_profile.command` names `tests/qualification/ebh_7/`), so the file is part of the replay. Cases, on identity-neutral canonical inputs (`dataclasses.replace(cached_canonical_f10_inputs(), identity_violations=())`, as in `test_ebh_7_b6_f10_c1_c4_closure_gates.py:_neutral`):
   - each of the three paths classifies exactly as in the table above (positive, parametrized);
   - each of the three entries is an exact path (the prefix does not end with `/`), has a non-empty reason and does not duplicate another prefix;
   - **negatives:** a sibling of each path (`intergrax/experiments/zz_unclassified.py`, `intergrax/integrations/credentials/zz_unclassified.py`, `platform_proofs/scenarios/ai_incident_investigation/application/zz_unclassified.py`, `platform_proofs/zz_unclassified/x.py`) still classifies as `(ChangeClass.UNKNOWN,)`, and appending it to `changed_paths` makes `C4-UNCLASSIFIED` FAIL and the C4 verdict BLOCKED ("an unknown path still yields UNKNOWN and a BLOCKER");
   - **negative on the rule set:** with the three entries removed from `c4_rules`, appending the three paths to `changed_paths` makes `C4-UNCLASSIFIED` FAIL with one reason per path; with the entries present the same input passes and the C4 verdict is PASS (the entries, and nothing else, close the three paths);
   - the default is unchanged: a path under no rule is `(ChangeClass.UNKNOWN,)`, no rule carries `ChangeClass.UNKNOWN`, and each of the three prefixes occurs exactly once in `C4_RULES`;
   - every class used by the three entries has regression nodes in `c4_class_regressions` that the replay profile covers (`argv_covers_pytest_node`).
2. **Classifier entries** — three `C4Rule` entries appended at the end of `C4_RULES` (`_ebh7_f10_canonical_inputs.py`, the tuple closes at `:606`, before `_TOOLS` at `:608`), with the classes and reasons of the table. Nothing else in the file changes: no default, no existing rule, no regression map.
3. **Evidence for the checkpoint (phase 2, recorded in the ticket, not committed code):** the probe of Problem 3 re-run on the implemented tree for both ranges (0 UNKNOWN, C4 PASS), and `git diff --name-only 5a3a05347 146e7acf1` showing the three paths are outside the current pinned range.

## Write set

The block below is the complete code and test write set, one path per line. It is equal, line for line, to the `writes` tuple of batch `B-P7-F10` in `scripts/qualification/pdc_x_wbs.py` and to the "Write set" line of the `B-P7-F10` section in `PDC_IMPLEMENTATION_WBS.md`. Anything else is a write-set amendment, which is not delegated.

```text
tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py
tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
```

Governance and documents (not part of the row): `governance/tickets/T-IGX-041-K.md`, this plan, `scripts/qualification/pdc_x_wbs.py` and the generated `PDC_IMPLEMENTATION_WBS.md` / `PDC_DEPENDENCY_DAG.md`.

Not edited: anything under `intergrax/`, `agents/`, `applications/`, `platform_proofs/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`; `_ebh7_f10_ecap_normative_matrix.py` (the classifier code), `_ebh7_f10_types.py`, `test_ebh_7_b6_f10_c1_c4_closure_gates.py` and every other existing test (byte-identical); `MANIFEST.json`, the pins and the allowlist; the F10 matrix and report documents; the COMPAT-X counters (T-IGX-041-L); the re-pin itself (T-IGX-041-J).

## Gate-semantics note

Entries are added to a closed-world classifier; the unknown-is-a-blocker default and every existing rule stay. No gate is weakened or deleted. Each new entry has a positive test and the unknown-path negatives above. The change lies under `tests/` and is outside the executable-identity allowlist: after it merges the EBH-7 identity gate stays red until the final re-pin (accepted in `DELEGATION-PDC-WAVE2.md` decision 6 and `DELEGATION-PDC-P7-REPIN.md`, "Ordering"). It touches no path the IIAC `MATERIAL_GOVERNANCE_PATHS` lists, which the implementer confirms before the PR.

## Protocol

1. Token-budget preflight (read scope, edit scope, tests) before the step; stop and ask when more files than this plan names are needed.
2. The new test file is committed first and recorded red (`.tmp/proof/pdc-x-041k/`); then the three entries.
3. One `uv run pytest` over `tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py` and `tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py`; then `uv run ruff check` and `ruff format --check` on the two touched files and `uv run pyright` on the new test file. Logs under `.tmp/proof/pdc-x-041k/`.
4. `uv run python scripts/qualification/pdc_x_wbs.py --check` and the equality of the Write set block with the `writes` tuple, before the commit and before the PR.
5. Commits carry exactly one `Ref: governance/tickets/T-IGX-041-K.md` trailer, no AI attribution, no `Approved:` line (the implementer never writes one). One pull request, not merged by the implementer; it merges before the pin `P` of T-IGX-041-J.

## Acceptance

- The three paths classify as in the table; the new tests are red before the entries and green after; the existing C1/C4 tests of `test_ebh_7_b6_f10_c1_c4_closure_gates.py` stay green byte-identical.
- The negatives hold: sibling paths and any unlisted path stay UNKNOWN and keep `C4-UNCLASSIFIED` failing.
- The probe over `5a3a05347..HEAD` and `dc18e36d9..HEAD` reports 0 UNKNOWN and C4 PASS on the implemented tree.
- `git diff --stat` lists only the write set and the governance documents.

## Test contract

```text
uv run pytest tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c1_c4_closure_gates.py -q
uv run ruff check tests/qualification/ebh_7/_ebh7_f10_canonical_inputs.py tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
uv run pyright tests/qualification/ebh_7/test_ebh_7_b6_f10_c4_041k_entries_gates.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- Any default change of the classifier, any directory entry for the three paths, any change of a class definition or regression map, the COMPAT-X counters (T-IGX-041-L), the re-pin (T-IGX-041-J), the IIAC audit, activation and rebind, FRZ promotion, OD signatures, certification.

## Operator decisions needed

None beyond `DELEGATION-PDC-P7-REPIN.md` part (a). Two judgment calls for the delegate to confirm: (1) exact-path entries instead of directory entries; (2) the pair `RUNTIME_BEHAVIOR` + `SECURITY_GOVERNANCE_TENANT` for `intergrax/experiments/workflow.py` (the stronger choice, binding the larger regression set).
