# PLAN T-IGX-041-L — PDC-X batch B-P7-CX: re-pin of the COMPAT-X counters (debt item Z-D1 of `PLAN_T-IGX-036-Z.md`) at the last product change

Parent: PDC-X implementation WBS (`PDC_IMPLEMENTATION_WBS.md`, batch **B-P7-CX**). Authority: `operator_decisions/DELEGATION-PDC-P7-REPIN.md`, part (a) ("T-IGX-041-L (re-pin the COMPAT-X counters of debt item Z-D1 at the last product change, without weakening any gate)"); the debt item is D1 of `PLAN_T-IGX-036-Z.md` (line 53: the counters "are re-pinned once, together with the final EBH-7 re-pin, after the last product change"). Requirements: none (no `PDC-Dn-Rnnn` row changes status). This plan is not a certification verdict. It changes **tests and test fixtures only**: nothing under `intergrax/`, `agents/`, `applications/`, `scripts/` or `.github/`.

Every number below was measured by running the repository's own COMPAT-X code on `63005e290` (`origin/development` after PR #576; `origin/development` is now `cba32d247`, which differs from it by `docs/` only, checked with `git diff --stat 63005e290 origin/development -- intergrax applications agents platform_proofs`) and on `05549f9acb` (the replay lineage head of the accepted counters, checked out in a throwaway worktree). Both runs call `build_closed_world_report()` of `tests/qualification/compat_x/_compat_x_closed_world.py` from the tree they run in; the `05549f9acb` run reproduces the accepted values exactly (7356 raw candidates, 5325 exclusions, 2031 semantic surfaces, 1933 legacy-projected surfaces). Scripts and dumps are session artifacts under `.tmp/session/041l/`.

## Problem

Ten tests of `tests/qualification/compat_x/` are red on `63005e290` (one pytest run over the six files that pin these counters: 10 failed, 107 passed, log `.tmp/proof/pdc-x-041l/compat_six_before.log`):

| Test | File | Why it is red |
| --- | --- | --- |
| `test_cx_r3_r1_combined_gate_passes_on_current_head`, `test_cx_r3_r1_probe_m_sanctioned_constructor_passes_classification`, `test_cx_r3_r1_compat_current_head_inventory_reconciled`, `test_cx_r3_r1_r3_candidate_semantic_identity_and_version_literal`, `test_cx_r3_r1_probe_q_equal_count_substitution_fails_inventory`, `test_cx_r3_r1_probe_r_hidden_removal_fails_inventory`, `test_cx_r3_r1_probe_s_balanced_add_remove_fails_inventory` | `test_compat_x_event_evolution_r3_r1_gates.py` | they read the pinned counters `ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES` (7356), `REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES` (7355) and `REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256` (`9dfd146f…`) (`assert 7417 == 7356`, `assert 7416 == 7355`, fingerprint mismatch) |
| `test_r5_p0_closed_world_gate_passes_on_current_head` | `test_compat_x_deprecation_r5_p0_gates.py` | one violation: `unexpected added shim paths: intergrax/agent_distribution/admin_service.py, intergrax/agent_distribution/runtime_revision.py, intergrax/applications/_shared/mcp_host_mount.py, intergrax/runtime/security/security_audit_trail.py` |
| `test_cx_r1_r3_p0_inventory_reconciliation` | `test_compat_x_versioning_r1_r3_identity_gates.py` | pinned `ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES`/`_EXCLUSIONS`/`_SEMANTIC_SURFACES` (lines 115 to 116) |
| `test_cx_r1_r5_closure_mechanical_inventory_and_delta` | `test_compat_x_versioning_r1_r5_semantic_identity_reconciliation_gates.py` | pinned counters (lines 44 to 48) |

The six files also contain `test_compat_x_event_evolution_f35_security_audit_trail_site.py` and `test_compat_x_deprecation_r5_shim_reconciliation_pure.py`; all their tests pass. The combined R3-R1 gate at `63005e290` reports six violations, all counter or fingerprint mismatches (raw 7417 vs 7356, exclusions 5357 vs 5325, semantic surfaces 2060 vs 2031, remainder 7416 vs 7355, remainder fingerprint). The "unclassified `RuntimeEvent` writer in `security_audit_trail.py:198,213`" that `PLAN_T-IGX-036-Z.md` (Problem 4) listed is no longer reported.

**Pin sites** (all read in code): `tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py` lines 39 to 44 (`REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES`, `REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256`, `ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES`, `ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES`, `ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES`, `ACCEPTED_CURRENT_HEAD_EXCLUSIONS`), and the accepted shim delta fixture `tests/fixtures/compat_x/accepted_shim_delta_replay_x_05549f9.json` (loaded by `load_replay_x_accepted_shim_delta`; `test_compat_x_deprecation_r5_p0_gates.py` lines 95 to 100 compare the production delta with it exactly). `ACCEPTED_P0_*` (the historical 7294 inventory) is not moved by this ticket.

## What moved each counter (measured, with evidence)

Raw inventory: 7356 at `05549f9acb` → 7417 at `63005e290`: 64 candidate ids added, 3 removed (net +61). No candidate of kind `defect.persisted_without_version` was added (6 at both commits), and `unclassified_candidate_ids` is empty at `63005e290` (the 64 additions are all resolved by the existing classification rules, none is an unclassified remainder).

| Counter | Pinned (replay lineage `05549f9acb`) | Measured at `63005e290` | Delta |
| --- | --- | --- | --- |
| `raw_candidates` (`ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES`) | 7356 | 7417 | +61 (+64, −3) |
| remainder without the R3 id (`REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES`) | 7355 | 7416 | +61 |
| remainder fingerprint (`REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256`) | `9dfd146f3b50bd56a8b997664569ac538396d322ceb21eb6682aa75a862716fa` | `8461d44eed400e7e51698dfa366fa4caaa9c8247e1cb73e54346805561aec45a` | changed with the set |
| `semantic_surfaces` (`ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES`) | 2031 | 2060 | +29 (+30, −1) |
| legacy-projected surfaces (`ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES`) | 1933 | 1963 | +30 |
| `exclusions` (`ACCEPTED_CURRENT_HEAD_EXCLUSIONS`) | 5325 | 5357 | +32 (+68, −36; 60 and 36 are wire-site exclusions renumbered or added, 8 are public exports of `agent_distribution/__init__.py`) |
| `compat.shim` candidates | 498 (474 historical + 25 accepted − 1 removed) | 502 | +4 (the four paths below) |

The 64 added and 3 removed raw candidates, grouped by the product code that owns them (`git log 05549f9acb..HEAD -- <path>` names the commits; every group is a legitimate, ticketed change):

| Group | Added | Removed | Owning change (introduced or last changed by) |
| --- | --- | --- | --- |
| D1 agent lifecycle: `agent_distribution/{__init__,admin_service,agent_manager_command_facade,control_plane_governance,upgrade_admission}.py`, `skills/host_lifecycle.py`, `contracts/agent_contract_meta.py`, `applications/_shared/durable_registry_projection_input_store.py` | 18 | 0 | B-D1 `bc7739c07` (T-IGX-036-J), B-P6-M `4dbe68452` (T-IGX-036-U), repair `f38a91f14` (T-IGX-041-A, PR #574) |
| D4 hosting: `applications/_shared/lkw_hybrid_daemon_wiring.py` | 1 | 0 | B-D4 `145b49bad` (T-IGX-036-R), `cc1c0efb6` (T-IGX-036-T) |
| D5 evaluation: `contracts/evaluation_contracts.py`, `runtime/evidence/eval_evidence_contracts.py`, `runtime/architecture/{online_evaluation_registry,evaluation_automation}.py` | 21 | 2 | B-D5 `85a8c441d` (T-IGX-036-O), repair `c076c7d0b` (T-IGX-041-E, PR #568) |
| D6 security: `core/plugins/admission.py`, `runtime/security/security_audit_trail.py` | 3 | 0 | B-D6 `adaaa374e` (T-IGX-036-P), `9ed0ee4f5` (audit trail through a validating port) |
| D7 scheduling: `runtime/long_running/{recurrence,scheduled_resume,postgresql_task_checkpoint_store}.py` | 5 | 0 | B-D7 `52f35a41f` (T-IGX-036-L), repairs `f01721ac6` and `ce6fbf67c` (T-IGX-041-G, PR #570) |
| D8 artifacts: `contracts/{artifact_ref,task_artifacts}.py`, `applications/contracts/application_artifacts.py`, `runtime/nexus/artifacts/{context_reference,sqlite_artifact_store}.py` | 8 | 1 | B-D8 `92ee26bf3` (T-IGX-036-Q), `97a5187b2` (T-IGX-036-T), repair `03844eda9` (T-IGX-041-H, PR #572) |
| MCP-SERVER-X: `applications/_shared/mcp_{host_mount,nexus_server,server_engine,wire_errors}.py`, `contracts/mcp_server_definition.py` | 8 | 0 | MCP-SERVER-X P2 to P4 and MRC (`e21453caf`, `bee38a76b`, `a21a6629a`, `a45d0df1b`, `16e376ea8`, `88419eb1c`, `3ba83b29b`, `420f42a6b`, T-IGX-038-*) |
| total | **64** | **3** | |

By kind, the 64 additions are 26 `wire.persistence`, 16 `migration.mechanism`, 8 `public.export`, 7 `class.field.version`, 4 `compat.shim`, 3 `version.constant`. The 3 removed ids are `class.intergrax.applications.contracts.application_artifacts.py::RunArtifactBundle.schema_version` (the field moved with the D8 contract de-duplication) and two `wire:` sites of `online_evaluation_registry.py` (`model_dump_json:2`, `model_validate:1`) whose numbering changed with the D5 commits. The surface delta is 30 added (10 version fields or constants, 16 migration mechanisms, 4 shims) and 1 removed (`semantic.class.field:…application_artifacts.py:RunArtifactBundle:schema_version`, the same field). The exclusion delta is wire-site and export exclusions of the same files.

**The four new shim paths are heuristic hits, not compatibility shims.** `build_compatibility_candidate_context` (`_compat_x_ast_signals.py:520` to `546`) records a signal for every path marker and for every case-insensitive source substring of `_COMPAT_SOURCE_SUBSTRINGS`, and any module with a signal becomes a `compat.shim` candidate (`is_compatibility_adapter_candidate`, line 547; `_shim_candidates_for_module`, `_compat_x_closed_world.py:318`); for the four new paths the primary signal is `source.contains:adapter` and the class is `TRANSLATION_ONLY`: `agent_distribution/admin_service.py:194` and `agent_distribution/runtime_revision.py:18` import `intergrax.llm_adapters.contracts.llm_profile`, `applications/_shared/mcp_host_mount.py:68` has the word in a docstring ("canonical harness adapter"), `runtime/security/security_audit_trail.py:34` imports `evidence_persistence_adapter`. The 25 paths of the accepted fixture are the same class of hit: 19 `source.contains:adapter`, 3 `source.contains:translate`, 2 `path.marker:_legacy`, 1 `path.marker:_adapter` (measured). They are accepted into the fixture as classified heuristic hits; the heuristic itself is not changed (see Out of scope).

**No counter increase is a defect:** the added candidates are the persistence calls, version fields, migration mechanisms and exports of ticketed product code; none is unclassified and none is a `defect.persisted_without_version`.

## Deliverables

Pinned tests first (red on the current pins, green after the re-pin), then the pin edits.

1. **New fixture** `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`, generated by a session script (not committed) from the two closed-world runs: `baseline_lineage_head_sha` (`05549f9acb9ecedf910c7470d442bba5d0a27337`), `measured_at_sha`, `previous` and `repinned` counter blocks (the table above), `added_candidate_ids` and `removed_candidate_ids` with `group`, `owning_change` and `kind`, `added_surface_ids` and `removed_surface_ids`, `added_exclusion_ids` and `removed_exclusion_ids`, and `shim_heuristic_hits` (the four paths with `signal`, `class`, `evidence_line` and `classification: "heuristic hit, not a compatibility shim"`).
2. **New test file** `tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py` (all red before the pin edits):
   - the pinned constants equal `previous + added − removed` of the fixture for raw candidates, remainder, surfaces, exclusions and legacy-projected surfaces (arithmetic, red today);
   - every id of `added_candidate_ids` is in the current closed-world inventory, every id of `removed_candidate_ids` is not, and the 25 original accepted shim paths plus the fixture's four equal `load_replay_x_accepted_shim_delta().added_paths`;
   - each `shim_heuristic_hits` entry reproduces: `_shim_candidates_for_module(path, source)` returns a candidate with the recorded signal and class (no full closed-world run);
   - the evidence guards: `unclassified_candidate_ids` is empty and the count of `defect.persisted_without_version` candidates equals its recorded value (6), so a later regression cannot hide inside a re-pinned count;
   - negative tests: a report with one of the 64 added candidates removed fails `validate_current_head_compat_closed_world_inventory` (hidden removal); a report with one extra synthetic candidate fails it; a balanced substitution (one added id replaced by a synthetic id, same count) fails it with a fingerprint violation; `shim_delta_accepted_r5_gate_violations` against the replay fixture fails for a fifth unexpected shim path, for a missing one of the four, and for a balanced substitution.
3. **Pin edits.** `_compat_x_semantic_identity_reconciliation.py` lines 39 to 44 take the measured values of the table, with a comment naming ticket T-IGX-041-L, `measured_at_sha` and the previous replay-accepted values; the constants keep their names (renaming would change the importing test files) and `REPLAY_X_COMPAT_LINEAGE_HEAD_SHA` stays the replay lineage; a new constant `PDC_X_P7_REPIN_MEASURED_AT_SHA` records the measurement commit. `tests/fixtures/compat_x/accepted_shim_delta_replay_x_05549f9.json` gets the four paths in `added_paths` and a `repin_additions` object (ignored by the loader, which reads `baseline_source_sha`, `added_paths` and `removed_paths`) naming the classification; `freeze_label` and `disposition` state the re-pin.
4. **Freshness rule.** The implementer re-measures at the `development` tip before editing. If a product path changed since `63005e290`, the numbers, the fixture and this table are re-derived, the change is named in the checkpoint, and the differences are explained group by group; an increase that cannot be attributed to a ticketed change, an unclassified candidate or a new `defect.persisted_without_version` candidate stops the batch and returns to the orchestrator and the Operator.

Design constraints: no gate weakened, narrowed, skipped or xfailed; every pin is moved to the exact measured value and bound to an exact id list by the fixture test (counts never stand alone); probes P, Q, R and S of the R3-R1 test file keep asserting hidden removal, equal-count substitution and balanced add and remove against the new pins; no classifier, heuristic or discovery rule is edited.

## Write set

`tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py`, `tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py` (new), `tests/fixtures/compat_x/accepted_shim_delta_replay_x_05549f9.json`, `tests/fixtures/compat_x/repin_pdc_x_p7_delta.json` (new)

Nothing under `intergrax/`, `agents/`, `applications/`, `scripts/`, `.github/` or `governance/` changes in the implementation pull request. A need to write elsewhere (another test, a classifier, the discovery code) stops the batch.

## Protocol

1. Re-measure (Freshness rule), write the fixture and the new tests first, and show the new tests red (log under `.tmp/proof/pdc-x-041l/`).
2. Edit the pins, then one `uv run pytest` over the whole `tests/qualification/compat_x` directory (about 14 minutes for the six pin files alone; run it in the background); ruff and pyright on the touched Python files.
3. Commit with exactly one `Ref: governance/tickets/T-IGX-041-L.md` trailer and no AI attribution; set the upstream to `origin/development` before the first push, push with an explicit refspec, open one pull request against `development`, never merge. The implementer never writes `Approved:` lines.
4. **Ordering.** `tests/qualification/compat_x/` and `tests/fixtures/` are not on the executable-identity allowlist (`scripts/qualification/verify_ebh7_p03_immutable_evidence.py` lines 19 to 37), so this pull request merges **before** the EBH-7 pin `P` is taken (`DELEGATION-PDC-P7-REPIN.md`, "Ordering").

## Acceptance

- The ten red tests of the table are green on the branch; the whole `tests/qualification/compat_x` directory is green or its remaining reds are named with a cause that is not this change.
- The new tests were red before the pin edits and are green after; each negative test fails if its guard is removed.
- The fixture accounts for every added and removed candidate, surface and exclusion id and for the four shim paths; the arithmetic of the pins is tested.
- No file outside the write set changed; no gate weakened; no classifier or heuristic changed.
- **Status statements.** This ticket is not a certification verdict. It changes tests and fixtures only, so it adds no product-path change and the IIAC activation stays stale for the earlier product changes (known red class, until the P8-04 rebind); the EBH-7 identity gate stays red until the final re-pin records the replay (known red class).

## Test contract

```text
uv run pytest tests/qualification/compat_x -q
uv run ruff check tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run pyright tests/qualification/compat_x/_compat_x_semantic_identity_reconciliation.py tests/qualification/compat_x/test_compat_x_pdc_x_p7_repin_delta.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Changing the shim heuristic (`source.contains:adapter`), any classifier, discovery rule or gate meaning; renaming the pinned constants; the `ACCEPTED_P0_*` historical pins; any `intergrax/`, `applications/` or `agents/` file (in particular `applications/_shared/mcp_host_mount.py`, which the debt item names as a hotspot file).
- The EBH-7 re-pin itself (T-IGX-041-J), the F10 classifier entries (T-IGX-041-K), the IIAC activation and its rebind, OD signatures, FRZ changes, certification.

## Reopen conditions of the debt item, as they stand

`PLAN_T-IGX-036-Z.md` D1 lists why the counters were not re-pinned earlier: (1) the pins are accepted facts of a closed stage — the replay-accepted values stay in the new fixture as `previous`, no history is rewritten; (2) the accepted shim delta stays with MCP-SERVER-X P7 — see "Operator decisions needed"; (3) `mcp_host_mount.py` under the `_shared` hotspot — it is not edited, its shim hit is classified in the fixture; (4) the unclassified `RuntimeEvent` writer — it no longer appears in the gate (the F-35 site test passes).

## Operator decisions needed

1. **The replay shim delta fixture is edited.** `PLAN_T-IGX-036-Z.md` D1 says the `roadmap_replay_x` accepted shim delta "stays with MCP-SERVER-X P7". The fixture `accepted_shim_delta_replay_x_05549f9.json` is loaded only by the `compat_x` modules and tests (grep over `tests/`), and `test_compat_x_deprecation_r5_p0_gates.py` lines 95 to 100 compare the production delta with it exactly, so the four paths can only be accepted by editing it (or by editing that test, which is a weaker change). `DELEGATION-PDC-P7-REPIN.md` part (a) delegates "the COMPAT-X counters of debt item Z-D1"; whether that includes this fixture is for the signer and the Operator. If it does not, this ticket is returned to the orchestrator before Phase 2.
