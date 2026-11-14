# PDC-P7-00 - ACT-N11 baseline independent audit, addendum 4 (G6)

Stage `PDC-X/P7-00/addendum_4` · policy `IIAC-2026-11-05.1` · auditor branch `p7-00-baseline-audit` · date 2026-11-14
Parents: the [first audit](PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md) (F-01..F-34), [addendum 1](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md) (F-35..F-40), [addendum 2](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_2.md) (F-41..F-48) and [addendum 3](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_3.md) (F-49..F-55).

| Item | Value |
|------|-------|
| Delta audited | `0fc8badd81c4c3e46a3d0ce262c08e8efba41898` .. `d02c399508c40b4a06d846410334d677f226c18d` (`mcp-rc`; development `4835bae8e` merged as `0cb5dd011`, development `83d939095` merged on top as `d02c39950`) |
| Current activation record | audited tree `324d9dba3` (unchanged in this range); previous pin `2d3eceb6c` |
| Product paths in the delta | 5 against `0fc8badd8`; the gate lists 6 against `324d9dba3` (the sixth, `ollama/register.py`, is unchanged since `0fc8badd8` and was audited in addendum 3) |
| **Addendum verdict** | **PASS_SCOPED** (IIAC manifest value `CONDITIONAL`) |
| **Combined verdict** (first audit + addenda 1-4) | **PASS_SCOPED** |
| New findings | 0 HIGH · 1 MEDIUM · 0 LOW · 5 INFO (F-56..F-61) |
| Product change required for the ACT-N11 rebind | none; one decision for the orchestrator/operator (F-56) |

Audit only: no product, test, ticket, activation-record or ruleset edit; the only merges are `origin/mcp-rc` into my own audit branch (`0cb5dd011`, then `d02c39950`), no PR, no rebind, no `Approved:` line; only `p7-00-baseline-audit` is pushed.

## 1. Verdict

The repairs for F-49 and F-41 do what the handoff asks. The `openai` row now dispatches on `integration_category` exactly like `ollama`, the Tier-3 root `applications/` has its own impact rule, and both repairs are pinned by tests that kill every non-equivalent mutant I tried. Nothing of the delta adds a credential, endpoint, tenant, process or network surface; one resolve result of 201 changes; no check is dropped for any tracked file. F-49, F-50, F-41 and F-09 are resolved; the whole `pdc_x` suite exits by itself. The 23 MCP-RC paths are byte-identical to `b0e19bf61` and the merge of `d02c39950` changed no product path.

The addendum is scoped, not clean, for one reason that the orchestrator did not announce: **deleting the unused `embedding_provider/ollama/manifest.py` turns a closed-stage replay gate red** (`roadmap_replay_x ... test_b6_03_scoped_disposition_valid_at_head`, green at `0fc8badd8` and at the pin, F-56). It needs a decision, not a product change.

## 2. Scope verified from git

- `git diff --name-only 0fc8badd8 0cb5dd011 -- intergrax agents applications` is five paths: `embedding_provider/openai/register.py` (+30/-3, F-49), `platform/enterprise_audit/impact_map.py` (+23/-1) and `data/changed_since_impact_map.v1.json` (+10, F-41), `embedding_provider/ollama/manifest.py` (deleted, -17, T-IGX-036-Z) and `applications/local_workspace_application/tests/test_run_artifact_metadata.py` (+8/-1). `embedding_provider/ollama/register.py` is not in it: unchanged since `0fc8badd8`. Against `324d9dba3` the set is the announced six.
- `git diff 0cb5dd011 d02c39950 -- intergrax agents applications` is empty, so every product finding below holds for `d02c39950`.
- The 23 MCP-RC paths are byte-identical to `b0e19bf61`.
- No path under `scripts/governance`, `scripts/maintenance`, `scripts/enterprise_audit`, `agent-env`, no activation record, G0 matrix, `pyproject.toml` or `uv.lock` changed. Two non-product exceptions to the statement "no governance or evidence path changed" (F-57): `.github/workflows/unit-tests.yml` (EBH-7 default pins) and `scripts/qualification/pdc_x_wbs.py`; evidence changes are docs only (certifier preflight addendum 2, evidence integration).
- `verify_iiac_policy_activation` on the merged tree fails with one `stale audit after material governance change` line (the protocol, see section 6) and exactly six `product path changed after the audited SHA` lines, the six announced paths.
- **Combined count:** `git diff --name-only 2d3eceb6c 0cb5dd011 -- intergrax agents applications` is **161** paths: the 158 of addendum 3 plus three new ones (`openai/register.py`, the deleted `ollama/manifest.py`, the `local_workspace` test); `impact_map.py`, its JSON and `ollama/register.py` were already counted.
- `origin/mcp-rc` was merged twice; the only conflict, the addendum 3 manifest, was resolved to the pin-only version on `mcp-rc` (three publication fields differ).

## 3. The delta, path by path

| Path | Result |
|------|--------|
| `openai/register.py` (F-49) | Same shape as `ollama/register.py`: `_dispatching_factory(owning_factory)`, embedding category builds `create_openai_embedding_provider_integration(enabled=bool(...))`, every other value goes to the owning factory with unchanged keyword arguments; the wrapped entry is registered with `override=True` after the augmentation. Added lines hold only imports of `dataclasses.replace`, `typing`, `IntegrationFactory`, the openai embedding bundle and `register_integration`. No credential, endpoint, tenant or network handling: the embedding branch resolves without `OPENAI_API_KEY` while `managed_retrieval` still fails closed (probe). `enabled` is coerced with `bool()` and other keywords (`base_url`, `credential_ref`, keys) are dropped, as for `ollama` (F-51, open, catalog-only integration). Of 201 (slug, category) pairs exactly one result changes: `openai|embedding_provider` (a `TypeError` before, the embedding integration now). Revision generation and digest behave as for `ollama`; second registration, missing owning row and sink isolation fail or behave as before. |
| Sibling rows `hf`, `vllm`, `llama_cpp` | Each owns its single row through `register_from_manifest` with its own manifest and slug; they share no slug with another category, so there is no augment and no dispatch to add. The only multi-category rows in the full catalog are `ollama` and `openai`, and both are now wrapped. All three resolve unchanged in the 201-pair matrix. |
| `ollama/register.py` | Unchanged since `0fc8badd8`; behaviour identical in every probe at both trees. |
| `impact_map.py` + SSOT JSON (F-41) | New table `PDC_TIER3_ROOT_IMPACT_RULES` with one key, `applications`, mapped to eight existing manifest ids (the same eight as the `intergrax/applications` rule), added to the generator loop; the longest-prefix rule keeps `applications` and `intergrax/applications` apart. The SSOT gains exactly one key (none removed or changed) and equals the generator output (`--emit-ssot` equality test and my probe). 0 of 19472 tracked files lose an impact id, 1016 gain ids. The XD-12 case and the `p6_impact` coverage test are byte-identical to the versions of addendum 2 and now pass on this tree, so they were not narrowed; the eight `applications/*/mcp/server.py` modules select the eight ids. The manifest stays at 85 checks. |
| `ollama/manifest.py` deleted | No importer in code, tests or scripts; no catalog reference (the catalog row comes from the model-serving manifest); env prefix `INTERGRAX_OLLAMA_EMBEDDING` appears nowhere else. Remaining mentions: the G0 impact matrix (two entries; `verify_iiac_g0_inventory_completeness` PASS and `test_iiac_g0_inventory` pass), docs, plans, the WBS and the frozen PRE-ENT list of `roadmap_replay_x`, which is what turns F-56 red. |
| `local_workspace` test | Adds `authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate()` to four `TaskResult(...)` calls and reorders imports; no assertion or fixture value changes; matches section 2 of the plan of T-IGX-036-Z (stale fixture); passes. |

The other test changes of the range (EBH-7 evidence and closure gates, the `pdc_x` leak guard, `xd_03`, the stale-red repairs, the new pins) were read: no assertion is weakened (F-60).

## 4. Pins, mutation, and the claim "F-41 tests pass unnarrowed"

- F-50: the pins for `enabled` default and pass-through, "the embedding category does not call the owning factory", keyword pass-through and wrapper depth exist for both rows (`p6_ecapfix`). Mutation sample: 11 mutants per row, **20 killed**, 2 survive (the equivalent string-comparison mutant); the four survivors of addendum 3 are all killed.
- Tier-3 rule: 8 mutants, 7 killed (drop table, reroute, shrink, trailing slash, unknown id, narrowed to one MCP app, tier-0 rule renamed); the survivor is the weak catch-all `intergrax/runtime` prefix, the known F-44 and outside the delta.
- F-41: `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check` and `test_every_pdc_touched_product_module_maps_to_at_least_one_check` pass in the one-process run and in the whole-directory run. Whether they also pass on development alone was not run here.

## 5. Own numbers

| Run | Result |
|-----|--------|
| One pytest process, 92 entries (MCP-RC contract, `p6_ecapfix`, `reg_p`, `p6_impact`, XD-12, `ent_audit_x`, `evol_x`, `roadmap_replay_x`, EBH-5, integration catalog/provider unit tests, rag bootstrap, the changed test files, F-35 status files) | 236 failed, 5833 passed, 176 skipped, 1 xfailed (known), 3190 s, exited by itself; no MCP test failed |
| The 236 ids in short-path checkouts: head / `0fc8badd8` / pin | 227 / 226 / 224 failed in isolation |
| Attribution | 1 new since `0fc8badd8` (F-56); 10 more new since the pin (F-35, unchanged); 216 red at the pin as well; 9 order dependent |
| EBH-7 directory alone | head 14 failed / 365 passed; `0fc8badd8` 16 / 341; none head-only |
| Whole `tests/qualification/pdc_x` directory | 975 passed, 2 skipped, exited by itself with exit 0 in 147 s (F-09 exit proof) |
| IIAC governance unit tests at `d02c39950` | 73 passed, 2 failed (both assert that `--require-in-force` passes; red by design before the rebind), 1 skipped |
| Mutation samples | wrappers 20 killed / 2 equivalent survivors; tier-3 rule 7 killed / 1 known survivor |
| ruff, pyright, tier and boundary gates, `uv lock --check` | clean; `check_intergrax_no_applications_imports` exit 1 with the same output hash as every earlier audit |
| A first one-process attempt that included `ebh_7` and `p6_debt` | aborted after 50 minutes; log kept; those parts were run separately |

## 6. The governance delta of `d02c39950` (T-IGX-040-B)

- `git diff 0cb5dd011 d02c39950` outside docs touches only `governance/tickets/T-IGX-040-B.md` (the delegate's approval block); `DELEGATION-PDC-P7-ACT.md` is new; the product diff is empty.
- **The protocol diff is exactly two lines** (numstat 2/2): line 4 and line 91 of `INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md`. Both replace the hard-coded status "NOT_IN_FORCE" (stale: the record says IN_FORCE) with a pointer to `IIAC_POLICY_ACTIVATION_RECORD.json` as the single source of truth. The sentences that `IN_FORCE` requires the fail-closed gate (`--require-in-force`) and a separate operator activation record, and that a certification session after the ACT-01 gate correction is required, are kept.
- No gate meaning is weakened: no script or test parses the protocol text for the status (git grep); the ACT-N05/ACT-N11 logic and the `--require-in-force` handling are unchanged; `iiac_activation_gate` lists the file in `MATERIAL_GOVERNANCE_PATHS`, so the gate reports it as a stale-audit trigger until the rebind, as the orchestrator expects. `verify_iiac_g0_inventory_completeness` passes.
- The rebind has to follow the protocol edit; my findings do not depend on the protocol text (F-61).

## 7. Status of earlier findings at `d02c39950`

| ID | Status |
|----|--------|
| F-49 openai row | **RESOLVED** |
| F-50 pins | **RESOLVED** (present for both rows, all non-equivalent mutants killed) |
| F-41 map rule | **RESOLVED** (rule present, tests pass unnarrowed) |
| F-09 exit hang | **RESOLVED** (whole `pdc_x` exits by itself) |
| F-52 two-write window | open, now also for `openai`; the handoff's single-write suggestion was not taken; soak 0 of 200 |
| F-51 `bool()` coercion, dropped keywords | open, also for `openai` (the config declares `credential_ref` and `base_url`, neither reachable) |
| F-35 ten COMPAT-X gates | open, unchanged |
| F-53 (216 reds at the pin) | open, unchanged |
| F-42, F-07, F-06 | resolved, unchanged |
| F-43, F-44, F-45 | unchanged (30 legacy ids; 1676 of 7045 `intergrax` files select no check; manifest 85 checks, no `pdc_x` entrypoint) |

## 8. New findings

Full evidence: `iiac_evidence/PDC-X/P7-00/addendum_4/iiac_audit_p7_00_addendum_4_findings.json`.

| ID | Sev | Finding | Verification | Disposition |
|----|-----|---------|--------------|-------------|
| F-56 | MEDIUM | The deletion of the unused ollama embedding manifest turns `roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py::test_b6_03_scoped_disposition_valid_at_head` red (`scoped_path_not_in_prod_diff`): the frozen PRE-ENT list names the file, which was added after the DUP-X pin and is now deleted, so it no longer shows in the pin-to-HEAD diff. The plan said the list "only names files"; its test contract did not run `roadmap_replay_x` | test run at 3 checkouts, code read | decision: test-side edit of the frozen list (closed-stage evidence) or accept; restoring the file would be a product path and a re-pin; no product change needed for the rebind |
| F-57 | INFO | `.github/workflows/unit-tests.yml` (EBH-7 default pins to `146e7acf1`) and `pdc_x_wbs.py` changed in the merged range; ebh_7 shows no new failure | code read, test run | accept; correct the statement "no governance path changed" |
| F-58 | INFO | `test_hf_execution_snapshot...` still fails in the shared venv: `sentence_transformers` raises on import, which `importorskip` does not treat as missing | test run | accept (environment) |
| F-59 | INFO | Positive: F-49/F-41 repairs add no surface, change one resolve result, drop no check, pins and mutation as above | probes, mutation, gates | accept |
| F-60 | INFO | Positive: stale-red repairs weaken no assertion; the manifest deletion leaves no importer or catalog reference | code read | accept |
| F-61 | INFO | Positive: the protocol edit is two header lines, no gate meaning weakened | code read, gates | accept |

## 9. Combined coverage statement

The five audits together cover the **161 unique ACT-N11 paths** that differ between the previous pin `2d3eceb6c` and the audited head `d02c39950`: 158 audited through addendum 3 plus the three first seen here. For the six paths the gate lists against the audited tree `324d9dba3` of the current record, five were audited in this addendum and `ollama/register.py` in addendum 3 (confirmed unchanged). The 23 MCP-RC paths are byte-identical to `b0e19bf61`. Every product or test change after `d02c39950` voids the audit for the changed paths.

## 10. What this addendum does not cover

- The eight per-domain audits IA-D1..IA-D8; CI contexts, `igx_gov ticket`, the full `tests/unit` tree, `tests/e2e/rag`, network and live-provider tests; P8 and FRZ (counter 225).
- Whether the F-41 tests pass on development alone, and whether the EBH-7 pins in the workflow match a future re-pin.
- The 216 failures that are red at the pin were attributed, not diagnosed.
- Whether the eight ids chosen for the Tier-3 root are the right ones; only existence, entrypoint existence, equality with the tier-0 rule and non-regression were verified. The manifest still has no MCP-specific check (F-44).
- Windows only; the shared `.venv` lacks `tzdata` and `duckdb`; a tzdata copy was used through `PYTHONPATH`.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the manifest are placeholders.

## 11. Open conditions for the orchestrator / operator

1. **F-56**: decide whether to repair the frozen PRE-ENT list (test-side) or accept the red gate; do not restore the manifest without another pin.
2. The rebind needs no product change; the gate lists exactly the six audited paths plus the protocol as a material governance change, so the protocol edit is already in `d02c39950` and the audit covers it.
3. F-35 stays a test-side item; F-52/F-51 are accepted soft spots for both rows.
4. Correct the claim "the merge changed no governance path" to "no governance script, record or evidence path; `.github/workflows/unit-tests.yml` pins changed" (F-57).

## 12. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/addendum_4/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_p7_00_addendum_4_findings.json`, `attribution_of_regression_failures.json`, `iiac_audit_session_manifest.PDC-X.p7_00.addendum_4.json`, `EVIDENCE_MANIFEST.txt`, `logs/`, `adversarial_scripts/`.
