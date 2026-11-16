# PDC-P7-00 - ACT-N11 baseline independent audit, addendum 2 (G6)

Stage `PDC-X/P7-00/addendum_2` · policy `IIAC-2026-11-05.1` · auditor branch `p7-00-baseline-audit` · date 2026-11-14
Parents: [`PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md) (audit of `fa6ff1c96`, F-01..F-34) and [`PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_ADDENDUM_1.md) (delta to `19d552bee`, F-35..F-40).

| Item | Value |
|------|-------|
| Delta audited | `19d552beeb6f0677a8bdc3b4ce65da15dd637c7b` .. `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` (`mcp-rc`, merge of development `a156cbbd5`, the operator's second re-pin) |
| Activation pin | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| Product paths in the delta | 2 (PR #525 = T-IGX-036-V, commit `df136ef85`): `intergrax/platform/enterprise_audit/impact_map.py` and `.../data/changed_since_impact_map.v1.json` |
| **Addendum verdict** | **PASS_SCOPED** (IIAC manifest value `CONDITIONAL`) |
| **Combined verdict** (first audit + addenda 1 and 2) | **PASS_SCOPED** |
| New findings | 0 HIGH · 2 MEDIUM · 2 LOW · 4 INFO (F-41..F-48) |
| Product change required for the ACT-N11 rebind | none; one decision for the orchestrator/operator (F-41) |

Audit only: no product, test, ticket, activation-record or ruleset edit; no merge of anything but `origin/mcp-rc` into my own audit branch, no PR, no rebind, no `Approved:` line; only `p7-00-baseline-audit` is pushed.

## 1. Verdict

The two product paths of PR #525 only add checks to the CHANGED-SINCE selection: no check is dropped or rerouted for any tracked file (the two own-file rules lose three ids that are not manifest ids), the new rules name existing manifest ids with existing entrypoints, the shipped JSON equals the generator output, and no import, credential, endpoint or process/network surface is added. The removal of the strict xfail in `test_xd_12_evol_ent_audit.py` removes the decorator only; no assertion is weakened and no gap is hidden. The addendum is scoped, not clean, because (a) the repaired XD-12 case and the new `p6_impact` coverage test are **red on the integrated baseline**: eight `applications/*/mcp/server.py` modules (MCP-RC) are in the PDC range and have no impact rule (F-41, decision needed); (b) a wider regression selection than before exposed two further red EBH-5 tests that are red since PDC-X and belong to a floor check (F-42, identical at `19d552bee`); (c) the tests prove existence, not relevance, of the mapping (F-44), and 30 legacy ids still select nothing (F-43).

## 2. Scope verified from git

- `git diff --name-only 19d552bee 324d9dba3 -- intergrax agents applications` lists exactly the two announced paths (+459 / -12). Both are new to the audited set: `git diff --name-only 2d3eceb6c 324d9dba3 -- intergrax agents applications` = **158** paths (156 + 2).
- The 23 MCP-RC paths are byte-identical to `b0e19bf61` (`git diff` empty).
- No path under `scripts/`, `governance/` or `.github/`, no IIAC activation record and not the G0 matrix changed in `19d552bee..324d9dba3`. The only other changes are docs: the RC contract log, the evidence integration of my earlier audits (pin-only edits of the two manifests, `EVIDENCE_INTEGRATION_RECORD.md`) and `PDC_P7_00_CERTIFIER_PREFLIGHT.md`.
- Test files changed (the three the certifier flagged): `tests/qualification/pdc_x/p6_impact/__init__.py` (new), `.../test_p6_impact_map_coverage.py` (new), `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py` (-13 lines).
- My branch contains `origin/mcp-rc` through merge `bbe54afbb`; the two auditor manifests (add/add conflict because the integration published them separately) were resolved to the pin-only versions on `mcp-rc`; the six pin-only fields are the only difference from my earlier files. Product tree identical to `324d9dba3`.

## 3. What the impact map controls, and what the delta does to it

`select_checks_for_changed_since` returns the invariant floor (checks flagged `fast_mandatory`, all `TEN-*` and `VBN-*`) plus the ids of the longest matching prefix of each changed path (first match only) plus every check whose entrypoint is touched. A path with no prefix selects only the floor; paths under `intergrax/platform/enterprise_audit/` and `scripts/enterprise_audit/` fail closed when unmapped.

| Question | Answer (probe results in `addendum_2/logs/addendum_2_impact_map_probes.results.json`) |
|----------|----------|
| Does the delta drop or reroute a check for any path? | No. Over 19319 tracked files the new selection is a superset of the old one except for the two own-file paths, whose lost ids (`C024`, `C029`, `C072`) are not manifest ids; their new ids are `ENT-A-EVOL-01..03`. 37 prefixes are added, none removed; 5780 files gain impact ids. The longest-prefix-wins rule cannot shadow an older broader prefix (no file's old ids are lost to a new more specific prefix). |
| MCP modules | MCP-RC modules under `intergrax/` (`applications/_shared/mcp_*`, `contracts/mcp_server_definition.py`, `tools/mcp_result_bounds.py`) gain the generic applications/contracts/tools gates. The manifest has no MCP-specific check. The eight `applications/*/mcp/server.py` modules still map to nothing (F-41). |
| Identity / governance / authority modules | 113 of the 259 identity/auth/governance/policy/tenant python files gain ids; 146 (for example `fastapi_core/auth`, `runtime/governance`, `runtime/policy`, `runtime/human`) still select only the floor, as before (F-44). `runtime/agent_governance` gains `ENT-A-GOV-01..03`, `runtime/security` gains `ENT-A-SEC-*`, `TEN-AUTHORITY`, `TEN-CREDENTIALS`. |
| Is the shipped JSON the generator output? | Yes (probe P5, test `test_shipped_ssot_equals_generator_output`). |
| Do the new rules name only real checks? | Yes: 52 distinct ids, all in the 85-check manifest, all entrypoint files exist (probe P4). |
| Mutation sample (scratch checkout, control already red) | killed: delete a rule, legacy ids back, unknown id; **survived**: reroute `runtime/security` or `runtime/agent_governance` to an unrelated check, shrink `applications` to one id, add a weak catch-all `intergrax/runtime` prefix (F-44). |

## 4. The XD-12 test change

`git diff 19d552bee 324d9dba3 -- tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py` removes the 13-line `@pytest.mark.xfail(strict=True, reason=...)` block and nothing else; the body of `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check` and the other XD-12 cases are byte-identical. No assertion is weakened. The removal does not hide a gap: on the integrated baseline the test is now red (F-41). What was dropped with the decorator is the second half of the repair named in its reason (a `pdc_x` entrypoint in the manifest); the ticket keeps the manifest at 85 checks, so that half is out of scope by approval, but no record carries it as open any more (F-45).

## 5. Own numbers

| Run | Result |
|-----|--------|
| One pytest process, 274 entries: MCP-RC contract (39 files, list from Python) + `ent_audit_x` + `evol_x` + `roadmap_replay_x` + `p6_impact` + XD-12 + EBH-5 gate files + `test_pdc_x_*` + the pytest entrypoints of every check the new rules select | 84 failed, 1359 passed, 176 skipped, 1 xfailed, 1022 s; pytest exited by itself; no MCP test failed; the xfail is the known TRANSFERRED case; the three EBH-5 gate files pass |
| The 84 node ids in a short-path checkout of `324d9dba3` | 15 failed, 69 passed (order/path dependent, F-46) |
| The 15 persisting ids at `19d552bee` / pin | 13 failed + 1 xfailed / 9 failed, 2 passed |
| Mutation sample, 7 mutants | 3 killed, 4 survived |
| ruff check + format on `impact_map.py` and the three tests | clean |
| tier and boundary gates, immutable security audit trail script, `uv lock --check` | exit 0, except `check_intergrax_no_applications_imports.py` exit 1 with output hash identical to the first audit and addendum 1 (12 pre-existing untouched files) |
| `verify_iiac_g0_inventory_completeness` | PASS (17510 paths) |

## 6. Status of earlier findings touched by this range

| ID | Status at `324d9dba3` |
|----|----------------------|
| F-05 declared-open gaps | The last one (FINDING-8) no longer carries an xfail, but its test and the new coverage test are red on the integrated tree (F-41); its second repair half is not recorded (F-45) |
| F-06 | RESOLVED (EBH-5 gate files pass) |
| F-07 inventory tests | OPEN (same two tests; red at `19d552bee`, absent at the pin) |
| F-09 pytest exit hang | not reproduced here (xd_03 not in this selection); not shown to be fixed |
| F-35 COMPAT-X gates | not re-run; unchanged (the delta touches neither `security_audit_trail.py` nor COMPAT-X) |
| F-38 order-dependent tests | extended by F-46 |
| others | unchanged |

## 7. New findings

Full evidence: `iiac_evidence/PDC-X/P7-00/addendum_2/iiac_audit_p7_00_addendum_2_findings.json`.

| ID | Sev | Finding | Verification | Disposition |
|----|-----|---------|--------------|-------------|
| F-41 | MEDIUM | XD-12 finding-8 case and `p6_impact` "every PDC-touched module maps" are red on `324d9dba3`: 8 `applications/*/mcp/server.py` modules (MCP-RC changes inside the PDC range) map to no check; at `19d552bee` the XD-12 case was a strict xfail. Not run on development alone | test run, 3 checkouts | orchestrator/operator decision: add `applications/*` rules (product path, voids ACT-N11 for it) or accept the red class; do not narrow the test |
| F-42 | MEDIUM | `test_ebh_5_behavioral_package_c_v2.py` (entrypoint of floor check `VBN-REACHABILITY`): two `[integration.embedding_provider]` tests red since PDC-X (`Integration slug 'ollama' is not registered in the catalog`), green at the pin, identical at `19d552bee`; candidate origin commit `c3d5b7005` (B-REG-P), not diagnosed; three further tests red for missing `duckdb` (environment, red at the pin too) | test run, 3 checkouts | test-only fix / catalog registration; route to IA-D3 |
| F-43 | LOW | 30 legacy `C0xx` ids in the map are not manifest ids (12 platform paths incl. `changed_since.py`, `runner.py`, `validators.py`); the delta fixes only the map's own two files; mutation M5 shows the consequence for XD-12 | probe, mutation | ENT-AUDIT owner, separate ticket |
| F-44 | LOW | Tests prove existence and non-emptiness of the mapping, not relevance: 4 of 7 mutants survive; coverage partial (1676 of 7046 intergrax files, 146 of 259 governance/identity files, 16 of 37 MCP-named files have no impact id; no MCP-specific check exists) | probe, mutation | ENT-AUDIT owner, IA-D6 / IA-D1; accept for the rebind |
| F-45 | INFO | Only the first half of the repair named in the removed xfail reason was delivered (ticket scope: manifest stays at 85 checks); the residual is no longer recorded | code read | accept, record in domain audits |
| F-46 | INFO | 84 failures in the long run: 69 pass in isolation (probable long-worktree-path / MAX_PATH), 15 persist, 9 of them red at the pin; attribution table in `attribution_of_regression_failures.json` | test run | accept as attributed |
| F-47 | INFO | Positive record: the XD-12 diff is the decorator only; the new tests are strictly stronger; no assertion weakened, no gap hidden | code read | accept |
| F-48 | INFO | Positive record: the two paths add no boundary violation, credential, endpoint, process/network/dynamic-code surface or authority; selection only gains checks; SSOT = generator | code read, probes, gates | accept |

## 8. Combined coverage statement

The three audits together cover the **158 unique ACT-N11 paths** that differ between the activation pin `2d3eceb6c` and the audited head `324d9dba3`: 153 audited at `fa6ff1c96` (23 MCP-RC relied on the existing delta re-audit + 130 PDC-X), the 3 paths first seen in `19d552bee` (addendum 1; the other 3 of its 6 were already in the 153 and re-read at their final state), and the 2 paths first seen in `324d9dba3` (this addendum). Equivalently: 153 + 6 (3 overlapping) + 2 = 158. Every product or test change after `324d9dba3` voids the audit for the changed paths; a different re-pin voids it for the whole delta.

## 9. What this addendum does not cover

- The eight per-domain audits IA-D1..IA-D8; F-41..F-44 are routed to ENT-AUDIT and the domain audits.
- CI contexts / GitHub Actions, enterprise and nightly profiles, `igx_gov ticket`, the full `tests/unit` tree, EBH replays, network and live-provider tests; P8 and FRZ (counter 225); documentation-only changes.
- Whether `test_xd_12` and `p6_impact` pass on development alone (PR #525's own CI); the COMPAT-X, `agent_distribution` and memory suites (unchanged by this range, not re-run); the `pdc_x` suites other than XD-12 and `p6_impact`.
- Whether the specific check ids chosen for each rule are the right ones; only their existence, entrypoint existence and non-regression of selection were verified.
- The failures that are red at the pin were attributed, not diagnosed; the 69 order/path-dependent failures were not isolated.
- Windows only; the shared `.venv` lacks tzdata (F-34) and duckdb; a tzdata copy was used through `PYTHONPATH`.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the addendum manifest are placeholders.

## 10. Open conditions for the orchestrator / operator

1. **F-41** needs a decision before the baseline is read as green: rule for `applications/*` (a product change that needs another pin) or an accepted known-red class on the integrated baseline.
2. F-42, F-07, F-35 are test-only repairs, to be done before merge or CI contexts are read.
3. Route F-43 and F-44 to the ENT-AUDIT owner and the domain audits; the residual of F-45 to the domain audits.
4. The wider selection (`ent_audit_x`, `evol_x`, `roadmap_replay_x`, the entrypoints of the selected checks) should be part of later replays.

## 11. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/addendum_2/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_p7_00_addendum_2_findings.json`, `attribution_of_regression_failures.json`, `iiac_audit_session_manifest.PDC-X.p7_00.addendum_2.json`, `EVIDENCE_MANIFEST.txt`, `logs/`, `adversarial_scripts/`.
