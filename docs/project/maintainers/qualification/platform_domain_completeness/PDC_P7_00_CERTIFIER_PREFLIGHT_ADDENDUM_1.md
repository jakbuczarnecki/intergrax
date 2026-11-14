# PDC-P7-00 — certifier preflight, addendum 1 (covers audit Addendum 2, delta `19d552bee..324d9dba3`)

| Field | Value |
| --- | --- |
| **Parent** | `PDC_P7_00_CERTIFIER_PREFLIGHT.md` (verdict `CERTIFIER_PREFLIGHT_PASS_SCOPED` at `ae5623efb`), unchanged |
| **Role** | Certifier, preflight addendum. IMPLEMENTER != AUDITOR != CERTIFIER; same certifier session as the parent: `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| **Branch** | `p7-00-certifier`, fast-forwarded to `origin/mcp-rc` @ `fde55618379ff99779172e40214b5879d9d0810d` before this commit; my parent preflight `5c25d4d8f` is on it |
| **Date** | 2026-11-14 |
| **Verdict** | **`CERTIFIER_PREFLIGHT_PASS_SCOPED`** (scope limits in section 4) |
| **Machine-readable** | `iiac_evidence/PDC-X/P7-00/certifier/addendum_1/certifier_preflight_addendum_1_result.json`; logs, data, scripts in the same folder |

Not the activation certification, not a rebind, not stage closure, not a production-readiness claim. FRZ counter stays 225. This session edited no auditor file, product file, test, ticket, activation record, P8 file, ruleset or `Approved:` line.

## 1. Pin chain and ACT-N11 union

| Role | SHA | Verified |
| --- | --- | --- |
| Activation pin | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ancestor of the head; record unchanged since `6a6279a49` |
| Addendum 1 audited SHA | `19d552beeb6f0677a8bdc3b4ce65da15dd637c7b` | ancestor of `324d9dba3` |
| Second re-pinned `development` | `a156cbbd5ff521dc7e99dd02ab6bebd4ca234a73` (PR #525) | ancestor of `324d9dba3` |
| Addendum 2 audited SHA (`audited_implementation_sha`, `test_execution_sha`, `audit_execution_sha`) | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` | ancestor of the head |
| `development` `884c177ad` merged as `14e1322df` | `884c177adc366004929641d029bba8d129b0ecc8` | ancestor of the head; adds docs, tickets, `scripts/qualification/pdc_x_*.py`, `uv.lock`; no product path, no `MATERIAL_GOVERNANCE_PATHS` path |
| Addendum 2 evidence publication | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` | 27 files, all under `docs/`, blob-identical to `origin/p7-00-baseline-audit` @ `3fd4d9160` (27 of 27) |
| Pin-only edit + round 3 record | `fde55618379ff99779172e40214b5879d9d0810d` | changes exactly the addendum 2 manifest and `EVIDENCE_INTEGRATION_RECORD.md` |

With the gate's own `product_paths_changed_between` against the pin: **153** (first audit) → **156** (addendum 1) → **158** at `324d9dba3` and at the head. The delta `19d552bee..324d9dba3` is the two paths `intergrax/platform/enterprise_audit/impact_map.py` and `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`, both new to the set; **158 = 23 MCP-RC + 135 non-MCP**; `audit_scope.json` of addendum 2 equals the git delta. `git diff --name-only 324d9dba3 HEAD -- intergrax agents applications` is **empty** (product tree at the head equals the audited tree); the 23 MCP-RC paths are byte-identical to `b0e19bf61`; the RC tests and the mutation harness are byte-identical to `1b5f26f24`. `data/addendum2_checks.json`.

## 2. Checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Pin chain, 158-path union, empty product diff to the audited tree | PASS | section 1 |
| 2 | Addendum 2 manifest: both verifiers | PASS | `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff`: exit 0 (`logs/verify_iiac_manifest_addendum_2.log`) |
| 3 | Session ids distinct | PASS_SCOPED | implementation `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (contains the MCP-RC implementer id, batches J..V); auditor `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-...-ADDENDUM-2` (the same auditor session as the first audit and addendum 1, plus the suffix); certifier = this session; none equals the MCP-RC auditors or my earlier ids. `process_enforced` attestation |
| 4 | Digests equal the published files, before and after the pin-only edit | PASS | 4 of 4 match the files at the head, at `29d66ae3f` and on the auditor branch (`data/addendum2_checks.json`) |
| 5 | Integration record round 3 claims | PASS | base `14e1322df` product tree equals `324d9dba3`: **confirmed**; 27 files imported, all additions under `docs/`, 27 of 27 blob-identical to the auditor branch: **confirmed**; pin-only edit changes exactly `certifier_session_id`, `evidence_publication_sha`, `certification_publication_sha` and nothing else (also against the auditor branch's version): **confirmed**; the first-audit and addendum 1 manifests and evidence are untouched since `5c25d4d8f`: **confirmed**; the recorded verifier PASS: **reproduced** (row 2). The designation `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` is the one this session uses, so no further pin-only edit is needed |
| 6 | MCP-RC contract, one pytest process, list from Python, `PYTHONPATH` = my checkout | PASS | **481 passed, 1 xfailed (strict), 0 failed, exit 0, 503.48 s** (`logs/rc_contract_certifier_at_fde556183.log`) |
| 7 | Findings F-41..F-48 have a disposition | PASS_SCOPED | section 3: 8 of 8, no BLOCKER/CRITICAL; severity 0 HIGH / 2 MEDIUM / 2 LOW / 4 INFO as the auditor counts |
| 8 | Guardrails | PASS | no change to the activation record (since `6a6279a49`), P8, FRZ checklist, `.github`, `scripts/governance/`; no `MATERIAL_GOVERNANCE_PATHS` path changed after the audited SHA (`logs/guardrails_and_pin_git_checks.txt`); the test changes in the delta are one new test directory (`p6_impact`, +169 lines) and the removal of the 13-line strict `xfail` decorator of `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`; I re-read the diff (`git diff 19d552bee 324d9dba3 -- tests`): only the decorator is removed, the test body is untouched (**F-47 confirmed**) |
| 9 | Activation gate at the head | FAIL, expected | by construction: 158 `ACT-N11` paths until the rebind (not re-run; the governance suite below shows the gate tests red for that reason) |

## 3. Findings F-41..F-48 and the older open items

Dispositions as recorded by the auditor (`iiac_audit_p7_00_addendum_2_findings.json`), first clause verbatim:

| ID | Sev | Disposition |
| --- | --- | --- |
| F-41 | MEDIUM | "ORCHESTRATOR_AND_OPERATOR_DECISION: either add rules for the applications/* roots to impact_map.py (a product path: voids ACT-N11 for it, needs another re-pin) or accept the red class on the integrated baseline until after the rebind; do not narrow the test (that would weaken a gate)" |
| F-42 | MEDIUM | "TEST_ONLY_FIX or catalog registration by the ebh_5 / integrations owner; route to the domain audit that covers LLM adapters / integrations (IA-D3)" |
| F-43 | LOW | "ROUTE_TO_ENT-AUDIT owner; separate ticket (a product path)" |
| F-44 | LOW | "ROUTE_TO_ENT-AUDIT owner and to IA-D6 (runtime security) / IA-D1; accept for the rebind" |
| F-45 | INFO | "ACCEPT; record as an explicit residual in the domain audits" |
| F-46 | INFO | "ACCEPT_AS_ATTRIBUTED; the order dependence extends F-38" |
| F-47 | INFO | "ACCEPT" |
| F-48 | INFO | "ACCEPT" |

**Open items, verbatim** (new in addendum 2, then earlier ones carried): F-41, F-42, F-43, F-44 as above; **F-35** "TEST_ONLY_FIX / classification by the COMPAT-X owner ... to be read together with F-06/F-07 as 'CI contexts red after merge'"; **F-09** "RECOMMEND_FIX (test owner)"; **F-07** "regenerate the inventory (scripts/qualification/pdc_x_inventory.py --write) by the owner" (see status below); **F-38**, **F-04/F-37**, **F-05** as in the parent; MCP-RC carry-overs (AF-05/AF-16 owner, RA-02..RA-04).

**Status of the red tests at the head, from my own runs** (one pytest process, 166 tests, `logs/status_f07_f35_f41_f42_at_fde556183.log`; EBH-5 gate files, F-07, F-35, F-41): 147 passed, 19 failed.

- **F-41: confirmed red on `mcp-rc`** (2 tests: `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`, `test_every_pdc_touched_product_module_maps_to_at_least_one_check`). **The claim "green on `development` alone" is now verified** (the auditor listed it as not run): on a detached checkout of `origin/development` (`434249534`; it holds the same product and test tree as `884c177ad`, later PRs add only tickets, plans and `scripts/qualification/pdc_x_wbs.py`) both tests pass (`logs/status_on_development_alone_884c177ad.log`). At that checkout 91 of 95 passed; the 4 failures are the COMPAT-X `r3_p0` gates (F-35), so **F-35 is also red on `development` alone**.
- **F-35: red** (10 COMPAT-X event-evolution gates new since PDC-X plus the same 7 red at the pin). Unchanged.
- **F-07: no longer red at the head.** `test_pdc_x_inventory.py` passes; `PDC_CLOSED_WORLD_INVENTORY.json` and `scripts/qualification/pdc_x_inventory.py` changed in the `884c177ad` merge. The auditor's "OPEN" refers to `324d9dba3`; I record F-07 as **resolved at test level at `fde556183`** (not a closure by an auditor).
- **F-42: confirmed** (2 `embedding_provider` nodes, `UnknownIntegrationError: Integration slug 'ollama' is not registered in the catalog`); the 3 `relational_store`/duckdb nodes of the same file fail in my environment only because `duckdb` is not installed (the CI extra `dev-ci` declares it).
- **F-09** is not exercised by my runs.

## 4. Do the red tests belong to the five CI contexts? (read from the workflows and selections; partly run)

Workflows read: `.github/workflows/unit-tests.yml` (`ci-smoke`, `governance-fast`, `evol-x-b4-preflight`) and `enterprise-audit.yml` (`ent-audit-fast`, `ent-audit-pr-gate`); selection code `scripts/ci/run_regression_gate_ci.py`, `scripts/ci/run_ci_smoke_pytest.py`, `tests/unit/conftest.py`, the audit manifest and `run_enterprise_audit.py`.

| Context | What it runs | Red items inside it | What I could / could not determine |
| --- | --- | --- | --- |
| `ci-smoke` | `check_environment_conformance --profile ci`; `run_ci_smoke_pytest.py` = 2 directories and 25 files with marker `ci_smoke`; two boundary scripts | none of F-07, F-09, F-35, F-41, F-42 (checked by prefix against the selection) | **Could not run it**; I cannot say it is green, only that the known red tests are not in its selection |
| `governance-fast` | `validate_policy_registry.py`, then `pytest tests/unit/governance/` | the real activation-gate tests (ACT-N11 until the rebind); F-07 lived here (`test_pdc_x_inventory.py`) | My run (`logs/ci_governance_fast_probe_at_fde556183.log`): policy registry OK; **155 passed, 2 failed, 15 errors, 1 skipped** (462 s). The 2 failures are `test_canonical_policy_is_in_force_and_verifies` and `test_n12_policy_is_in_force_only_with_valid_activation_evidence`, the ACT-N11 effect the rebind removes. The 15 errors are fresh-clone fixtures failing at `git checkout` ("cannot create directory ... freeze_normal_at_0ad6511aca...") on this Windows workstation, a path-length effect I cannot attribute to the repository; **I cannot say whether they pass on the CI Linux runner**. F-07 passes here |
| `evol-x-b4-preflight` | two files, `test_evol_x_closure_preflight_b4.py` and `test_evol_x_b4_ruff_evidence_exemption.py` (`-n auto`) | none | Not run; reading shows none of the red areas is selected |
| `ent-audit-fast` | step 1: `pytest tests/qualification/ent_audit_x/{unit,integration,l3}`; step 2: `run_enterprise_audit.py run --mode FAST --non-certifying` (all **85** manifest checks carry the mode FAST) | **F-42**: check `VBN-REACHABILITY` has the entrypoint `tests/qualification/ebh_5/test_ebh_5_behavioral_package_c_v2.py` (whole file), which contains the two failing `embedding_provider` nodes; if the runner takes a failing `pytest_file` as a violation, this check, and so the FAST run, is red on the baseline. The manifest nodes of `ENT-A-REG-01/DOC-01/DOC-03` (`roadmap_replay_x`) passed in my run (the three `roadmap_replay_x` tests that fail are different nodes and red at the pin too) | Step 2 **not run end to end** (85 checks; the `test_mechanical_readiness` test alone did not finish in 15 minutes here). Step 1: my one full attempt ended silently around 52 % with two failures already printed and exit code 0 (`logs/ci_ent_audit_first_probe_died_silently_at_52pct.log`), so **I cannot give a result for step 1**. F-41 and F-35 tests are not in this context; F-42 is, by the manifest |
| `ent-audit-pr-gate` | `test_mechanical_readiness.py` and `test_task_id_registry.py` | none of the known red tests | `test_task_id_registry.py` passed; `test_mechanical_readiness_script_emits_fail_closed_report` (spawns `scripts/enterprise_audit/mechanical_readiness_check.py`) ran for more than 10 minutes here and was stopped (`logs/ci_ent_audit_mechanical_readiness_aborted_after_15min.log`), **no result** |

Summary: by selection, **F-41, F-35 and F-09 are in none of the five contexts**; **F-07** was in `governance-fast` and now passes at the head; **F-42 is inside `ent-audit-fast`** through `VBN-REACHABILITY`; `governance-fast` is red at least by the ACT-N11 gate tests until the rebind. The F-41 tests live in `tests/qualification/pdc_x`, which none of the five contexts selects as read; any other suite or gate that runs that directory (I did not search for one) would be red. I did not verify behaviour of GitHub Actions itself, only the commands it runs.

## 5. Verdict and scope limits

`CERTIFIER_PREFLIGHT_PASS_SCOPED`. Pin chain, 158-path union, the addendum 2 manifest, digests, the integration record's round-3 claims and the MCP-RC contract hold on my own runs. PASS applies only under these limits:

1. **F-41 is a pending operator decision turned into a pending change.** `OD-MCP-RC-04` §5 (recorded chat answer) says PDC-X adds an impact-map rule for the `applications/*` roots on `development` as a one-time exception to the freeze, then `mcp-rc` merges `development` again, the auditor adds addendum 3 for the changed map files and a certifier addendum follows. That rule is **not on `development` yet** (`origin/development` @ `3a7694cb7` has no product-path change after `884c177ad`). Editing `impact_map.py` and `changed_since_impact_map.v1.json` changes bytes of two paths already inside this audit: **addendum 2 is void for those two paths once the rule lands**, the union stays 158 paths, and this addendum then needs a successor.
2. **F-42 is red and sits in `ent-audit-fast`** (section 4); nobody owns the repair yet (the auditor routes it to an ebh_5/integrations owner and IA-D3). `OD-MCP-RC-04` §2 names only F-07, F-35 and F-09 for the PDC-X repair on `development`; F-42 and F-41 are not in that list.
3. **I could not determine** whether `ci-smoke`, `evol-x-b4-preflight`, `ent-audit-fast` step 1 and 2 and `ent-audit-pr-gate` are green, nor the 15 fresh-clone fixtures of `governance-fast` on a Linux runner. This addendum does not assert that the five contexts are green.
4. All limits of the parent preflight stay: 12 red PDC-X-only tests at the first baseline (now 10 COMPAT-X + the F-41 pair + F-42 pair), the 40 pin-red regression failures attributed but not diagnosed, IA-D1..IA-D8 not done, the unreproduced leads, the audits' PASS_SCOPED versus the gate's `PASS`, AF-05/AF-16 and RA-02..RA-04.
5. Further `development` product commits after `884c177ad` are none at my fetch (`3a7694cb7`: four docs/ticket PRs and `scripts/qualification/pdc_x_wbs.py`); the operator's "re-pin and freeze" is a declaration, not an enforced control.

## 6. What the activation certifier needs, revised for 158 paths

The parent's section 5 holds (gate checks confirmed against `iiac_activation_gate.py`). Changes and additions:

1. **Scope:** 158 ACT-N11 paths against the pin `2d3eceb6c` = 23 MCP-RC + 135 non-MCP, today; the "Widened scope" text of `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` must cite `324d9dba3` (or the later audited tree), 158, the four audit scope files (first audit, addenda 1 and 2, plus addendum 3 when it exists), the dossier verdicts and both certifier preflights. The runbook still says 125 non-MCP paths.
2. **Candidate pins** (proposal): `corrective_implementation_sha` = binding manifest `audited_implementation_sha` = the audited tree after the F-41 rule (today `324d9dba3`; the head's product tree is identical); `audit_execution_sha`, `evidence_publication_sha` at or after the commit that publishes the last addendum's evidence (today `29d66ae3f`; addendum 3 will move it); `certification_publication_sha` a later docs commit that is not `corrective`; all with the ancestry chain of the parent section 5, item 2. Product diff between `audited_implementation_sha` and `audit_execution_sha` must be empty (`--strict-product-diff`), and the product diff from `audit_execution_sha` to the PR merge head must be empty (ACT-N11), so **no `development` product commit may land after the pinned tree**.
3. **Bound verdict:** the operator recorded the P6 pattern in `OD-MCP-RC-04` §3 (recorded chat answer, not a signed line): the activation certifier authors a new `IIAC-GOV-1` binding manifest with `verdict = PASS` and states the scope limits, the open findings (F-41..F-44, F-35, F-09, F-42, F-07 status), the red-test caveat and the accepted conditions in `limitations` and in the decision text. Because the five audit manifests are `CONDITIONAL` (gate: ACT-N05), none can be bound as authored.
4. **Evidence paths:** all five bound artifacts (`audit_report_path`, manifest, scope, matrix, findings) must sit under `iiac_evidence/` (ACT-N14); the P7-00 dossiers under `platform_domain_completeness/` do not; the report bytes must equal those at `evidence_publication_sha`. This is a correction of the runbook (it names only the manifest).
5. **Findings register:** the bound `findings_path` needs a real `findings` array without an open BLOCKER/CRITICAL (ACT-N15). The gate function `_findings_errors` returns no error for the first-audit, addendum 1 and addendum 2 files of P7-00 (the last has 8 entries, MEDIUM at most); it fails for the MCP-RC re-audit file (no `findings` array).
6. **Sequence dependencies:** F-41 rule on `development` (one-time exception) → `mcp-rc` merges `development` → auditor addendum 3 → evidence integration (rounds 3 and 4 together, as `OD-MCP-RC-04` says) → certifier addendum 2 → activation line (signing delegate, renewed delegation, 24 h from the start of the signing session) → two-phase rebind → PR → five contexts green on the pinned head → merge → terminal verification.
7. **CI readiness:** F-42 needs an owner before `ent-audit-fast` can be expected green; F-35 and F-09 repairs are on `development` per the operator, but F-35 is still red on `development` (my run). Windows-only fixture errors in `governance-fast` need a Linux confirmation (CI run on the PR head).
8. All gate checks listed in the parent section 5 (binding keys, phase-1/phase-2 blobs, operator file lines, ACT-N18 text scan, handoff file contents, material governance paths) unchanged.

## 7. Remaining steps (orchestrator and operator)

1. PDC-X/ENT-AUDIT owner adds the `applications/*` impact-map rule on `development` (F-41); then `mcp-rc` merges `development`; auditor addendum 3; certifier addendum 2 (short).
2. Assign an owner for F-42 (ebh_5 embedding_provider / integrations catalog registration) and confirm that the owners of F-35 and F-09 repair on `development`.
3. Confirm the five CI contexts on the pinned PR head; treat my section 4 as a reading of the workflows, not as a CI result.
4. Operator-owned items unchanged: owner of the `[product]` xfail (`T-IGX-039-A`), RA-02..RA-04 and AF-05, D-10, P8 documentation owner (G-DOC-01..12), host composition (F-04, F-05, F-37).
