# PDC-P7-00 — certifier preflight, addendum 3 (covers audit Addendum 4, delta `0fc8badd8..d02c39950`)

| Field | Value |
| --- | --- |
| **Parent** | `PDC_P7_00_CERTIFIER_PREFLIGHT.md`, `..._ADDENDUM_1.md`, `..._ADDENDUM_2.md` (all `CERTIFIER_PREFLIGHT_PASS_SCOPED`), unchanged |
| **Role** | Certifier, preflight addendum. IMPLEMENTER != AUDITOR != CERTIFIER; same certifier designation as the parent: `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| **Branch** | `p7-00-certifier`, fast-forwarded to `origin/mcp-rc` @ `d02c399508c40b4a06d846410334d677f226c18d` before this commit; my addendum 2 `0c79c9f65` is on it |
| **Audit under review** | Addendum 4 on `origin/p7-00-baseline-audit` @ `0d4758ca1a185f150c7e0c37ea5dbc1a6eea03cb` (read from that branch at my base; its evidence was integrated into `mcp-rc` later, section 9) |
| **Date** | 2026-11-14 |
| **Verdict** | **`CERTIFIER_PREFLIGHT_PASS_SCOPED`** (scope limits in section 7) |
| **Machine-readable** | `iiac_evidence/PDC-X/P7-00/certifier/addendum_3/certifier_preflight_addendum_3_result.json`; logs, data, scripts in the same folder |

Not the activation certification, not a rebind, not stage closure, not a production-readiness claim. FRZ counter stays 225. I edited no auditor file, product file, test, ticket, activation record, signed decision, delegation, binding manifest, handoff, P8 file, ruleset or `Approved:` line, and signed nothing. For one verifier run I copied the auditor's five addendum 4 files into my worktree from the auditor branch and removed them again; none was committed.

## 1. Pin chain, union and the audit's scope claims

With the gate's own `product_paths_changed_between` against the pin `2d3eceb6c` (`data/addendum4_checks.json`):

| Claim of the audit | Result |
| --- | --- |
| 161 unique ACT-N11 paths since `2d3eceb6c` | **confirmed**: 153 → 156 → 158 (`0fc8badd8`) → **161** at `d02c39950` and at the head; no path removed; 23 MCP-RC + 138 non-MCP; the 23 are byte-identical to `b0e19bf61` at the head |
| Three new paths | **confirmed**: `applications/local_workspace_application/tests/test_run_artifact_metadata.py`, `intergrax/integrations/providers/embedding_provider/ollama/manifest.py` (deleted), `intergrax/integrations/providers/embedding_provider/openai/register.py` |
| Five delta paths `0fc8badd8..d02c39950` | **confirmed**, equal to the audit's `delta_act_n11_paths_vs_0fc8badd8`: the three above plus `intergrax/platform/enterprise_audit/impact_map.py` and `.../data/changed_since_impact_map.v1.json` |
| Six paths against the audited tree `324d9dba3` (and against the record's `29d66ae3f`) | **confirmed**: the five plus `ollama/register.py` (unchanged since `0fc8badd8`, audited in addendum 3) |
| Product diff `0cb5dd011` → head empty | **confirmed**; also `d02c39950` → head empty |
| The merge `d02c39950` changes, outside `docs/`, only | `governance/tickets/T-IGX-040-B.md` (approval block) relative to `0cb5dd011`. No change under `scripts/governance/`, `scripts/maintenance/`, `scripts/enterprise_audit/`, `agent-env/`, the activation record, `pyproject.toml` or `uv.lock` since `0fc8badd8`. `.github/workflows/unit-tests.yml` differs from the pin (EBH-7 default SHAs, commit `0b62781a2`, already seen in addendum 2) |
| Gate lists the six product paths plus the protocol as "stale audit after material governance change" | **reproduced** (`logs/activation_gate_at_head_d02c39950.log`): `FAIL`, exactly seven lines: one `ACT-N11 stale audit after material governance change: ...INTEGRAX_INTERNAL_INDEPENDENT_AGENT_CERTIFICATION_PROTOCOL.md` and the six `ACT-N11 product path changed after the audited SHA` lines; exit 1. Nothing else |

The protocol is one of the six `MATERIAL_GOVERNANCE_PATHS`; it changed in `7b1c0a4b7` (T-IGX-040-B) and arrived through `d02c39950`. `_material_changes` against earlier candidate audit SHAs lists it for `29d66ae3f`, `0fc8badd8` and `0cb5dd011` and is empty for `d02c39950` (section 6).

## 2. Checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Pin chain, 161-path union, five delta paths, six against the audited tree, empty product diff to the head | PASS | section 1 |
| 2 | Gate at the head lists exactly the six paths and the protocol | PASS | `logs/activation_gate_at_head_d02c39950.log` |
| 3 | Addendum 4 manifest (as published by the auditor): both verifiers | PASS | `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff` exit 0 (`logs/verify_iiac_addendum_4_manifest_unpinned_at_d02c39950.log`); the manifest has the placeholders `PENDING-CERTIFIER-PDC-P7-00` and 40 zeros; the evidence was not on `mcp-rc` at my base (round 5 landed later, section 9) |
| 4 | Digests equal the published files | PASS | 4 of 4 equal the blobs on the auditor branch; before and after the pin-only edit: section 9 |
| 5 | Session ids distinct | PASS_SCOPED | implementation `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (the addendum 2 and 3 label, kept although the delta is `T-IGX-036-Y/-Z`, `T-IGX-040-A/-B`: a label, not an attested session); auditor `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-...-ADDENDUM-4` (the same auditor session plus the suffix; equals none of the MCP-RC auditors); certifier placeholder. `process_enforced` |
| 6 | MCP-RC contract, one pytest process, list from Python, `PYTHONPATH` = my checkout | PASS | **481 passed, 1 xfailed (strict `[product]`), 0 failed, exit 0**, 637.56 s (`logs/rc_contract_certifier_at_d02c39950.log`); equals your number |
| 7 | "Resolved" claims F-49, F-50, F-41, F-09 | PASS | section 3 |
| 8 | Protocol two-line edit weakens no gate | PASS | section 4 |
| 9 | Findings F-56..F-61 have a disposition | PASS_SCOPED | section 5: 6 of 6, no BLOCKER/CRITICAL; 0 HIGH / 1 MEDIUM / 0 LOW / 5 INFO, recomputed from the register |
| 10 | F-56 position | agree, with qualifications | section 6 |
| 11 | Guardrails | PASS | no change to the activation record since phase 2, to any auditor file outside the new `addendum_4/` folder, to P8 or FRZ files; no AI-attribution trailer in `0c79c9f65..HEAD` (the ladder commits are the project's own) |
| 12 | Re-pin preconditions | PASS_SCOPED | section 6 |

## 3. "Resolved" claims, from my own runs

Two pytest runs on this head (`PYTHONPATH` = my checkout):

- **Whole `tests/qualification/pdc_x` directory**, one process, no `os._exit`, cap 1500 s: **975 passed, 2 skipped, the pytest process exited by itself with exit 0**, 266.21 s (`logs/pdc_x_whole_dir_exit_proof_at_d02c39950.log`). This directory contains the XD-12 test, the `p6_impact` coverage test and the new Tier-3 root rule test (F-41), the `p6_ecapfix` wrapper pins for `ollama` and `openai` (F-49, F-50), and `test_xd_03_agent_governance.py` (F-09). Therefore: **F-41 RESOLVED** (the two tests that were red in addenda 1 to 3 pass; their test files are byte-unchanged since `0fc8badd8`, so they were not narrowed); **F-49 RESOLVED** at the test level (the pins pass; the auditor's own probe showed `openai|embedding_provider` now resolves, one of 201 results differs); **F-50 RESOLVED** at the test level (the pins exist and pass; I did not re-run the auditor's 20 wrapper mutants); **F-09 RESOLVED** (the process terminates by itself; the xd_03 diff keeps `pytest.raises(ProductionRuntimeToolInvokerCompositionError)` and adds only fixture closing of the invokers and the boundary).
- **Status run, 239 tests, 8 files** (`logs/status_f56_f35_f42_iiac_unit_at_d02c39950.log`): **215 passed, 23 failed, 1 skipped**. The 23: **F-56** ×1 (`test_b6_03_scoped_disposition_valid_at_head`); **F-35** ×17 (4 in `..._r3_p0_gates.py`, 13 in `..._r3_r1_gates.py`, the same 17 as in addenda 1 and 2: 10 new since PDC-X plus 7 red at the pin); the three duckdb cases of `test_ebh_5_behavioral_package_c_v2.py` (environment: `duckdb` is missing from the shared `.venv`); and 2 IIAC unit tests that fail by design until the re-pin (`test_canonical_policy_is_in_force_and_verifies`, `test_n12_policy_is_in_force_only_with_valid_activation_evidence`; both run the gate, which reports the seven lines above). The other cases of that file (the F-42 `embedding_provider` cases), `test_pdc_x_inventory.py` (F-07), `test_iiac_g0_inventory.py` and the rest of the IIAC unit tests pass. `verify_iiac_g0_inventory_completeness` PASS (`logs/verify_iiac_g0_inventory_at_d02c39950.log`).

Not re-run by me: F-35's pin comparison, the auditor's 227-failure regression attribution (F-53), the mutation samples (F-59), the ebh_7 directory.

## 4. The protocol edit (T-IGX-040-B)

`git diff 0cb5dd011 HEAD` on the protocol is numstat 2/2, lines 4 and 91 only (`logs`: `data/addendum4_checks.json`, key `protocol`). Removed: the hard-coded status `**NOT_IN_FORCE**` in the header and `Status remains **NOT_IN_FORCE** until ...`. Added: a pointer to `IIAC_POLICY_ACTIVATION_RECORD.json` as the single source of truth. I verified that the three requirement sentences are still present verbatim (`IN_FORCE requires the fail-closed gate (--require-in-force) and a separate operator activation record`; `a certification session after the ACT-01 gate correction is required`; `All required before IN_FORCE:`). Consumers of the file: `iiac_common.py` (a path constant), `iiac_g0_inventory.py` and `verify_arch_freeze_plan_traceability.py` (existence and category rules, no text match); the only `NOT_IN_FORCE` string in a script is in `evol_x_closure_publication_preflight.py` and reads the record, not the protocol. `verify_iiac_g0_inventory_completeness` PASS and the IIAC unit tests (section 3) agree. **Confirmed (F-61): the edit weakens no gate.** Its consequence is the one the gate prints: `ACT-N11 stale audit after material governance change` until `audit_execution_sha` is at or after `d02c39950`.

## 5. Findings F-56..F-61

Dispositions are the auditor's (`iiac_audit_p7_00_addendum_4_findings.json`, read from the auditor branch). The register has a `findings` array and no open BLOCKER/CRITICAL.

| ID | Severity | Auditor's recommendation (verbatim) | My check |
| --- | --- | --- | --- |
| F-56 | MEDIUM | "orchestrator/operator decision: a test-side edit of the frozen list (closed-stage evidence, so it needs acceptance) or accept the red; restoring the file would be another product path and a re-pin; no product change is required for the ACT-N11 rebind" | **reproduced** (section 6). **Open**, decision |
| F-57 | INFO | "accept" | `.github/workflows/unit-tests.yml` EBH-7 default pins moved to `146e7acf1` (confirmed, already in addendum 2). See also my observation below |
| F-58 | INFO | "accept; sync the extra or use a clean venv" | not re-run |
| F-59 | INFO | "accept" | partly confirmed: the pins pass (section 3); the 201-pair probe and the mutation runs not repeated |
| F-60 | INFO | "accept" | confirmed by code read for the manifest deletion (no importer in code, tests or scripts; only the frozen list, the WBS and docs name it: `logs/f56_mechanism_and_ci_reach_by_git.txt`); the test diffs I read are listed below |
| F-61 | INFO | "accept" | confirmed, section 4 |

**Status of earlier findings, auditor's words kept short:** F-52 `OPEN_EXTENDED` (the two-write registration now also for `openai`), F-51 `OPEN_EXTENDED` (the `bool()` coercion and dropped keywords also for `openai`; `credential_ref` and `base_url` are not reachable through the factory), F-35 `OPEN_UNCHANGED`, F-53 `OPEN_UNCHANGED`. Resolved: F-49, F-50, F-41, F-09 (section 3), F-42, F-07, F-06 (addendum 2). Open and carried unchanged by this delta: F-05, F-04/F-37, F-38, F-43, F-44 (the weak catch-all runtime prefix, still a surviving mutant), F-10, the MCP-RC carry-overs (owner of the strict `[product]` xfail, RA-02, RA-03, RA-04, AF-05).

**My own observation on the test diffs (not in the auditor's F-57 or F-60).** I scanned `git diff 0fc8badd8 HEAD -- tests` for removed assertions (`data/addendum4_checks.json`, `test_diff_scan`; 20 files). Besides the ones the auditor lists (the `run_id` attribute in `test_sqlite_run_trace_store.py`, the `pytest.raises` re-indentation in `xd_03`, `importorskip` in the HF test), there are two **relaxed EBH-7 assertions** from the re-pin batch `T-IGX-036-W` (B-P6-E): (1) `_ebh7_p03_evidence_integrity.py`: `assert baseline == test_sha` became `assert baseline == test_sha or _git_is_ancestor(baseline, test_sha)`, and the matrix check gained a `PENDING` branch for a not-yet-recorded replay (with extra assertions that no replay counts are invented); (2) `test_ebh_7_b6_f10_f11_corrective_gates.py`: two `assert EBH7_TEST_EXECUTION_SHA in <dossier|traceability text>` became `assert EBH7_INDEPENDENT_ACCEPTANCE_BASELINE_SHA in ...`. The comments in the code give the reason (the signed acceptance stays at the accepted tree and the re-pinned replay is not accepted until re-audited), which matches the recorded operator item that the EBH-7 re-pin is done once at the integrated baseline. These are test files, not ACT-N11 paths, so they do not affect the rebind; I did not audit them and report them as a lead for the EBH-7 owner and the per-domain audits.

## 6. F-56 and the re-pin

**F-56, reproduced.** `test_b6_03_scoped_disposition_valid_at_head` fails at this head (status run). Mechanism by git (`logs/f56_mechanism_and_ci_reach_by_git.txt`): the frozen PRE-ENT list names `intergrax/integrations/providers/embedding_provider/ollama/manifest.py`; the check requires every listed path to be in `git diff --name-only a7d83925b..HEAD`; the file exists at the pin `2d3eceb6c` and at `0fc8badd8` (the path is in that diff, 1 line) and is gone at `0cb5dd011` and the head (0 lines), because T-IGX-036-Z deleted it. I did not run the test at the older SHAs (no old checkout); the auditor's logs say green at `0fc8badd8` and the pin, which agrees with the mechanism.

**Your position, checked.** I agree that it is not a blocker of the ACT-N11 rebind: the gate reads product-path bytes, tests are not ACT-N11 paths, and the auditor's `product_change_required_for_activation` is empty. On CI reach: no file under `.github/` or `scripts/ci/` names `roadmap_replay_x`, `closure_counter_integrity` or `test_b6_03`; `unit-tests.yml` runs `tests/unit`, named qualification files and the `diagnostics_full` profile, which is `workflow_dispatch` only; `enterprise-audit.yml` runs `tests/qualification/ent_audit_x`. The one route I found into `roadmap_replay_x` is the ENT-AUDIT check manifest, which names two entrypoints in `test_roadmap_replay_x_mechanical_exit_gates.py` (not `test_b6_03`). So by reading, `test_b6_03` is in none of the five PR contexts. What I cannot support: that `roadmap_replay_x` is therefore green in CI (my addendum 1 targeted ENT-AUDIT run had three `roadmap_replay_x` failures that were red at the pin as well; I did not map them to these entrypoints), and that the item is cheap: it is a green-to-red regression of a closed-stage gate caused by a delta in the audit, the frozen list is closed-stage evidence, and the auditor says an edit needs acceptance. It should have a named owner and an operator answer, not only be filed as a PDC-X test-side item. I do not disagree with treating it as a non-blocker.

**Re-pin, what this addendum changes compared with addendum 2 section 5** (gate code, `data/addendum4_checks.json`, `repin_simulation_at_head_evidence_placeholder_HEAD`; the evidence-publication commit does not exist yet, so the simulation uses the head as a stand-in):

| Candidate | Result with the gate's own rules |
| --- | --- |
| corrective `d02c39950`, `audit_execution_sha` `d02c39950` (or the round 5 evidence commit) | ancestry, product diff to the head, material paths and strict product diff all empty: **passes** |
| corrective `0fc8badd8`, `audit_execution_sha` `d02c39950` | strict product diff reports the five delta paths: **fails** |
| corrective `d02c39950`, `audit_execution_sha` `0cb5dd011` (before the protocol edit) | corrective is not an ancestor of the audit SHA, and the protocol is a stale material change: **fails** |
| the current record values | the six paths and the protocol: **fails (today)** |

- `corrective_implementation_sha` and the manifest's `audited_implementation_sha`, `test_execution_sha` → `d02c39950`; `audit_execution_sha` must be at or after `d02c39950` (the protocol rule). The first-rebind pattern (`audit_execution_sha` = `evidence_publication_sha` = the evidence commit) works if round 5 is a docs-only commit on top.
- **Round 5 first:** the addendum 4 evidence has to be integrated into `mcp-rc` (publication commit, then the pin-only edit of `certifier_session_id`, `evidence_publication_sha`, `certification_publication_sha`), as in rounds 2 to 4; the activation certifier cannot bind a SHA that does not exist. My own commit is not a precondition for that; `certification_publication_sha` is the commit that carries this addendum or a later one.
- Everything else in addendum 2 section 5.2 to 5.4 still holds, with `-ADDENDUM-4` as the auditor id, the addendum 3 and addendum 4 manifests in `product_session_manifest_paths`, the handoff naming the new ids, and the limitations rewritten (161 paths; F-41, F-42, F-49 and F-09 resolved; F-56 and the carried items open).
- **Signature.** The gate's operator-file rules are unchanged: no SHA, no path count, no tree is compared (section 5.5 of addendum 2 applies). The text of the signed decision is staler now than at addendum 2: it says 158 paths and the audited tree `324d9dba3`, calls F-41, F-42 and the drift open, and does not know the protocol edit. The two options of addendum 2 (a: the signature stands, confirmed by the operator; b: a fresh signature on revised text) stay as they were; I still cannot tell which the operator intends.
- **A tension the orchestrator should resolve before the activation certifier starts.** `DELEGATION-PDC-P7-ACT` (new in this range, `platform_domain_completeness/operator_decisions/`), item (c), lets a new delegate write a new `IIAC_OD_ACTIVATION_*.md` and run the two-phase publication, but only when "the audit manifest verdict is `PASS` with the certifier's evidence published"; item (e) treats "a verdict the gate does not accept (CONDITIONAL, PARTIAL, PASS_SCOPED)" as a loop trigger. The addendum 4 manifest is `CONDITIONAL` (dossier `PASS_SCOPED`). The MCP-RC route (`OD-MCP-RC-04` section 3) answers this differently: the activation certifier authors a `PASS` binding manifest from `PASS_SCOPED` audits, and that is what the signed decision describes. Both are operator records; I only note that they are two routes and that the re-pin should name one.

## 7. What this addendum does not cover

1. **Not an audit.** I audited no product path; I reproduced the gate output, the union and the F-56 mechanism and ran the tests above.
2. **Round 5 landed after my base.** Sections 1 to 8 were checked against `d02c39950` and the auditor branch `0d4758ca1`; section 9 checks the integration at `fbe400f06`.
3. **CI contexts.** `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast` and `ent-audit-pr-gate` were not run by me; no claim that any is green. `governance-fast` runs `tests/unit/governance`, where the two activation-gate tests fail by design until the re-pin.
4. **Not re-run:** the auditor's regression attribution (F-53), mutation samples, probes, the ebh_7 directory, F-35's pin comparison; `test_b6_03` at the older SHAs.
5. **The gate after the re-pin is predicted from the gate's rules, not run.**
6. **Freeze and drift:** the freeze is a declaration; `origin/development` is `83d939095` and `mcp-rc` has merged it; I did not check for later development commits beyond this fetch.
7. **Independence** is a `process_enforced` attestation (one git identity signs all commits); implementer and certifier ids are labels.

## 9. Round 5 (landed on `origin/mcp-rc` after my base; checked in a second commit)

While I was working, `mcp-rc` received the evidence integration of addendum 4: publication commit `a1b3053b49c28124a0abde679d733f21a597797f` and pin-only commit `fbe400f06cd9b77f0fb548ec0d2b2aaeafe10dac` (head of `mcp-rc` at my last fetch). My branch has not merged them (it would only add a merge commit); I checked them from `origin/mcp-rc` blobs, copying the manifest and its bound files into the worktree for the verifier run and removing them afterwards (`scripts/check_round5.py`, `logs/round5_checks_at_fbe400f06.log`). This closes the item of check 4 that section 2 could not do.

| Check | Result |
| --- | --- |
| Publication commit | 40 files, all under `docs/`, **40 of 40 blob-identical** to `origin/p7-00-baseline-audit` @ `0d4758ca1` |
| Pin-only edit | changes exactly `certifier_session_id` (`PENDING-CERTIFIER-PDC-P7-00` → `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`), `evidence_publication_sha` and `certification_publication_sha` (zeros → `a1b3053b4`); the only fields that differ from the auditor branch's manifest are the same three; the commit touches only that manifest and `EVIDENCE_INTEGRATION_RECORD.md` |
| Digests | report, scope, test matrix, findings: claimed value equals the bytes at the head of `mcp-rc`, at the publication commit and on the auditor branch (12 of 12) |
| Verifiers on the pinned manifest | `verify_iiac_audit_manifest` PASS and `verify_iiac_role_separation --strict-product-diff` PASS (exit 0) |
| Other changes `d02c39950..fbe400f06` | none under `intergrax/ agents/ applications/ scripts/`, none to the protocol, none to earlier P7-00 evidence |

I did not read the round 5 section of `EVIDENCE_INTEGRATION_RECORD.md` line by line; the checks above are made directly on git. The manifest keeps `audit_execution_sha` = `audited_implementation_sha` = `d02c39950` (so `--strict-product-diff` again compares an identical pair; my own diff in section 1 covers the delta), `test_execution_sha` = `0cb5dd011` (product-identical), and `certification_publication_sha` = the publication commit, not a preflight commit. The activation gate at `fbe400f06` was not re-run by me; the product and governance diff from `d02c39950` is empty, so section 1 applies unchanged. The evidence publication SHA for the re-pin is `a1b3053b4` (or the pin-only commit `fbe400f06` as a docs-only descendant, the activation certifier's choice).

## 8. Remaining steps (not mine)

Round 5 evidence integration of addendum 4 (done, section 9) → operator decision on F-56 (and owners for F-35, F-52/F-51 extension, the MCP-RC carry-overs) and on the signature question → route decision (the tension in section 6) → activation certifier: new binding manifest and handoff, both verifiers, contract run, phase 1, phase 2 → PR → five CI contexts on the pinned head → merge (operator) → terminal verification.
