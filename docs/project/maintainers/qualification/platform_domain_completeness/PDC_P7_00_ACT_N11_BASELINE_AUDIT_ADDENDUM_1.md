# PDC-P7-00 - ACT-N11 baseline independent audit, addendum 1 (G6)

Stage `PDC-X/P7-00/addendum_1` · policy `IIAC-2026-11-05.1` · auditor branch `p7-00-baseline-audit` · date 2026-11-13
Parent: [`PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md`](PDC_P7_00_ACT_N11_BASELINE_AUDIT_DOSSIER.md) (audit of `fa6ff1c96`, PASS_SCOPED, findings F-01..F-34).

| Item | Value |
|------|-------|
| Delta audited | `fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f` .. `19d552beeb6f0677a8bdc3b4ce65da15dd637c7b` (`mcp-rc`, merge of development `3dd991336`, the operator's re-pinned baseline) |
| Activation pin | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` |
| Product paths in the delta | 6 (PR #523 = T-IGX-036-T item 11, PR #524 = T-IGX-036-U B-P6-M) |
| **Addendum verdict** | **PASS_SCOPED** (IIAC manifest value `CONDITIONAL`) |
| **Combined verdict** (first audit + addendum) | **PASS_SCOPED** |
| New findings | 0 HIGH · 1 MEDIUM · 3 LOW · 2 INFO (F-35..F-40) |
| Product change required for the ACT-N11 rebind | none |

Audit only: no product, test, ticket, activation-record or ruleset edit; no merge of anything but `origin/mcp-rc` into my own audit branch, no PR, no rebind, no `Approved:` line; only `p7-00-baseline-audit` is pushed.

## 1. Verdict

The six delta paths close the declared gaps FINDING-6 (tenant-bound memory plane), FINDING-9 (revision does not pin the LLM profile) and FINDING-10 (model features not checked at composition) at component level, tighten rather than weaken governance, and add no import, credential, endpoint or process/network surface. F-06 of the first audit is resolved. The addendum is scoped, not clean, because (a) the mechanisms of PR #524 and the pin are opt-in with no production caller (F-37), (b) the memory guard protects only the shipped capability wrapper (F-36), and (c) a wider regression selection than the first audit's exposed a red COMPAT-X block that PDC-X introduced and the first audit had not run (F-35, identical at `fa6ff1c96`, so not an effect of this delta). All of it is a test-only repair, a domain-audit lead or an orchestrator decision; none needs a product change before the rebind.

## 2. Scope verified from git

- `git diff --name-only fa6ff1c96 19d552bee -- intergrax agents applications` lists exactly the 6 announced paths: `agent_distribution/admin_service.py`, `agent_distribution/runtime_revision.py`, `applications/_shared/llm_resolver.py`, `contracts/agent_contract_meta.py`, `memory/default_memory_control_plane.py`, `runtime/agent_governance/pipeline.py` (+124 / -25 lines). Three were already among the first 153 (`admin_service`, `default_memory_control_plane`, `pipeline`), three are new to the set.
- The 23 MCP-RC paths are byte-identical to `b0e19bf61` (`git diff` empty).
- No path under `scripts/governance/`, `governance/agent-control-plane/`, the IIAC activation record, `iiac_evidence/` or `.github/` changed between the two SHAs (the only governance-adjacent changes are the claims-only tickets `T-IGX-036-T`, `T-IGX-036-V` and their plans).
- The two source commits carry their tickets: `25aaa1bfd` (T-IGX-036-T, #523) and `4dbe68452` (T-IGX-036-U, #524).
- My branch contains `origin/mcp-rc` through merge `2bbcca89c`; its product tree is identical to `19d552bee`.

## 3. Own numbers

| Run | Result |
|-----|--------|
| One pytest process, 268 files: MCP-RC contract (39 files, list from Python) + `tests/qualification/pdc_x/**` (incl. new `p6_model`) + EBH-5 gate files + `test_pdc_x_*` + `tests/unit/memory` + `tests/unit/agent_distribution` + `tests/qualification/compat_x` + agent-contract unit tests | 53 failed, 3384 passed, 3 skipped, 2 xfailed, 2832.94 s; no MCP test failed; pytest did not exit after the summary (killed by the runner after 30 s, F-09 unchanged) |
| The 53 node ids re-run in short-path detached checkouts (head / `fa6ff1c96` / pin) | 52 failed, 1 passed / the same 52 / 40 failed of 51 existing |
| ruff on the 6 paths, delta tests and `pdc_x_wbs.py` | all checks passed |
| tier and boundary gates, `uv lock --check` | exit 0, except `check_intergrax_no_applications_imports.py` exit 1 with output byte-identical to the first audit (12 pre-existing, untouched files) |
| probes | 5, 0 errors |

## 4. Status of the first-audit findings that the delta touches

| ID | Status at `19d552bee` |
|----|----------------------|
| F-06 EBH-5 `sqlite_closed_world_accountability` red | **RESOLVED**: the test now asserts `SQLITE_TOUCHPOINT_COUNT == 39`; the three EBH-5 gate files pass |
| F-07 `test_pdc_x_inventory` red | **OPEN** (same two tests) |
| F-05 six declared-open gaps | **PARTLY RESOLVED**: item 11 (two tests), FINDING-9, FINDING-10, FINDING-6 repaired, xfail markers removed, tests pass; FINDING-8 (ENT-AUDIT impact map) remains the only declared gap and is deferred to batch B-P6-G / T-IGX-036-V "after MCP-SERVER-X P8" |
| F-04 hardening without production callers | **OPEN, extended** by F-37 |
| F-09 pytest exit hang | **OPEN** |
| other F-01..F-34 | unchanged (the delta does not touch their files; `pipeline.py` is format/import only) |

## 5. New findings

Full evidence: `iiac_evidence/PDC-X/P7-00/addendum_1/iiac_audit_p7_00_addendum_1_findings.json`.

| ID | Sev | Finding | Verification | Disposition |
|----|-----|---------|--------------|-------------|
| F-35 | MEDIUM | Ten COMPAT-X event-evolution gates (`..._r3_p0_gates` x4, `..._r3_r1_gates` x6) are red since PDC-X and green at the pin: `security_audit_trail.py:198/213` (evidence-plane audit trail appending canonical RuntimeEvents) is an unclassified persistence append / reader site ("production persistence bypass count 1, expected 0"). Identical at `fa6ff1c96`; the first audit did not run `compat_x`, `agent_distribution` or `memory` suites | test run, attribution in three checkouts | test-only fix / classification by the COMPAT-X owner |
| F-36 | LOW | Tenant-bound memory guard is gated by `isinstance(..., UserProfileManagerMemoryCapability)` and reads a private `_tenant_id`; a duck-typed capability accepts a foreign-tenant identity. Shipped wrapper refuses tenant-B and a whitespace variant; covers remember, recall, supersession, forget, reconcile for USER scope | probe Q1 | IA-D2 |
| F-37 | LOW | `required_llm_features` and `llm_profile_source` have no production caller; an unpinned revision activates without a check; a pinned revision without a composed source fails closed. Positive: undeclared adapter refused, profile digest sensitive to model/options/fallbacks/routing hint, unknown or upper-case feature names rejected | probes Q2-Q5 | IA-D1/D3, orchestrator |
| F-38 | LOW | `test_cx_r3_p0_q2_inventory_schema_ids_match_registry` fails in the long combined run and passes in a fresh process (global payload-schema registry, order dependent; cause not isolated) | test run | COMPAT-X owner |
| F-39 | INFO | 40 of the 53 failures are red at the pin too (agent_distribution materialization adapter/materialization/ac6/remediation tests, 10 further compat_x gates); cause not investigated | test run | accept as pre-existing |
| F-40 | INFO | Positive record: the six paths add no boundary violation, credential, endpoint, process/network/dynamic-code surface and no weakened guard; MCP-RC paths still byte-identical | code read, gates | accept |

## 6. Combined coverage statement

The two audits together cover the **156 unique ACT-N11 paths** that differ between the activation pin `2d3eceb6c` and the audited head `19d552bee`: 153 audited at `fa6ff1c96` (23 MCP-RC relied on the existing delta re-audit + 130 PDC-X) plus the 3 paths first seen in the delta (`runtime_revision.py`, `llm_resolver.py`, `agent_contract_meta.py`); the other 3 delta paths (`admin_service.py`, `default_memory_control_plane.py`, `pipeline.py`) were in the 153 and are re-read at their final state here. Equivalently: 153 + 6 delta paths, 3 of them overlapping. Every product or test change after `19d552bee` voids the audit for the changed paths; a different re-pin voids it for the whole delta.

## 7. What this addendum does not cover

- The eight per-domain audits IA-D1..IA-D8; leads F-36, F-37, F-38 and the first-audit leads are theirs.
- CI contexts / GitHub Actions, enterprise and nightly profiles, `igx_gov ticket`, the full `tests/unit` tree, EBH replays, network and live-provider tests; P8 and FRZ (counter 225); documentation-only changes.
- No mutation sample was run on the six paths (the first-audit sample did not include them); the review is a full diff read plus five probes and the tests above.
- The 40 failures that are red at the pin were attributed, not diagnosed.
- The `fa6ff1c96`..`19d552bee` changes in tests, tickets and `scripts/qualification/pdc_x_wbs.py` were read only for loosening (none found: the xfail removals correspond to repaired, now passing tests; the EBH-5 literal change is a repair); the new batch B-P6-G is a claim, not audited.
- Windows only; the shared `.venv` lacks tzdata (F-34), a copy was used through `PYTHONPATH`.
- `certifier_session_id`, `evidence_publication_sha` and `certification_publication_sha` in the addendum manifest are placeholders.

## 8. Open conditions for the orchestrator / operator

1. F-06 is gone; F-07 and F-35 (test-only) should be repaired before the baseline is merged or CI contexts are read.
2. Decide whether host composition of `llm_profile_source` / `required_llm_features` (F-37) and of the first audit's F-04 items is a PDC-X claim.
3. Route F-36/F-37/F-38 and the first-audit leads to the domain audits.
4. The wider regression selection used here (`compat_x`, `agent_distribution`, `memory`) should be part of the later domain replays; the first audit's selection did not include it.

## 9. Evidence index

`docs/project/maintainers/qualification/iiac_evidence/PDC-X/P7-00/addendum_1/`: `audit_scope.json`, `test_matrix.txt`, `iiac_audit_p7_00_addendum_1_findings.json`, `attribution_of_regression_failures.json`, `iiac_audit_session_manifest.PDC-X.p7_00.addendum_1.json`, `EVIDENCE_MANIFEST.txt`, `logs/`, `adversarial_scripts/`.
