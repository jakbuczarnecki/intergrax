# PDC-P7-00 — certifier preflight (integrated baseline: MCP-RC together with PDC-X)

| Field | Value |
| --- | --- |
| **Campaign** | PDC-P7-00 ACT-N11 activation window: MCP-RC (`T-IGX-038-A..F`) merged with the PDC-X batches on the re-pinned `development` baseline |
| **Role** | Certifier, **preflight** (precedent: `MCP_SERVER_X_RC_CERTIFIER_PREFLIGHT.md`, `MCP_SERVER_X_P6_TERMINAL_DOSSIER.md`). IMPLEMENTER != AUDITOR != CERTIFIER; this session implemented and audited nothing, and is the same session that made the earlier MCP-RC preflight |
| **Session** | `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` (the designation the evidence-integration commit reserved; adopted here, so no pin-only edit is needed), branch `p7-00-certifier` cut from `origin/mcp-rc` @ `ae5623efb` |
| **Date** | 2026-11-13 |
| **Verdict** | **`CERTIFIER_PREFLIGHT_PASS_SCOPED`** (scope limits in section 3) |
| **Machine-readable** | `iiac_evidence/PDC-X/P7-00/certifier/certifier_preflight_result.json`; logs, data and scripts in the same folder |

This preflight is **not** the activation certification, **not** an IIAC rebind, **not** stage closure and **not** a production-readiness claim. FRZ counter stays 225; this session edited no FRZ status, activation record, P8 file, ruleset, auditor file, product file, test, ticket or `Approved:` line.

## 1. Pin chain (derived from git)

| Role | SHA | Verified |
| --- | --- | --- |
| Activation pin (`audit_execution_sha` of the record) | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ancestor of the head; the record file is byte-unchanged since the campaign base `6a6279a49` |
| MCP-RC last product change | `b0e19bf61b346e6941b2e190b703c6922e708b34` | the 23 MCP-RC paths are byte-identical at `19d552bee` and at the head |
| First audit, integrated product tree | `962c1d6706042675a7a2add6f4a89eb4d9deedb3` (merge of `development` `87f4d4f41`) | `962c1d670..fa6ff1c96` has 0 product paths (it adds one evidence log) |
| First audited SHA (`audited_implementation_sha`, `audit_execution_sha`) | `fa6ff1c9616a6df28c11b087eb3f1b0dc4097b5f` | ancestor of `19d552bee`; ACT-N11 gate paths at this SHA: **153** (= `audit_scope.json`) |
| Re-pinned `development` | `3dd991336233cea1662518df8b35e5e1ea901e00` (PR #524) | ancestor of `19d552bee`; supersedes the candidate `87f4d4f41` |
| Addendum 1 audited SHA | `19d552beeb6f0677a8bdc3b4ce65da15dd637c7b` (merge of `3dd991336` into `mcp-rc`) | ACT-N11 gate paths: **156** |
| Evidence publication commit | `6b195980d20067449ade0d2c2fd2280781c5a25d` | 56 files, all under `docs/`, blob-identical to `origin/p7-00-baseline-audit` @ `6b89a368e` (56 of 56) |
| Pin-only edits + integration record | `ae5623efbdfd69ddf4ba67f82861ccad130c9b33` | changes exactly 3 files (two manifests, the record) |
| Branch head certified | `ae5623efbdfd69ddf4ba67f82861ccad130c9b33` | `git diff --name-only 19d552bee ae5623efb -- intergrax agents applications` = **0**; the whole diff outside `docs/` = none |

**ACT-N11 union against the pin**, computed with the gate's own `product_paths_changed_between`: 153 at `fa6ff1c96`, 156 at `19d552bee`, 156 at the head. The delta `fa6ff1c96..19d552bee` is 6 paths: 3 already in the 153 (`agent_distribution/admin_service.py`, `memory/default_memory_control_plane.py`, `runtime/agent_governance/pipeline.py`) and 3 new (`agent_distribution/runtime_revision.py`, `applications/_shared/llm_resolver.py`, `contracts/agent_contract_meta.py`). The union is **156 = 23 MCP-RC + 133 non-MCP** (130 first-audit PDC-X + 3 new). Both `audit_scope.json` files equal the git-derived sets (`data/act_n11_union.json`). The product diff between the audited tree and the branch head is **empty**.

## 2. Checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Pin chain and ACT-N11 union (section 1) | PASS | `data/act_n11_union.json`, `logs/guardrails_and_pin_git_checks.txt` |
| 2 | Product diff audited tree to head empty | PASS | 0 paths for `19d552bee..ae5623efb` and for `962c1d670..fa6ff1c96` |
| 3 | Manifest session ids distinct | PASS_SCOPED | implementation `T-IGX-036-J-T-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER`; auditor `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-...`; addendum auditor = the same id plus the suffix `-ADDENDUM-1` (a continuation by the same auditor session, distinct strings); certifier = this session. None equals the MCP-RC auditors `MCP-RC-G6-AUDITOR-90803ebc-...` / `MCP-RC-G6-REAUDIT-AUDITOR-9579becc-...` or my earlier MCP-RC preflight id. Independence is a `process_enforced` attestation (one git identity signs all commits) |
| 4 | Manifest verifiers, per manifest | PASS | `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff`: PASS on the first and on the addendum manifest (`logs/verify_iiac_manifests_p7_00.log`) |
| 5 | Digests equal SHA-256 of the files, also after the pin-only edits | PASS | 8 of 8 digests match the files at the head, at `6b195980d` and on the auditor branch; the pin-only diff touches only `certifier_session_id`, `evidence_publication_sha`, `certification_publication_sha` in each manifest; no manifest is an input of a digest (`data/p7_manifest_checks.json`). The integration record's claim "no digest changed" is **confirmed** |
| 6 | MCP-RC contract, one pytest process, list from Python, `PYTHONPATH` = my checkout | PASS | **481 passed, 1 xfailed (strict), 0 failed, exit 0, 352.43 s**; equal to the implementer log `rc_contract_at_rebaselined_19d552bee.log` (481 / 1, 816.27 s). The xfail is the TRANSFERRED `test_scaffold_defaults_mcp_to_opt_in[product]` (`logs/rc_contract_certifier_at_ae5623efb.log`) |
| 7 | EBH-5 gate files, `test_pdc_x_inventory.py` and the COMPAT-X gates, one process | **PASS with open reds** | 116 tests: 97 passed, **19 failed** (`logs/gates_certifier_at_ae5623efb.log`). EBH-5 (3 files): all green, **F-06 resolved** (`SQLITE_TOUCHPOINT_COUNT == 39`). `test_pdc_x_inventory.py`: **2 failed, F-07 is still OPEN** (3 documents under `docs/project/capabilities/` are `UNMAPPED`, first `ARTIFACT_LIFECYCLE.md`). COMPAT-X event-evolution gates: 17 failed = 10 red only since PDC-X (F-35) + 7 red at the activation pin too; I re-ran both COMPAT-X files on a detached checkout of the pin and exactly the same 7 fail there (`logs/compat_x_at_pin_2d3eceb6c.log`). My 12 PDC-X-only reds equal the auditor's `fail_at_head_not_at_pin` set node for node |
| 8 | Findings F-01..F-40 each have a disposition | PASS_SCOPED | section 4: 40 of 40, no duplicates, no BLOCKER/CRITICAL; severity 0 HIGH / 7 MEDIUM / 25 LOW / 8 INFO (counts of both files re-derived) |
| 9 | ACT-N15: findings JSON has a real `findings` array | PASS | the gate's `_findings_errors` returns none for both P7-00 files (34 and 6 entries); it returns `ACT-N15 findings register has no findings array` for the MCP-RC **re-audit** file (confirms runbook Q3) |
| 10 | No FRZ change (225), activation record, P8, ruleset, governance script | PASS | the activation record is unchanged since `6a6279a49`; `mcp-rc` adds nothing sensitive on top of merged `development` `3dd991336`, and `development` changed none between the base and `3dd991336`; no `D`/`R` entries pin..head; no `MATERIAL_GOVERNANCE_PATHS` path changed after the audited SHA |
| 11 | No weakened or removed test in the delta | PASS | across the 30 modified existing test files (pin..head) `def test_` 290 to 292, `assert` 699 to 711, no `deselect`, skips 2 to 2, `xfail` 0 to 1 (only the strict TRANSFERRED case); the four MCP-RC files with fewer asserts were read in the first preflight (contract-driven replacements); 132 files are new (`logs/test_diff_scan_p7.txt`) |
| 12 | MCP-RC mutation harness | not re-run (justified) | the 23 product paths, the harness and the RC test directories are byte-identical to `1b5f26f24`, where my own run killed 18 of 18 (`iiac_evidence/PDC-X/MCP-RC/certifier/mutation/`) |
| 13 | Activation gate state at the head | **FAIL, expected** | `verify_iiac_policy_activation --require-in-force`: exit 1, **156 errors, all `ACT-N11 product path changed after the audited SHA`**, no other error class (`logs/verify_iiac_policy_activation_at_ae5623efb.log`) |
| 14 | Pin-only edits by this certifier | NONE | the placeholders were already replaced by the integration session; my session id equals the reserved designation |

## 3. Verdict and scope limits

`CERTIFIER_PREFLIGHT_PASS_SCOPED`. The pin chain, product-tree identity, independence attestations, digests, MCP-RC contract and guardrails hold on my own runs. PASS applies **only** under these limits:

1. **The baseline carries red tests that the auditors found and that are still red.** 12 tests are red only since PDC-X (10 COMPAT-X event-evolution gates, F-35; 2 inventory tests, F-07), 7 more COMPAT-X gates are red at the pin, and the auditors attribute 40 regression failures to the pin (F-39; I re-ran only the COMPAT-X 7). These are test-only repairs (auditor) and no ACT-N11 argument covers them: `governance-fast`, `ent-audit-fast` or `ent-audit-pr-gate` can be red on the PR head. This preflight does not assert that the five CI contexts are green.
2. **F-09** (pytest does not exit after `test_xd_03_agent_governance.py`) is not exercised by my runs; a CI job running that file can hang.
3. **Domain audits not done.** The eight per-domain audits IA-D1..IA-D8 are out of scope of both audits and of this preflight; 27 findings carry a ROUTE disposition. F-18 and F-24..F-28 are unreproduced leads from review helpers.
4. **`development` has moved after the pinned baseline.** `origin/development` is `a156cbbd5` (PR #525), 2 commits after `3dd991336`. It changes two product paths under ACT-N11 that are not in the 156 (`intergrax/platform/enterprise_audit/impact_map.py` and `intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json`) and three tests (including a 13-line removal in `tests/qualification/pdc_x/xd/test_xd_12_evol_ent_audit.py`). Neither audit covers them. The CI ACT-N11 evaluation runs on the PR merge ref, which contains the current `development`, so the gate would list 158 paths until a delta audit and re-pin, or a development freeze for the window. Any further product change on `development` repeats this.
5. **The auditors' PASS_SCOPED is not a PASS.** Both manifests carry `CONDITIONAL`; the gate rejects `CONDITIONAL` and `PASS_SCOPED` as the bound verdict (ACT-N05, `_verdict_error`). See section 5, item 3.
6. **Coverage statements inherited from the audits** (not re-verified by me): the 130 first-audit PDC-X paths were reviewed by diff scan, selective manual reading, three read-only review helpers, 22 probes and 12 mutants; there is no mutation sample on the six delta paths; CI contexts, G-DOC-01..12, P8, full `tests/unit`, network and live-provider paths are not covered; the shared `.venv` lacks the `tzdata` the baseline declares (my runs did not hit a `tzdata` error).
7. **MCP-RC carry-overs** from the first preflight stay open: owner of AF-05 / AF-16 (`T-IGX-039-A` is a draft), RA-02, RA-03, RA-04.

## 4. Findings F-01..F-40

All 40 have a disposition in the two findings files (`iiac_audit_p7_00_findings.json`, 34 findings; `addendum_1/iiac_audit_p7_00_addendum_1_findings.json`, 6 findings; re-derived in `data/findings_disposition.json`). The addendum's status of the first-audit findings: F-06 **RESOLVED**; F-07 **OPEN**; F-09 **OPEN**; F-05 **PARTLY_RESOLVED** (the one remaining declared-open gap is FINDING-8 / `xd_12`, deferred to `B-P6-G` / `T-IGX-036-V` "after MCP-SERVER-X P8"); F-04 **OPEN_EXTENDED** (see F-37); the rest **UNCHANGED**.

Disposition classes: ACCEPT 8 (F-08, F-30..F-34, F-39, F-40), ROUTE_TO_IA-D* 27, test-only fix 2 (F-06 resolved, F-35 open), recommended fix 2 (F-09 open, F-10), inventory regeneration 1 (F-07 open).

**Open and not accepted (auditor dispositions, verbatim):**

- **F-07** (LOW): "regenerate the inventory (scripts/qualification/pdc_x_inventory.py --write) by the owner". Confirmed red by me.
- **F-35** (MEDIUM): "TEST_ONLY_FIX / classification by the COMPAT-X owner (add the site to the persistence classification registries); not a product change; to be read together with F-06/F-07 as 'CI contexts red after merge'". Confirmed red by me (10 nodes).
- **F-09** (LOW): "RECOMMEND_FIX (test owner)".
- **F-10** (LOW): "RECOMMEND_FIX: check decision.is_allowed in _guard" (a product change is recommended; the auditor requires no product change for activation).
- **F-38** (LOW): "ROUTE_TO_COMPAT-X owner" (an order-dependent COMPAT-X gate).
- **F-04** (MEDIUM): "ROUTE_TO_IA-D6 (and D1/D5); orchestrator to decide whether host composition wiring is a PDC-X claim; activation not blocked"; extended by **F-37** (LOW): "ROUTE_TO_IA-D1/D3; orchestrator to decide whether host composition is a PDC-X claim".
- **F-05** (MEDIUM): "ROUTE_TO_IA-D2/D3; keep the xfails strict (they are); orchestrator to track the held write-set amendment".

**Routed to the later per-domain audits** (first audit's register plus the addendum): IA-D1 F-11, F-12, F-21, F-26, F-37; IA-D2 F-15, F-17, F-18, F-19, F-28, F-36; IA-D3 F-05, F-37; IA-D4 F-13, F-16, F-22; IA-D5 F-24, F-25; IA-D6 F-04, F-10, F-20, F-23, F-30; IA-D7 F-03, F-14; IA-D8 F-01, F-02, F-27, F-28, F-29. (The register lists F-10 and F-30 under D6 although their disposition text is RECOMMEND_FIX and ACCEPT.)

Full table (auditor dispositions, first clause):

| ID | Sev | Title | Disposition (auditor, verbatim head) | Routed to |
| --- | --- | --- | --- | --- |
| F-01 | MEDIUM | Workspace scoping of a governed artifact read is optional | ROUTE_TO_IA-D8 | IA-D8 |
| F-02 | MEDIUM | TTL deletions can run without any audit record, and delete_artifact removes versions written af | ROUTE_TO_IA-D8 | IA-D8 |
| F-03 | MEDIUM | Recurrence occurrence identity has no tenant component; a conflict is swallowed | ROUTE_TO_IA-D7 | IA-D7 |
| F-04 | MEDIUM | Several PDC-X hardening capabilities have no production caller; fail-closed behaviour exists on | ROUTE_TO_IA-D6 (and D1/D5) | IA-D6 |
| F-05 | MEDIUM | The baseline carries six declared-open gaps pinned as strict xfail, including a tenant-isolatio | ROUTE_TO_IA-D2/D3 | IA-D3 |
| F-06 | MEDIUM | A previously green EBH-5 gate is red on the baseline (hard-coded SQLite touchpoint count) | TEST_ONLY_FIX (implementer/orchestrator) | - |
| F-07 | LOW | The new PDC-X inventory tests are red on the baseline (documents added on development are unmap | regenerate the inventory (scripts/qualification/pdc_x_inventory.py --write) by the owner | - |
| F-08 | INFO | Seven existing failures are not caused by the delta: five README contract tests and two audit-i | ACCEPT_AS_PRE_EXISTING | - |
| F-09 | LOW | pytest does not exit after its final summary when test_xd_03_agent_governance.py runs | RECOMMEND_FIX (test owner) | - |
| F-10 | LOW | GovernedArtifactReadPort._guard ignores the authorization decision value | RECOMMEND_FIX: check decision.is_allowed in _guard | IA-D6 |
| F-11 | LOW | upgrade_skill restores the previous release only for Exception, not for BaseException | ROUTE_TO_IA-D1 | IA-D1 |
| F-12 | LOW | Skill upgrade ordering ignores pre-release and build tags | ROUTE_TO_IA-D1 | IA-D1 |
| F-13 | LOW | With a RetryPolicy, crash-interrupted RUNNING tasks are redelivered while the worker still pass | ROUTE_TO_IA-D4 | IA-D4 |
| F-14 | LOW | Recurrence materialization runs in a finally for dead tasks and a materialization failure only  | ROUTE_TO_IA-D7 | IA-D7 |
| F-15 | LOW | Retention sweep task trusts producer-controlled as_of and joins a helper thread without timeout | ROUTE_TO_IA-D2 | IA-D2 |
| F-16 | LOW | Lifecycle metric publishes await the exporter without a bound in the shutdown path | ROUTE_TO_IA-D4 (not probed) | IA-D4 |
| F-17 | LOW | A raising reindex hook turns a successful retrieval into an exception | ROUTE_TO_IA-D2 | IA-D2 |
| F-18 | LOW | Freshness identity comes from mutable chunk metadata without tenant; ingest stamps an absolute  | ROUTE_TO_IA-D2 (verify) | IA-D2 |
| F-19 | LOW | Provider egress filtering runs after required-source validation | ROUTE_TO_IA-D2 | IA-D2 |
| F-20 | LOW | Evidence-plane security audit trail: list_entries returns the oldest 1000 entries, secret value | ROUTE_TO_IA-D6 | IA-D6 |
| F-21 | LOW | Plugin provenance: attestation lookup is bound to the operator mapping key, not to attestation. | ROUTE_TO_IA-D6 | IA-D1 |
| F-22 | LOW | HostDeploymentProfile changed from extra='forbid' to extra='allow'; unknown keys and a wide bin | ROUTE_TO_IA-D4 | IA-D4 |
| F-23 | LOW | Non-strict profiles sign boundary events with the public demo seed when no key is set | ROUTE_TO_IA-D6 | IA-D6 |
| F-24 | LOW | Evaluation evidence digests are verified against digests stored with the payload; promotion com | ROUTE_TO_IA-D5 (verify) | IA-D5 |
| F-25 | LOW | Online evaluation registry: cross-tenant observation id existence oracle and lock-file removal  | ROUTE_TO_IA-D5 (verify) | IA-D5 |
| F-26 | LOW | Agent upgrade admission: 0.x compatibility rule, pre-release ordering, replay paths before auth | ROUTE_TO_IA-D1 (verify) | IA-D1 |
| F-27 | LOW | Run artifact bundle builder follows file symlinks under the sandbox root and now publishes a co | ROUTE_TO_IA-D8 (verify) | IA-D8 |
| F-28 | LOW | Retention and artifact store edge cases: naive/aware datetime TypeError kills the sweep, one ta | ROUTE_TO_IA-D8 (verify) | IA-D2, IA-D8 |
| F-29 | LOW | Mutation sample: two redundant tenant checks are not pinned by any PDC-X test | ROUTE_TO_IA-D6/D8: add a hostile-source test | IA-D8 |
| F-30 | INFO | raw_secret_gate is a heuristic AST scan with known evasions | ACCEPT | IA-D6 |
| F-31 | INFO | Behaviour changes that can break callers: sha256 fields now accept only empty or 64 lowercase h | ACCEPT | - |
| F-32 | INFO | Existing tests were retargeted rather than removed; two claims in the gate script are not met | ACCEPT | - |
| F-33 | INFO | The import-boundary gate is red on both the pin and the baseline for 12 files the delta did not | ACCEPT_AS_PRE_EXISTING | - |
| F-34 | INFO | The shared .venv lacks tzdata that the baseline pyproject now declares | ACCEPT | - |
| F-35 | MEDIUM | Ten COMPAT-X event-evolution gates are red since PDC-X; they are green at the activation pin | TEST_ONLY_FIX / classification by the COMPAT-X owner (add the site to the persistence classification registries) | COMPAT-X owner (test-only) |
| F-36 | LOW | The tenant-bound memory guard applies only to the shipped UserProfileManagerMemoryCapability | ROUTE_TO_IA-D2 | IA-D2 |
| F-37 | LOW | The LLM profile pin and the required-feature check are opt-in; no production caller uses either | ROUTE_TO_IA-D1/D3 | IA-D1/D3 |
| F-38 | LOW | A COMPAT-X schema-id gate fails in the long combined run but passes in isolation | ROUTE_TO_COMPAT-X owner | COMPAT-X owner |
| F-39 | INFO | Combined regression run: 53 failures, 40 of which are red at the activation pin | ACCEPT_AS_PRE_EXISTING | - |
| F-40 | INFO | The six delta paths contain no new import of agents/applications from intergrax/, no credential | ACCEPT | - |

## 5. What the activation certifier will need (checked against the gate code)

I read `scripts/governance/iiac_activation_gate.py`, `verify_iiac_policy_activation.py`, `iiac_common.py` and `MCP_SERVER_X_RC_ACTIVATION_RUNBOOK.md` (branch `origin/mcp-rc-activation` @ `7f00ca8be`). Items marked **confirmed** match the gate; items marked **correction** differ from the runbook or are missing in it.

1. **Binding manifest** (**confirmed**): `stage_id = IIAC-GOV-1`; schema and `validate_manifest_dict`; `audited_implementation_sha` = record `corrective_implementation_sha`; `audit_execution_sha` and `certification_publication_sha` equal to the record; implementer and auditor sessions and agents differ; non-empty auditor and certifier session ids; the four digests are SHA-256 of the working-tree bytes of the files at the record paths; both companion verifiers must pass on it, including `--strict-product-diff`.
2. **Ancestry chain** (**confirmed**): `governance_implementation_sha` to `corrective_implementation_sha` to `audit_execution_sha` to `evidence_publication_sha` to `effective_from_sha`, and `certification_publication_sha` to `effective_from_sha`; `certification_publication_sha` must differ from `corrective_implementation_sha`; `effective_from_sha` must differ from HEAD. Candidate values from my pin chain (a proposal, not a decision): `corrective_implementation_sha` = manifest `audited_implementation_sha` = `19d552bee` (product tree identical to the head); `audit_execution_sha` and `evidence_publication_sha` at `6b195980d` or later; `certification_publication_sha` = `ae5623efb` or later. The runbook's table pins (`b0e19bf61`, `78d4c4960`) are superseded, as the runbook itself says.
3. **Verdict (decision needed).** The gate requires the binding manifest verdict `PASS`; `CONDITIONAL` and `PASS_SCOPED` are rejected (ACT-N05, `_verdict_error`), so none of the four audit manifests (two MCP-RC, two P7-00) can be bound as they are. The activation certifier must author a new binding manifest with `verdict = PASS` (the P6 pattern) and state the scope limits in `limitations`. That turns the auditors' scoped verdicts into a bound PASS; the operator decision text should say so.
4. **Evidence path root (correction).** The runbook says "the gate requires only the prefix `iiac_evidence/` for the manifest". The gate requires it for **all five** of `audit_report_path`, `session_b_manifest_path`, `audit_scope_path`, `test_matrix_path`, `findings_path` (ACT-N14, `resolve_evidence_path`). The P7-00 dossier lives under `platform_domain_completeness/` and the MCP-RC dossiers under `mcp_server_x/`; they cannot be `audit_report_path` as they are. The P6 pattern (keep the `IIAC-GOV-1` report whose digest `c405c65f...` the draft decision already cites) works; if a P7-00 report is bound it needs a copy or a certifier-authored report under `iiac_evidence/`. The working-tree report bytes must also equal the bytes at `evidence_publication_sha` (ACT-N11 report check), so that commit must contain the bound report.
5. **Findings register** (**confirmed**, runbook Q3): the bound `findings_path` must be a JSON object with a `findings` array and no `BLOCKER`/`CRITICAL` entry that is not `RESOLVED`/`CLOSED` (ACT-N15). Both P7-00 files and the first MCP-RC audit file pass; the MCP-RC re-audit file does not (`new_findings`, no `findings`).
6. **Operator decision file** (**confirmed**, with additions to the runbook's step 3): besides the signed line the gate requires the exact text lines `Decision status: AUTHORIZED`, `Decision ID: <record operator_decision_id>`, `Policy version: IIAC-2026-11-05.1`, `Decision scope: IIAC policy activation` and `Audit report digest: <record digest>`; the activation line must be a whole line of the file and match the regex; the file name starts with `IIAC_OD_ACTIVATION_` and contains neither `PACKET` nor `DRAFT` in any case; the file must not contain `________________`, `NOT SIGNED` or `Decision status: PENDING`. The current draft already carries the other lines.
7. **ACT-N18** (**confirmed**): no `EVOL-X`, `ENT-AUDIT-X` or `ARCH-FREEZE` within 80 characters before `CLOSED` in the record or the decision text.
8. **Handoff record** (**confirmed**): name `IIAC_SESSION_HANDOFF_RECORD_*.md` under `docs/project/maintainers/qualification/`; the text contains `IIAC-2026-11-05.1`, the manifest's `implementation_session_id` and `auditor_session_id` (for P7-00 the implementation id is the long combined string, quote it exactly); the digest is SHA-256 of the working-tree bytes. The CRLF requirement is a P6 observation that the gate code does not show; I did not verify it. `iiac_evidence/** -text` in `.gitattributes` is confirmed.
9. **Phase 1 / phase 2** (**confirmed**): phase 1 leaves `effective_from_sha` and `policy_activation_sha` empty (expected: four errors, two missing-field and two not-40-hex); phase 2 sets both to the phase-1 commit SHA and changes nothing else; the ten `BINDING_KEYS` of the phase-1 blob must equal the record; the working-tree record must equal HEAD.
10. **ACT-N11 at the final head (correction of scope).** The union is **156** paths today (the runbook's 23 + 125 predates the baseline move); with `development` at `a156cbbd5` it would be 158 on the CI merge ref. The decision's "Widened scope" text must cite `19d552bee`, 156 paths, both audit scope files, both dossier verdicts, this preflight and the red-test caveat of section 3. `MATERIAL_GOVERNANCE_PATHS` (six files) did not change after the audited SHA (checked through `development` `a156cbbd5`).
11. **Window and order.** `DELEGATION-MCP-RC-2` expires 24 hours after the operator's commission answer (dated 2026-11-13) and probably needs a renewed commission; the PDC-P7-00 window has no opening record (runbook Q6).
12. **Tests before merge.** Mandatory: the five CI contexts green on the pinned PR head (`headRefOid` equals the pushed SHA). With F-07 and F-35 red on the baseline, `governance-fast` and the `ent-audit-*` contexts need the test-only repairs first; that work belongs to an implementer or the operator, not to a certifier step.
13. **Runbook Q11** (**confirmed**): `iiac_audit_session_manifest.v1.schema.json` allows `PASS_SCOPED`, contrary to the auditors' `LIM-verdict-mapping` wording; irrelevant to the gate, which rejects it anyway.

## 6. What this preflight does not do

Not the activation certification; no edit of `IIAC_POLICY_ACTIVATION_RECORD.json`; no binding manifest, handoff or decision file; no rebind, signing, merge or PR; no push to `mcp-rc`, `mcp-rc-activation`, `p7-00-baseline-audit` or `development`; no stage closure; no FRZ promotion; no production-readiness claim; no check of GitHub CI contexts; no audit of PDC-X product code (that is IA-D1..IA-D8).

## 7. Remaining steps (orchestrator and operator)

1. Decide the `development` drift: a delta audit and re-pin for `a156cbbd5` and later, or a freeze for the PDC-P7-00 window.
2. Test-only repairs for F-07 (inventory regeneration) and F-35 (COMPAT-X persistence classification) and a decision on F-09, by their owners; then confirm the CI-equivalent gates.
3. Commission the activation certifier (separate session) with items 1..10 of section 5; renew `DELEGATION-MCP-RC-2` for the signing delegate; the operator signs the activation line after the revised decision cites the integrated baseline.
4. Two-phase rebind inside the window; PR; merge only after the five contexts are green on the pinned head; terminal verification in a further session.
5. Operator-owned items: owner of the `[product]` xfail (`T-IGX-039-A`), acceptance of RA-02..RA-04 and AF-05, D-10 (closed PRs #475/#476/#478), owner for the P8 documentation corrections (G-DOC-01..12), and whether host composition is a PDC-X claim (F-04, F-05, F-37).
