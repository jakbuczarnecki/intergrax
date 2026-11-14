# PDC-P7-00 — certifier preflight, addendum 2 (covers audit Addendum 3, delta `324d9dba3..0fc8badd8`, and the re-pin of the activation record)

| Field | Value |
| --- | --- |
| **Parent** | `PDC_P7_00_CERTIFIER_PREFLIGHT.md` and `PDC_P7_00_CERTIFIER_PREFLIGHT_ADDENDUM_1.md` (both `CERTIFIER_PREFLIGHT_PASS_SCOPED`), unchanged |
| **Role** | Certifier, preflight addendum. IMPLEMENTER != AUDITOR != CERTIFIER; same certifier designation as the parent: `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13` |
| **Branch** | `p7-00-certifier`, fast-forwarded to `origin/mcp-rc` @ `bb78a7432f7da4b38d3f25ada42310306df0f36c` before this commit; my addendum 1 `7801dcb12` is on it |
| **Date** | 2026-11-14 |
| **Verdict** | **`CERTIFIER_PREFLIGHT_PASS_SCOPED`** (scope limits in section 6) |
| **Machine-readable** | `iiac_evidence/PDC-X/P7-00/certifier/addendum_2/certifier_preflight_addendum_2_result.json`; logs, data, scripts in the same folder |

Not the activation certification, not a rebind, not stage closure, not a production-readiness claim. FRZ counter stays 225. This session edited no auditor file, product file, test, ticket, activation record, signed decision, delegation, binding manifest, handoff, P8 file, ruleset or `Approved:` line, and signed nothing.

## 1. Pin chain and ACT-N11 union

| Role | SHA | Verified |
| --- | --- | --- |
| Activation pin before the first rebind | `2d3eceb6cf7e93dccd4d96919319a6c969b4444b` | ancestor of the head |
| Addendum 2 audited tree | `324d9dba37cb54e1b74a79a05148d2dcc05e6d09` | ancestor of `0fc8badd8`; the current record's `corrective_implementation_sha` |
| Addendum 2 evidence publication | `29d66ae3fcc7a06a919d7e07541740841e8abfe8` | ancestor of `0fc8badd8`; the current record's `audit_execution_sha` and `evidence_publication_sha` |
| Signed decision | `afdf27e622369c3c635aeb1301c9d90d962d8371` | file bytes unchanged since (`0e0aed5816ff76c631d8aefb897e83399d26aa6d9692f3b0c2fe6b1fd8d25147`) |
| Activation certifier: binding manifest, handoff, pre-checks | `2899735e56556b3d022da9fe427042ac66a08922` | files unchanged since |
| Phase 1 / phase 2 | `92d7308213bd46e76ce0b545cd94b36bc65d8e0e` / `9949fe2c3dd64064f3b2002565b31c3c80b8cad6` | record unchanged since phase 2 (`git diff 9949fe2c3 HEAD` on the record is empty) |
| `development` `146e7acf1` merged as | `0fc8badd81c4c3e46a3d0ce262c08e8efba41898` | the addendum 3 audited tree (`audited_implementation_sha` = `test_execution_sha` = `audit_execution_sha`) |
| Addendum 3 evidence publication | `d5c51df0ef2ae63730129e673b426937867c1568` | 44 files, all under `docs/`, blob-identical to `origin/p7-00-baseline-audit` @ `793cce5c8` (44 of 44) |
| Pin-only edit + round 4 record | `bb78a7432f7da4b38d3f25ada42310306df0f36c` (the head) | changes exactly the addendum 3 manifest and `EVIDENCE_INTEGRATION_RECORD.md` |

With the gate's own `product_paths_changed_between` against the pin: **153** (first audit) → **156** (addendum 1) → **158** at `324d9dba3`, **158** at `0fc8badd8` and **158** at the head; the three 158-sets are equal. **158 = 23 MCP-RC + 135 non-MCP.** The delta `324d9dba3..0fc8badd8` is the single path `intergrax/integrations/providers/embedding_provider/ollama/register.py`, which was already in the set (it entered through B-REG-P), so the union does not grow. The product diff `0fc8badd8..HEAD` is **empty**; the product diff `324d9dba3..HEAD` is that one path. The 23 MCP-RC paths are byte-identical to `b0e19bf61` at the head. Non-`docs/` changes `324d9dba3..HEAD`: four ticket files (`T-IGX-036-W`, `T-IGX-036-X` and their plans), `register.py`, `scripts/qualification/pdc_x_inventory.py`, `pdc_x_wbs.py`, `tests/qualification/ebh_5/_package_c_v2_primitives.py`, the new `tests/qualification/pdc_x/p6_ecapfix/` (2 files) and `uv.lock`; between `0fc8badd8` and the head there is **no** non-`docs/` change at all. No `MATERIAL_GOVERNANCE_PATHS` file and nothing under `scripts/governance/` or `.github/` changed since the pin.

I read the product delta (`git diff 324d9dba3 HEAD -- register.py`): `_dispatching_factory` wraps the owning row's factory; for `integration_category == EMBEDDING_PROVIDER` it builds the embedding integration with `enabled=bool(kwargs.get("enabled", False))` and drops every other keyword, otherwise it calls the owning factory with unchanged keywords; the registration is `augment_integration_contract_specs` followed by `register_integration(replace(entry, factory=...), override=True)`.

**The activation gate at this head** (`python -m scripts.governance.verify_iiac_policy_activation --require-in-force`, `logs/activation_gate_at_head_bb78a7432.log`): `IIAC policy activation: FAIL`, **exactly one line**: `ACT-N11 product path changed after the audited SHA: intergrax/integrations/providers/embedding_provider/ollama/register.py`; exit 1. That is the expected state until the re-pin, and it is the only error.

## 2. Checklist

| # | Check | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Pin chain, 158-path union with the gate's own function, the single drift path, empty product diff `0fc8badd8..HEAD` | PASS | section 1; `data/addendum3_checks.json` |
| 2 | Addendum 3 manifest: both verifiers | PASS | `verify_iiac_audit_manifest` and `verify_iiac_role_separation --strict-product-diff`: exit 0 (`logs/verify_iiac_addendum_3_manifest_at_head.log`) |
| 3 | Session ids distinct | PASS_SCOPED | implementation `T-IGX-036-J-V-PDC-X-IMPLEMENTERS+T-IGX-038-A-F-MCP-RC-IMPLEMENTER` (the addendum 2 label, kept although the delta is the `T-IGX-036-X` batch; a label, not an attested session); auditor `PDC-P7-00-ACT-N11-G6-BASELINE-AUDITOR-85fb061a-...-ADDENDUM-3` (the same auditor session as before plus the suffix; equals none of the MCP-RC auditors); certifier = the reserved designation, not an attested session. All three distinct. `process_enforced` attestation |
| 4 | Digests equal the published files, before and after the pin-only edit | PASS | 4 of 4 match the files at the head, at `d5c51df0e` and on the auditor branch `793cce5c8` |
| 5 | Round 4 record claims | PASS | see section 3 |
| 6 | MCP-RC contract, one pytest process, list from Python, `PYTHONPATH` = my checkout | PASS | **481 passed, 1 xfailed (strict, the TRANSFERRED `test_scaffold_defaults_mcp_to_opt_in[product]`), 0 failed, exit 0**; pytest 743.70 s, wall 809.84 s (slower than the 503 s of addendum 1 because the non-pytest verifiers and probes ran beside it) (`logs/rc_contract_certifier_at_bb78a7432.log`) | |
| 7 | Findings F-49..F-55 have a disposition | PASS_SCOPED | section 4: 7 of 7, no BLOCKER/CRITICAL; 0 HIGH / 1 MEDIUM / 2 LOW / 4 INFO as the auditor counts |
| 8 | Guardrails | PASS | no change to `scripts/governance/`, `.github/`, `MATERIAL_GOVERNANCE_PATHS`, P8, FRZ files or auditor files outside `addendum_3/` and the round 4 record; the activation record changed only by the two rebind commits `92d730821` and `9949fe2c3` (`logs/guardrails_and_pin_git_checks_a2.txt`); no AI-attribution trailer in `7801dcb12..HEAD`; the four product-path commits carry `Ref: governance/tickets/...`. One new `Approved: operator-delegate @ 2026-11-14T08:07:57+01:00` line in `governance/tickets/T-IGX-036-X.md` arrived with the `development` merge (a PDC-X plan approval, not mine) |
| 9 | Activation gate at the head | FAIL, expected | exactly the one `register.py` line (section 1) |
| 10 | Re-pin analysis from the gate code | PASS_SCOPED | section 5; the simulation with the gate's ancestry, product-diff and material-path rules is in `data/addendum3_checks.json` (`repin_simulation_at_head`) |

## 3. Addendum 3 manifest and the round 4 record (claims checked against git)

| Claim | Result |
| --- | --- |
| Base `0fc8badd8` is the tip of `origin/mcp-rc` at integration time and equals the audited head | **confirmed**: the evidence commit `d5c51df0e` sits directly on it, the manifest's three SHA fields are `0fc8badd8` |
| 44 files imported byte-for-byte, none outside `docs/`, none present on `0fc8badd8` | **confirmed**: 44 of 44 blob-identical to `origin/p7-00-baseline-audit` @ `793cce5c8`; the branch adds exactly these 44 files relative to `0fc8badd8`; relative to the current `origin/mcp-rc` it adds nothing (all 44 are already on it) |
| Pin-only edit changes three fields and nothing else | **confirmed** against `d5c51df0e` and against the auditor branch's manifest: `certifier_session_id` (`PENDING-CERTIFIER-PDC-P7-00` → `PDC-P7-00-CERTIFIER-PREFLIGHT-2026-11-13`), `evidence_publication_sha` and `certification_publication_sha` (zeros → `d5c51df0e`) |
| The manifest is not an input of any digest | **confirmed**: the four digests (report, scope, test matrix, findings) match the published bytes; the manifest's own SHA-256 differs from the row in the auditor's `EVIDENCE_MANIFEST.txt` (`aadfcd50...`, pre-edit) as expected (now `aa486c42809ace6be0be3628f2da16a31458d779052c5f28e914d418f7733be2`) |
| First-audit, addendum 1 and addendum 2 manifests and evidence untouched | **confirmed**: no change under `P7-00/` since `fde556183` outside `addendum_3/` and the record |
| `audited_implementation_sha` equals `audit_execution_sha`, so `--strict-product-diff` compares an identical pair | **confirmed** and it matters: the verifier proves nothing about `324d9dba3..0fc8badd8`; my own git diff of that range (section 1) does |
| Verifier PASS recorded | **reproduced**, both exit 0 |
| `certifier_session_id` is a reserved designation, not an attested session | **confirmed**; I am the same certifier designation. The `certification_publication_sha` in the addendum 3 manifest is `d5c51df0e`, not a preflight commit |

## 4. Findings F-49..F-55 and earlier findings

Dispositions are the auditor's (`iiac_audit_p7_00_addendum_3_findings.json`); my own check is in the last column. The register has a `findings` array and no open BLOCKER/CRITICAL (the gate's `_findings_errors` returns nothing for it).

| ID | Severity | Auditor's recommendation / status (verbatim) | My check |
| --- | --- | --- | --- |
| F-49 | MEDIUM | "orchestrator/operator decision: development is frozen on product paths, so a mirrored dispatch for openai would be a new product path and another pin; alternatively accept, record as debt for the domain audit IA-D3 and add a pinned test" | **reproduced at the head** (`logs/openai_sibling_probe_certifier_at_bb78a7432.json`): the row `openai` has categories `[managed_retrieval, embedding_provider]` and factory `create_openai_managed_retrieval`; `resolve(EMBEDDING_PROVIDER, "openai")` fails with `IntegrationConfigurationError` without a key and with a placeholder key with `TypeError ... returned OpenAIManagedRetrievalIntegration, expected EmbeddingProvider`. **Open.** Operator or orchestrator decision |
| F-50 | LOW | "test-only addition by the ticket owner (assert enabled default False and True pass-through; assert keyword arguments reach the owning factory); route to IA-D3; accept for the rebind" | not re-run (mutation sample not repeated); I read the product code: `enabled` and the keyword pass-through are exactly the behaviour the finding says is unpinned. **Open**, optional |
| F-51 | INFO | "accept for the rebind; address together with the runtime binding (B3) in IA-D3" | **confirmed by code read**: `bool(kwargs.get("enabled", False))`, other keywords dropped in the embedding branch |
| F-52 | LOW | "accept; a single-write form (wrap the factory before the one write) would remove the window if the file is touched again" | **confirmed by code read**: `augment_integration_contract_specs` then `register_integration(..., override=True)`; the window itself not re-probed. **Open**, accepted by the auditor |
| F-53 | INFO | "route the 210 to the owners of the integration provider tests; accept as attributed" | not re-run; my own status run (section 4a) is narrower |
| F-54 | INFO | "accept" | **confirmed**: `pyproject.toml` unchanged; `uv.lock` bumps `opentelemetry-*` 1.38.0 → 1.45.2 (plus two new exporter packages, `mistralai` 1.10.1 → 1.12.4, `importlib-metadata` and `zipp` dropped); `pdc_x_inventory.py` adds `ARTIFACT_LIFECYCLE` → `PDC-D8`; the `_package_c_v2_primitives.py` diff adds only the owning-row registration. Extras membership of the packages and `uv lock --check` not re-run |
| F-55 | INFO | "accept" | **confirmed by code read** of the one product path (section 1); the 201-pair resolve comparison not re-run |

**Resolved, stated plainly** (auditor status, with my own run in section 4a): **F-42** RESOLVED (the `[integration.embedding_provider]` cases of `test_ebh_5_behavioral_package_c_v2.py` pass; fixed by the harness registration order plus the dispatch of `T-IGX-036-X`); **F-07** RESOLVED (the inventory tests pass; `ARTIFACT_LIFECYCLE` classified to `PDC-D8`); **F-06** RESOLVED_UNCHANGED (the three EBH-5 gate files pass).

**Open, carried (auditor status verbatim where given):** **F-41** `OPEN_UNCHANGED` — "impact_map.py and the SSOT JSON are unchanged in this range; the eight applications/*/mcp/server.py modules still select no check ...; the XD-12 case and the p6_impact coverage test are red at 0fc8badd8 ... A rule is not part of this delta."; **F-35** `OPEN_UNCHANGED` — the same 10 COMPAT-X event-evolution gates fail at the head and at `324d9dba3` and pass at the pin; **F-09** `OPEN_REPRODUCED` — pytest does not exit after `test_xd_03_agent_governance.py`; **F-05** `UNCHANGED` ("red on the integrated tree through F-41"); **F-04/F-37, F-38, F-43, F-44, F-10** as the earlier dossiers state; MCP-RC carry-overs: owner of the strict xfail `[product]` (AF-05/AF-16, `T-IGX-039-A` draft), operator acceptance of RA-02, RA-03, RA-04, AF-05.

The auditor's list of what needs a decision: "F-49", "F-41 (carried, no rule at this head)"; test-only before merge or CI: "F-50 (optional test addition)", "F-41 test scope or map rule (decision)", "F-35", "F-09". The auditor's `product_change_required_for_activation` is empty.

## 4a. Test status at the head (my own run)

One pytest process (`scripts/run_status_a2.py`, `logs/status_f06_f07_f35_f41_f42_at_bb78a7432.log`), 259 tests in 10 files: **237 passed, 22 failed**, exit 1, 157.87 s. The 22 failures:

| Failures | Finding | Reading |
| --- | --- | --- |
| 2: `test_xd12_every_pdc_touched_product_module_maps_to_a_domain_check`, `test_every_pdc_touched_product_module_maps_to_at_least_one_check` | **F-41** | still red at the head, as the auditor says; green on `development` alone (addendum 1) |
| 17: 4 in `test_compat_x_event_evolution_r3_p0_gates.py`, 13 in `..._r3_r1_gates.py` | **F-35** | the same 17 as in addendum 1 (10 new since PDC-X plus 7 that are red at the pin as well); `unclassified persistence append intergrax/runtime/security/security_audit_trail.py:198`, inventory counts 7411 against 7356 |
| 3 in `test_ebh_5_behavioral_package_c_v2.py`, all `relational_store` / duckdb | environment | `mandatory dependency 'duckdb' missing`; the shared `.venv` lacks `duckdb`; these are the three duckdb cases the auditor also names, red at the pin |

Everything else passes: the **F-42** cases (the `embedding_provider` cases of `test_ebh_5_behavioral_package_c_v2.py`), the **F-06** files (`test_ebh_5_g06_enterprise_integrity_gates.py`, `test_ebh_5_proof_integrity_gates.py`, `test_ebh_5_rpl_f_integrity_gates.py`), the **F-07** file (`tests/unit/governance/test_pdc_x_inventory.py`) and the new `p6_ecapfix` tests. So F-42, F-07 and F-06 are resolved at the test level in my run too. F-09 (pytest does not exit after `test_xd_03_agent_governance.py`) is not exercised by this selection; the run used `os._exit` as before. What I could not run because of the environment: the three duckdb cases.

## 5. What the activation certifier has to change for the re-pin

Source: the gate (`iiac_activation_gate.py`, read in full), the activation certifier's report, its binding manifest, handoff and the record at `HEAD`. The simulation below uses the gate's own `_ancestor` rule, `product_paths_changed_between`, `_material_changes` and the `--strict-product-diff` rule of `verify_iiac_role_separation`.

**5.1 Which SHAs and why.** The gate fails today because `audit_execution_sha` `29d66ae3f` is older than the `register.py` change. A re-pin must move `audit_execution_sha` to a commit whose product diff to `HEAD` is empty, i.e. `0fc8badd8` or later. It must move `corrective_implementation_sha` too: the manifest's `audited_implementation_sha` must equal the record's `corrective_implementation_sha` (ACT-N09) and the role-separation verifier with `--strict-product-diff` fails if `audited_implementation_sha` and `audit_execution_sha` differ in a product path (candidate C below). Simulation at the head:

| Candidate | corrective | audit_execution | evidence | gate rules |
| --- | --- | --- | --- | --- |
| A | `0fc8badd8` | `0fc8badd8` | `d5c51df0e` | ancestry chain true, product diff audit→head empty, material paths empty, strict product diff empty: **passes** |
| B | `0fc8badd8` | `d5c51df0e` | `d5c51df0e` | the same, and it keeps the first rebind's pattern (`audit_execution_sha` = `evidence_publication_sha`): **passes** |
| C | `324d9dba3` | `d5c51df0e` | `d5c51df0e` | ancestry passes, but the strict product diff reports `register.py` (`ACT-N10 companion` failure): **fails** |
| D | current record values | `29d66ae3f` | `29d66ae3f` | the one `register.py` line: **fails (today)** |

I recommend B or A; the choice is the activation certifier's. `certification_publication_sha` must be a commit that is an ancestor of the new `effective_from_sha`, differs from `corrective_implementation_sha`, and equals the manifest's field; the commit that carries this addendum is the natural value (its SHA is in my report; a file cannot name its own commit).

**5.2 Fields of `IIAC_POLICY_ACTIVATION_RECORD.json`.**

| Field | Moves? | New value |
| --- | --- | --- |
| `corrective_implementation_sha` | **yes** | `0fc8badd8` |
| `audit_execution_sha` | **yes** | `0fc8badd8` or `d5c51df0e` (5.1) |
| `evidence_publication_sha` | **yes** | `d5c51df0e` |
| `certification_publication_sha` | **yes** | the commit that carries this addendum (or a later preflight commit) |
| `session_b_manifest_digest` | **yes** | SHA-256 of the new binding manifest bytes |
| `handoff_digest` | **yes** | SHA-256 of the new handoff bytes (CRLF kept) |
| `effective_from_sha`, `policy_activation_sha` | **yes, two phases** | phase 1 leaves `effective_from_sha` empty (the gate fails with the four known errors), phase 2 sets both to the phase-1 commit; the gate compares the ten `BINDING_KEYS` between the phase-1 blob and the record |
| `notes` | should | free text; it names the audited tree `324d9dba3` and the previous pin; no ACT-N18 pattern may appear |
| `governance_implementation_sha`, `session_b_verdict` (`PASS`), `session_b_governance_audit_report_digest` (`c405c65f...`), the five bound paths (report, scope, matrix, findings, manifest), `handoff_record_path`, `operator_decision_path`, `operator_decision_id`, `operator_activation_line`, `operator_decision_class`, `status`, `policy_version` | **no** | unchanged |

The four bound `IIAC-GOV-1` artifacts (report, scope, matrix, findings) keep their bytes, so the manifest digests of them do not move and the report bytes still equal the bytes at the new evidence SHA (ACT-N11 report check).

**5.3 Binding manifest** (`iiac_evidence/PDC-X/MCP-RC/activation/iiac_audit_session_manifest.IIAC-ACT-N11-MCP-RC-ACTIVATION.json`), `verdict` stays `PASS` (ACT-N05):

- `audited_implementation_sha`, `test_execution_sha` → `0fc8badd8`; `audit_execution_sha`, `evidence_publication_sha`, `certification_publication_sha` → as in 5.2 (the gate compares `audited_implementation_sha` with the record's corrective SHA, `audit_execution_sha` with the record, `certification_publication_sha` with the record).
- `auditor_session_id` → the `-ADDENDUM-3` id (the auditor who audited the tree that is being bound). If it stays at `-ADDENDUM-2` the gate still passes (it needs only the id to appear in the handoff), but it would name an auditor of a different tree.
- `implementation_session_id`: the addendum 3 manifest keeps the addendum 2 label; `register.py` came from `T-IGX-036-X`. A label question for the activation certifier; the gate only needs it to differ from the auditor id.
- `certifier_session_id`: the activation certifier's own id (the same or a new session; the gate needs it non-empty).
- `product_session_manifest_paths`: add the addendum 3 manifest.
- `limitations`: scope and audited tree, the "Drift" limitation (it says `a156cbbd5` and `324d9dba3`), F-42 (now resolved by the auditor), F-49..F-55 (F-49 open), F-41 still open, the list of unverified CI contexts, the preflight addenda.
- Unchanged: `stage_id` `IIAC-GOV-1`, the four digests, `activation_audit_report_path`, `handoff_record_path`, `verdict`.
- After the edit both verifiers (`verify_iiac_audit_manifest --require-handoff-template`, `verify_iiac_role_separation --strict-product-diff`) have to be run again on it.

**5.4 Handoff record** (`.../activation/IIAC_SESSION_HANDOFF_RECORD_PDC_X_MCP_RC_ACTIVATION.md`): it contains no `-ADDENDUM-3` string today (0 occurrences), so with the new `auditor_session_id` the gate's ACT-N17 check (`auditor id in handoff text`) would fail until the handoff names it. Update the header ids, the "All sessions" table (addendum 3 auditor; preflight addenda 1 and 2 of the certifier), the "Binding" SHAs and the evidence publication SHA. Keep the CRLF line endings (the `iiac_evidence/** -text` rule and the fresh-clone tamper test depend on it). Its digest goes into `handoff_digest`.

**5.5 Signed decision text and the signature.**

What the gate compares in `IIAC_OD_ACTIVATION_MCP_RC_2026-11-13.md` (`_operator_errors`, `closure_conflicts`, and the file-name and root rules of `resolve_evidence_path`): `Decision status: AUTHORIZED`, `Decision ID: <record's operator_decision_id>`, `Policy version: IIAC-2026-11-05.1`, `Decision scope: IIAC policy activation`, `Audit report digest: <record's session_b_governance_audit_report_digest>`, the record's `operator_activation_line` present as a whole line and matching the `Approved IIAC policy activation IIAC-2026-11-05.1: operator @ <timestamp>` form, none of `________________`, `NOT SIGNED`, `Decision status: PENDING`, and no ACT-N18 closure pattern. **It compares no SHA, no path count and no audited tree.** The file is not covered by a digest field. Since the re-pin keeps the `IIAC-GOV-1` report set, the report digest line (`c405c65f...`) stays true. So **the gate does not need a second signature, and does not need the decision file to change at all**.

What the re-pin makes untrue in the signed text (the file stays byte-identical if not edited): the scope paragraph and the "Audited tree" rows (`324d9dba3`), "`git diff --name-only 324d9dba3 HEAD -- intergrax agents applications` is empty", "`origin/development` has no product-path change after `a156cbbd5`" (false since `f209febbe`), the audit table and digest table (they end at addendum 2 and preflight addendum 1; no addendum 3, no preflight addendum 2), open condition 2 (F-42, now resolved per the auditor) and open condition 8 (the drift that occurred). The text itself expects this ("If an addendum 3 is made ..., a further round follows and the SHAs in this record move").

`DELEGATION-MCP-RC-2` is single use and covers only the unsigned text with SHA-256 `a4deb39c...`; "if the decision is revised afterwards ... a new addendum records the new digest and commit", the delegation names a draft that is not current as one the designated session "does not sign", and its renewal window ends `2026-11-15T08:15:29+01:00`. The delegation does not cover an edited text; after the signature was written (`2026-11-14T08:21:13+01:00`) it is used up. `OD-MCP-RC-04` section 6, the operator's recorded chat answer (Polish label `Poczekaj na koniec B-P6, jeden addendum 3 (Recommended)`, English rendering by the orchestrator, not an operator-written line), describes the re-pin as "a new two-phase update" by the activation certifier and mentions no new signature, and it does not say whether the existing signature stands for a different audited tree either.

So there are two consistent paths, and the choice is the operator's, not mine and not the gate's:

- **(a) The signature stands.** The signed file stays byte-identical (SHA-256 `0e0aed58...`). The operator confirms in chat that the signature of `2026-11-14T08:21:13+01:00` stands for the re-pin to `0fc8badd8` (one added product path, already inside the 158), and that is recorded as a new `OD-MCP-RC-05` by the orchestrator (a recorded answer, not a signed line). The activation certifier states the stale statements of the signed text in the binding manifest `limitations` and in the handoff. This is mechanically sufficient.
- **(b) A fresh signature.** The decision text is revised (scope, tree, audits, digests, open conditions) and re-digested; a new addendum to `DELEGATION-MCP-RC-2` or a new delegation records the new digest and commit (the operator's renewal; the `2026-11-15T08:15:29+01:00` limit applies); a designated signing session writes a new line; the record's `operator_activation_line` moves with it. If the old line stays in the file next to a new one, the gate accepts either as long as the record's line is present; two lines in one file would be confusing, so a new decision ID or a new file name (`IIAC_OD_ACTIVATION_*.md`) is cleaner. This is more work and not needed by the gate.

I cannot determine from the repository which of the two the operator intends; I did not ask the operator. I would ask the operator that single question before the activation certifier starts.

**5.6 Remaining order** (not mine): operator answer on 5.5 → activation certifier session: new binding manifest and handoff (5.3, 5.4), both verifiers, the contract run, phase 1 (gate fails with the four known errors), phase 2 (gate passes) → PR → the five CI contexts on the pinned head → merge (operator) → terminal verification. Any product or `scripts/governance/` commit on `development` after `146e7acf1` voids the pin again. At my fetch `origin/development` is `8d90d345b`: the product diff `146e7acf1..8d90d345b` is empty; non-`docs/` changes are `.github/workflows/unit-tests.yml` (EBH-7 default SHAs re-pinned to `146e7acf1`) and three `tests/qualification/ebh_7/` files; none is in `MATERIAL_GOVERNANCE_PATHS`; `mcp-rc` does not contain them yet.

## 6. What this addendum does not cover

1. **Not an audit.** I audited no product path. The substance of F-49..F-55 is the auditor's; I reproduced F-49 and read the one product diff.
2. **CI contexts.** `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast` and `ent-audit-pr-gate` were not run by me in this addendum; the addendum 1 limits stand (no claim that any is green). New fact only: the F-42 failure, which addendum 1 found inside `ent-audit-fast` through `VBN-REACHABILITY`, is resolved at the test level (section 4a), so that known red element is gone; F-41, F-35 and F-09 are in none of the five selections.
3. **F-09, F-50, F-52, F-53, F-55 not re-run;** the auditor's runs are not reproduced.
4. **The gate pass after the re-pin is predicted, not run.** The simulation in 5.1 uses the gate's own rules for ancestry, product diff, material paths and strict product diff; the complete gate (digests, handoff, operator file, phase-1 blob) can only run on the real re-pin.
5. **Freeze.** The freeze is a declaration; `origin/development` moved again during this window (`8d90d345b`, no product-path change). `mcp-rc` has not merged it.
6. **Independence** is a `process_enforced` attestation (one git identity signs all commits); session ids of implementers and certifiers are labels.

## 7. Remaining steps (not mine)

Operator: decision on 5.5, decision on F-49 (accept as debt for IA-D3 with a pinned test, or another product path and another pin), F-41 (rule on `development`, which voids addendum 3 for the two impact-map paths, or accept the red tests), owner of F-35 and F-09, the MCP-RC carry-overs. Activation certifier: 5.2 to 5.4, two-phase rebind. Then PR, five CI contexts on the pinned head, merge, terminal verification.
