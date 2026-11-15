# PDC-X P7 — IA-D6 independent audit report: D6 RUNTIME-SECURITY

| Field | Value |
| --- | --- |
| **Report** | PDC-P7-D6 (IA-D6), batch audited: B-D6 / T-IGX-036-P (PR #494, merge `1f6dc3c4f`; implementation commits `adaaa374e`, `453cc10ce`) |
| **Audited SHA** | `c08efe46a74a20595468d8d8bf06eab2dca5f573` (product tree identical at execution HEAD, see Method) |
| **Execution HEAD** | `b7889220607b462195308fa186f4ba66fb78f3a8` (descends from the audited SHA; the only differences are docs and `governance/tickets/T-IGX-038-*.md`, no product, test or script path) |
| **Auditor** | IA-D6, a clean session. It did not implement, plan, sign or certify any D6 item. Roles: IMPLEMENTER != INDEPENDENT AUDITOR != CERTIFIER ([`OD-PDC-IIAC-SCOPE`](operator_decisions/OD-PDC-IIAC-SCOPE.md)). |
| **Scope** | The 28 requirement rows `PDC-D6-R001..R028` of `PDC_DOMAIN_REQUIREMENTS_MATRIX.json`, `PDC_OWNERSHIP_MATRIX.md`, ADR-PDC-06, the B-D6 plan in `PDC_IMPLEMENTATION_WBS.md`, the F-35 classification of `EvidencePlaneSecurityAuditTrail` (T-IGX-040-D, PR #557), and the P7-00 dossier findings routed to IA-D6 |
| **Evidence** | `docs/project/maintainers/qualification/iiac_evidence/PDC-X/D6/` (manifest, findings JSON, scope, test matrix, logs, probe and mutation scripts) |
| **Overall verdict** | **PARTIAL** (IIAC manifest value `CONDITIONAL`). 0 HIGH, 6 MEDIUM, 9 LOW, 5 INFO. 14 rows PASS, 3 PASS_SCOPED, 11 PARTIAL, 0 FAIL. |

The auditor edited no product, test, script, ticket or governance file. All probes and mutations ran in memory or against temporary directories under `.tmp/`.

## 1. Method

1. **Reading.** OD-PDC-IIAC-SCOPE, the D6 rows (JSON, Ownership, ADR-PDC-06, WBS B-D6 and ticket T-IGX-036-P). The P7-00 dossier and addenda 1-4 were used as leads only. No other domain's audit report was read.
2. **Code.** Owner paths read for every row that carries a product claim (see §3). The B-D6 diffs `adaaa374e` and `453cc10ce` were read in full for the changed owner modules.
3. **Tests, one pytest command at a time** (outputs under `.tmp/session/ia-d6/`, copied to `logs/`):

| Command (narrow path) | Result |
| --- | --- |
| `tests/qualification/pdc_x/d6` | 134 passed, 104 s, process exited normally |
| `tests/qualification/pdc_x/xd/test_xd_02…, test_xd_06…, test_xd_10…` | 19 passed |
| `tests/qualification/governance` | 403 passed, 139 s |
| `tests/qualification/tenant_x` | **1 failed, 193 passed** (F-D6-06) |
| `tests/qualification/prod_q` | 445 passed, 2 skipped (POSIX or job-object only), 291 s; **the process did not exit after the summary** and was killed by the 590 s timeout (F-D6-18) |
| `tests/unit/runtime/sandbox` + `tests/integration/runtime/sandbox/reference_substrate` | **1 failed**, 157 passed, 11 skipped (physical egress needs Linux/WSL2) (F-D6-07) |
| `tests/unit/runtime/architecture/test_audit_ideal_depth_gate.py` | 70 passed, 2 failed (`29_1`, `30_4`): the known pre-existing reds (P7-00 F-08), not D6 cases. The D6 cases `4_1` and `23_1` pass. |
| `tests/qualification/mcp_server_x/…l2_security_trace, …l3_invoke_security` | collection errors, `fastmcp` is not installed in this environment (F-D6-19) |
| `scripts/maintenance/check_critical_action_signing.py`, `check_immutable_security_audit_trail.py`, `check_tool_injection_defense.py` | all three OK |

4. **Adversarial probes** (own scripts `adversarial_scripts/ia_d6_probes*.py`, 21 probes) and a re-run of the P7-00 D6 probe script as a reproduction of its leads (`logs/lead_reproduction_p7_00_probes_d6.out.json`).
5. **Runtime mutation sweep** (`adversarial_scripts/ia_mut.py`, `run_mutants.sh`): 12 in-memory mutants applied through a pytest plugin, each run against the matching D6 test files. **All 12 were killed** (signing literal fallback, poisoning default off, tool-guard no-op, provenance allow-all, demo seed ignores strict, broker scope check no-op, delegation over-reach, secret gate blind, audit secret validation no-op, audit trail not durable, strict validator always clean, bridge always lab). The D6 tests can fail. Two redundant tenant re-checks stay unpinned (P7-00 F-29, not re-run).
6. **Product tree identity.** `git diff --name-only c08efe46a HEAD` lists only `docs/**` and `governance/tickets/T-IGX-038-A..F.md`.

## 2. Findings summary

| ID | Sev | Rows | Title |
| --- | --- | --- | --- |
| F-D6-01 | MEDIUM | R020 | F-35: `EvidencePlaneSecurityAuditTrail` is a legitimate persistence site, but its `CANONICAL_VALIDATING_PATH` classification rests on a composition assumption the code does not enforce |
| F-D6-02 | MEDIUM | R020 | The "immutable, replicated" audit trail report is bookkeeping: deleting stored entries leaves `replicated=True, immutable=True` and nothing detects it |
| F-D6-03 | LOW | R020, R007 | Audit trail `list_entries` returns the oldest 1000 entries only; secret-shaped values under benign keys are stored (re-verified P7-00 F-20) |
| F-D6-04 | MEDIUM | R008, R009, R016, R020, R024 | Hardening has no production composition: signing, audit trail, strict-config validator, plugin provenance gate and the credential broker with its admission port have no non-test caller (re-verified P7-00 F-04) |
| F-D6-05 | MEDIUM | R016 | Plugin provenance is bound to the operator mapping key, not to `attestation.distribution`, and the installed files are never hashed (re-verified P7-00 F-21) |
| F-D6-06 | MEDIUM | R001, R005 | TENANT-X carrier-count gate is red at this SHA (831 vs pinned 819); not covered by any earlier audit |
| F-D6-07 | MEDIUM | R010, R011, R012 | The sandbox unit suite has a red governance-deny test, and the physical egress tests are skipped on this host |
| F-D6-08 | LOW | R013 | Tool-input guard is a four-regex heuristic with simple evasions; the middleware passes an empty `tool_id` and the default allow-list is empty (allow all) |
| F-D6-09 | LOW | R014 | Poisoning default differs by path: the plan-context path stays off in non-production mode when the profile is absent |
| F-D6-10 | LOW | R024, R008 | Strict-config validator and signing wiring accept weak or disabled settings; strict mode on a lab profile is unchecked |
| F-D6-11 | LOW | R005, R001 | A resource with no tenant is accessible to any scoped tenant; `RequestIdentity.tenant_id` is not normalised |
| F-D6-12 | LOW | R006 | The credential resolver keeps the provider exception, with its text, as `__cause__` |
| F-D6-13 | LOW | R016, R015 | Security-defense plugin admission defaults to structural-only and `isolate` on load failure (a failed defense plugin is silently absent) |
| F-D6-14 | INFO | R007 | `raw_secret_gate` evasions (re-verified P7-00 F-30) |
| F-D6-15 | INFO | R027 | Lab profiles use the public demo seed implicitly; the seed is still importable (re-verified P7-00 F-23) |
| F-D6-16 | LOW | R021 | Satellite wording is stale or imprecise (link note, definition of "strict", coverage depth) |
| F-D6-17 | INFO | process | Both B-D6 implementation commits carry a forbidden AI `Co-Authored-By` trailer |
| F-D6-18 | INFO | process | `prod_q` pytest process does not exit after its summary |
| F-D6-19 | INFO | R017 | MCP exposure security not independently exercised (BLOCKED row, `fastmcp` absent) |
| F-D6-20 | LOW | R015, R019 | R015 and R019 are proven for credentials, tool allow-list and three XD pairs, not for tenant data sent to a model provider or for all of D1-D8 |

### F-D6-01 (MEDIUM): F-35 classification of the evidence-plane audit trail

**Judgement.** The site is a legitimate persistence site in kind. `EvidencePlaneSecurityAuditTrail.append` (`intergrax/runtime/security/security_audit_trail.py:198-213`) builds a fresh `RuntimeEvent` from live arguments with a registered typed payload (`security.audit_entry.v1`), tenant-scoped, and appends it through `EvidencePersistencePort`. It does not reconstruct persisted events. The COMPAT-X rule is narrow (module path, function `append`, callee `self._evidence.append`) and the negative tests in `test_compat_x_event_evolution_f35_security_audit_trail_site.py` fail an unclassified site, another function, another callee and another module. Both gate runs are clean.

**Defect.** The label `CANONICAL_VALIDATING_PATH` claims the append goes through the validating port. The code does not enforce that. The class and `resolve_security_audit_trail_wiring` accept any object and never normalise it; `event_bus.py` and `orchestration_governance_evidence_composition.py` both call `as_evidence_persistence_port(...)` for exactly this reason.

**Reproduction.** Probe `A1_trail_and_wiring_accept_non_validating_port` (`logs/ia_d6_probes.out.json`): a recording port that is not a `CanonicalRuntimeEventWriteValidatedPort` is accepted by the product wiring (`enabled=True`), and both bootstrap events are written to it without `prepare_canonical_production_write_event` or `assert_runtime_event_schema`. The structural gate cannot see this because it matches names only.

**Suggested repair.** Normalise in `EvidencePlaneSecurityAuditTrail.__init__` with `as_evidence_persistence_port` (or reject a port that is not validated), add a hostile-port test, and keep the COMPAT-X rule. Owner: `intergrax/runtime/security/security_audit_trail.py`, `intergrax/applications/_shared/security_audit_trail_wiring.py`, `tests/qualification/compat_x/…f35…`. Domain: D6 ticket, a different session.

### F-D6-02 (MEDIUM): audit-trail immutability and replication are not evidence

**Reproduction.** Probe `A2_tamper_evidence_delete_rows_in_durable_store`: with the durable SQLite evidence plane, deleting two `runtime_events` rows directly makes `list_entries` return 3 instead of 5 entries, while `verify_replication()` still reports `replicated=True, immutable=True`.

**Cause.** `MultiRegionSecurityAuditTrail` (`multi_region_audit_trail.py`) keeps per-region lists of entry ids in the same process and appends the same id to every list in lockstep. `verify_replication` compares those lists with each other and with a sealed prefix of the same lists. The entries carry no hash chain or signature. The wiring forces two or more region names, so the configuration looks replicated while no replica exists. The satellite says region replication of the evidence store is not performed (§5, §6), but the report object and the name `immutable` say otherwise.

**Suggested repair.** Either add tamper evidence (hash chain over entries or a signed head) and verify it in `verify_replication`, or rename the report fields and the wiring result so they do not claim immutability or replication. Owner: `intergrax/runtime/security/multi_region_audit_trail.py`, `security_audit_trail.py`, `security_audit_trail_wiring.py`, tests `tests/qualification/pdc_x/d6/test_d6_audit_trail_durable.py`.

### F-D6-03 (LOW): list truncation and value secrets

Probe `A3`: 1003 entries appended, 1000 listed, the newest is not listed (`limit=1000`, oldest first). Probe `A4`: a metadata value whose `str()` is a bearer token is stored (only secret-like key names are rejected). Suggested repair: page or return the newest entries, and apply the value-level secret patterns of `raw_secret_gate` to metadata values.

### F-D6-04 (MEDIUM): no production composition of the D6 controls

At the audited SHA the following have no caller outside tests and scripts: `resolve_critical_action_signing_wiring`, `resolve_security_audit_trail_wiring`, `assert_strict_security_configuration` and `validate_strict_security_configuration`, `require_plugin_provenance` and `PluginProvenanceAdmission`, `ScopedCredentialBroker` (its only non-test user is `intergrax/integrations/qualification/reference_scoped_adaptive_integration_execution.py`) and any implementation of `CredentialScopeAdmissionPort` (the Protocol is the only product definition; the XD-06 test says so itself). `sign_critical_action` is called only to self-sign a bootstrap payload. The fail-closed behaviour is therefore a property of the functions, not of a composed host. `bootstrap_catalogs` keeps `discover_entry_points=False`, which limits the exposure for plugins. The decision whether host composition is a PDC-X claim stays with the orchestrator (as in P7-00 F-04); the audit records that R009 (revocation) holds only through a test-side admission port.

**Suggested repair scope.** Either wire the controls in the Tier-3 product host factory (`intergrax/applications/_shared/…` composition roots) with an integration test through the real host, or record them in the satellite and the matrix as library-level only and downgrade the rows.

### F-D6-05 (MEDIUM): provenance binding

`PluginProvenanceAdmission._attestation_for` finds the attestation by the operator's mapping key, then `evaluate` checks the installed version of `attestation.distribution`, never of `spec.distribution`. Probe `PV1`: an attestation for `other-signed-plugin` stored under the key `victim-plugin` admits the `victim-plugin` entry point. Probe `PV2`: installed files are never consulted, so a tampered install with the attested version string passes (the satellite §6 discloses the second limit). Suggested repair: require `normalize(attestation.distribution) == normalize(spec.distribution)` and add a hostile-mapping test; decide whether a digest check of the installed distribution is in scope. Owner: `intergrax/core/plugins/admission.py`, `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py`.

### F-D6-06 (MEDIUM): TENANT-X carrier gate red

`tests/qualification/tenant_x/test_tenant_x_p0_current_head.py::test_tx_p0_r3_head_q02_class_field_carrier_count_stable` fails: `831 == 819` (`TENANT_X_REPLAY_HEAD_CARRIER_TOTAL` in `_replay_x_carrier_reconciliation.py`). The other 193 tests pass. A diff of `tenant_id` fields since the pin shows the 12 new class-field carriers come from PDC-X paths of other domains (evaluation contracts, retention sweep, artifact refs, recurrence, queue contracts), not from D6 files. The P7-00 audits did not run `tenant_x`. R001 and R005 are "requalify by replay" rows, so their replay evidence is not green. This is not on the known-red list. Suggested repair: test-side reconciliation (classify the 12 carriers in `REPLAY_DELTA_CLASS_FIELD_SURFACE_IDS` and bump the pinned total) by the owner of the TENANT-X replay gate, or an Operator decision on the stale pin.

### F-D6-07 (MEDIUM): sandbox evidence

`tests/unit/runtime/sandbox/test_cap_c1c_sandbox_evidence_and_contract_purity.py::test_c1c_q_governance_deny_zero_physical_execute` fails with `RuntimeError: active execution governance identity required` raised from `record_governance_policy_decision_evidence_for_active_identity` (called from `RuntimeToolInvoker._require_current_attempt_authorization`), where `DeclarativePolicyViolationError` is expected. Neither the invoker nor the test changed during PDC-X (last changes in Oct and Sep 2026), so this is a pre-existing red or stale test setup; it is the only unit proof that a governance deny runs zero physical sandbox execution. The behaviour is still a denial (an exception before execution), but the typed error contract and the proof are not green. The 11 reference-substrate physical egress tests are skipped on this Windows host (Linux/WSL2 only), consistent with the declared C10/C15 exception, so physical egress is not proven here. Suggested repair: give the test an active governance identity (or fix the error contract) in the sandbox test owner's ticket. Owner: `tests/unit/runtime/sandbox/…c1c…`, `intergrax/runtime/nexus/tools/invoker.py`.

### F-D6-08 to F-D6-13, F-D6-16, F-D6-20 (LOW)

- **F-D6-08.** `assert_tool_input_safe` has four regexes. Probe `T1`: not blocked: "ignore all previous instructions", "disregard the prior instructions", zero-width and Cyrillic-homoglyph variants, "assistant:", base64 and full-width colon; blocked: the four literal forms. The middleware returns ALLOW for an empty `tool_id` even with an injection argument (probe `T2`), and `default_tool_invocation_policy()` has an empty allow-list, which means allow all tools (tool authorization is PolicyEngine's job). The ADR requires one defense path, which exists; the strength is limited. Owner: `intergrax/runtime/security/tool_injection_guard.py`, `intergrax/applications/_shared/application_security_wiring.py`.
- **F-D6-09.** `plan_context_invocation.run_rag_context` keeps the filter on only when `production_mode` is true; otherwise an absent profile means off (probe `R1`, pinned by `test_d6_r014_plan_context_non_strict_behaviour_is_unchanged`). The `rag.retrieve` tool defaults to on in every mode. ADR-PDC-06 says "defaults to enabled when the profile is absent" without that scope. Align the ADR text or the code.
- **F-D6-10.** Probes `S1` to `S3`: a one-character signing secret is accepted on a product profile; a product profile with signing, tool defense, poisoning defense and the audit trail all disabled returns no strict-config issue; a lab profile in STRICT mode is not validated; `plugin_entry_points_enabled` is a caller-supplied flag defaulting to False. R024 (fail-closed security defaults) holds for the listed missing items only.
- **F-D6-11.** `tenant_scope_is_valid("a", None, allow_unscoped=False)` is True, and `TenantSecurityMiddleware` lets a task with no resource tenant through (probes `TS1`, `TS2`). `RequestIdentity(tenant_id="  tenant-a  ")` keeps the padding while the scope and audit code strip it (probe `ID1`). Confirm that untagged resources are intended to be tenant-open.
- **F-D6-12.** Probe `CR1`: when the secrets store raises with secret text in its message, the sanitised `CredentialProviderUnavailableError` message is clean, but the chained `__cause__` carries the text, so `logging.exception` or a full traceback prints it. Use `from None` or sanitise the cause.
- **F-D6-13.** `SecurityDefenseAdmissionPolicy()` defaults to `provenance=None` and `on_load_failure="isolate"` (probe `PL1`): a defense plugin that fails to load is dropped and the host continues without it. Discovery is off by default (probe `BS1`).
- **F-D6-16.** The satellite link note says hub docs link to a missing `IDENTITY_TRUST.md`; the hub links were repointed in P5-DOC and resolve. It defines "strict" as `PRODUCT` only while the attestation bridge and R027 tests also treat `ExecutionMode.STRICT` as strict. Rows for D5, D7 and D8 are "owner-elsewhere" pointers, so R021 is a coverage map over the EE-B3 threat model.
- **F-D6-20.** `test_xd06_provider_boundary.py` states that the context-plane egress sensitivity policy belongs to D2 and is not exercised there. R015 as worded (tenant data to an unapproved provider) is therefore proven only for credentials, tool allow-list and tenant intake. R019 is "across D1-D8" but the cited cases are XD-02, XD-06 and XD-10.

### INFO findings

- **F-D6-14.** `raw_secret_gate` is heuristic; probe `G1` and the P7-00 matrix: string concatenation, f-strings, bytes, a split `sk-` token and `or "literal"` fallbacks escape. It catches the known forbidden literal, secret-named literals, env-default literals and credential formats, and the product tree is clean. It runs as a pytest test, not as a separate CI or pre-commit step.
- **F-D6-15.** With a LAB profile in non-strict mode, `build_boundary_event_buffer` signs with the public demo seed when no key is set; `POC_ATTESTATION_DEMO_SIGNING_SEED` remains importable through the module `__getattr__`; `build_deterministic_test_attestor` defaults to `strict_profile=False`. `ApplicationProfile` has only LAB and PRODUCT, so this is the named lab path; accepted.
- **F-D6-17.** Commits `adaaa374e` and `453cc10ce` carry `Co-Authored-By: Claude …`; 302 of 981 commits since 2026-11-01 do. `agent-env/rules/intergrax-no-ai-attribution.md` forbids this. The history is merged; the audit records it only.
- **F-D6-18.** The `prod_q` pytest process stayed alive about 5 minutes after "445 passed" and exited 124 under the timeout. The D6, governance, tenant_x and xd runs exited normally, so the R-05 fix holds for the suites run here; the full `pdc_x` tree was not run.
- **F-D6-19.** R017 stays BLOCKED on MCP-SERVER-X P8. The two MCP security test modules cannot be collected here (`ModuleNotFoundError: fastmcp`); no MCP security claim is made.

## 3. Per-row verdicts

Vocabulary: PASS, PASS_SCOPED (Operator-authorised exclusions only), PARTIAL, FAIL. "T" = D6 or cited tests run and green at this SHA.

| Row | Verdict | Evidence and reason |
| --- | --- | --- |
| R001 Principal and workload identity | PARTIAL | `RequestIdentity` read; governance 403 green; `tenant_x` red (F-D6-06); padding not normalised (F-D6-11) |
| R002 Authentication (architecture) | PASS_SCOPED | `harness_principal.py` read; `prod_q` 445 passed (process hang F-D6-18); live proof is Path B (`OD-PDC-PATHB-01`) |
| R003 Single policy authority | PASS | `PolicyEngine` facade over `runtime/policy`; governance 403 green; code read is structural only (§5) |
| R004 Delegation narrowing | PASS | Probe `D1`: over-reach, case, whitespace, wildcard and empty scope denied; child cannot exceed parent; mutant M7 killed; T |
| R005 Tenant isolation (FRZ-SEC-01) | PARTIAL | `tenant_x` 193/194, carrier gate red (F-D6-06); unscoped resource open (F-D6-11) |
| R006 Secret references, late resolution | PASS | Resolution only inside `ScopedCredentialBroker`/resolver at use; `repr` redacted; mutants M6 and M8 killed; T; see F-D6-12, F-D6-04 |
| R007 Raw secrets never cross layers/logs | PARTIAL | Static gate green and effective on the known shapes (M8 killed), caplog and evidence scans pass; heuristic gate (F-D6-14); audit metadata accepts secret values (F-D6-03) |
| R008 Signing fails closed | PASS | Literal removed repo-wide (grep); product missing secret raises; lab opt-in rejected on product; M1 killed; script OK; weak secrets and no consumer noted (F-D6-10, F-D6-04) |
| R009 Credential scope and revocation | PARTIAL | Broker denies mismatched tenant/host/expired and honours a revoking admission on every retry (T); no admission implementation and no composition in product (F-D6-04) |
| R010 Sandbox enforcement | PARTIAL | Gate fails closed when isolation is required and absent (unit tests); live is Path B; red deny test (F-D6-07). The gate checks presence of a session/host, not isolation strength. |
| R011 Filesystem and process isolation | PARTIAL | Sandbox unit suite 157 passed, 1 failed (F-D6-07) |
| R012 Network egress controls | PASS_SCOPED | `test_network_egress_contract.py` and reference-substrate logic green; physical egress skipped (Linux/WSL2) under the declared C10/C15 exception |
| R013 Tool-input guard has a consumer | PASS | Guard called in `ToolInjectionDefenseMiddleware`; AST test finds exactly one consumer; M3 killed; strength limited (F-D6-08) |
| R014 Poisoning defense fails closed | PASS | Absent or malformed profile means on in the RAG tool and in strict plan-context; M2 killed; scope note F-D6-09 |
| R015 Exfiltration across provider boundaries | PARTIAL | Credential egress scope, tool allow-list and tenant intake proven; tenant data to a model provider is a D2 control not exercised in D6 (F-D6-20) |
| R016 Plugin entry-point provenance | PARTIAL | Real Ed25519 verification, unsigned rejected, provenance checked before import; M4 killed; binding and digest gaps (F-D6-05), no production gate (F-D6-04), defense-loader defaults (F-D6-13) |
| R017 MCP exposure security | PARTIAL | BLOCKED row; not audited here (F-D6-19) |
| R018 Side-effect governance | PASS | `runtime/policy` side-effect authorization; governance 403 green; structural read only (§5) |
| R019 Cross-tenant negatives | PASS | XD-02, XD-06, XD-10: 19 passed, tests use real stores, queue, broker and governance pipeline; coverage note F-D6-20 |
| R020 Durable security audit trail | PARTIAL | Durable across restart (T, script OK, M10 killed); F-D6-01, F-D6-02, F-D6-03, F-D6-04 |
| R021 Threat model D1-D8 | PASS | Satellite links resolve; surface table present; F-D6-16 |
| R022 Retention/deletion/redaction | PASS | XD-10 TTL sweep per tenant as worker work, nothing deleted early; scope is artifacts (F-D6-20) |
| R023 Encryption/residency matrix | PASS | Satellite §5 published; owner split matches code (no encryption claim) |
| R024 Fail-closed security defaults | PARTIAL | Validator reports the listed missing items (M11 killed); weak settings pass, never composed (F-D6-10, F-D6-04) |
| R025 HARNESS-FINAL wording | PASS | Scoped note added under the table (`HARNESS_FINAL_CURRENT_HEAD_CERTIFICATION.md` line 85), FRZ-SEC OPEN stated |
| R026 Production auth-bypass live proof | PASS_SCOPED | Excluded by `OD-PDC-PATHB-01` |
| R027 Demo seed only in lab/proof | PASS | Strict path raises, no implicit fallback, bridge and gate tests; M5 and M12 killed; F-D6-15 |
| R028 GR-10 gates | PASS | `tests/qualification/governance` 403 passed; the "13 source mutations" claim was not re-run (§5) |

## 4. Review of earlier findings routed to IA-D6

| P7-00 ID | State at this SHA |
| --- | --- |
| F-04 | Re-verified and extended: F-D6-04 |
| F-10 | D8 path (governed read), not re-audited here |
| F-20 | Re-verified: F-D6-03 (truncation, value secrets); the replication point is stronger as F-D6-02 |
| F-21 | Re-verified: F-D6-05 |
| F-23 | Re-verified, downgraded to INFO: F-D6-15 |
| F-29 | Not re-run; the 12 D6 mutants killed here are different mutants |
| F-30 | Re-verified: F-D6-14 |
| F-35 | Judged: F-D6-01 |
| F-44 | Not examined (impact-map coverage, out of the D6 requirement rows) |

Known reds were not re-reported: EBH-7 identity gate, the five `roadmap_replay_x` tests, `dist_scale_x`, Docker scope, the two `test_audit_ideal_depth_gate` reds.

## 5. Limitations (what this audit did not cover)

- R003 and R018: judged from the facade and the 403-test governance suite plus a structural read; `runtime/policy` and `runtime/governance` internals were not audited line by line.
- R028: the claim "13 source mutations rejected" was not repeated.
- R017 and any MCP security: not exercised (`fastmcp` missing).
- Physical sandbox egress, live provider and live auth proof: skipped, Path B or the C10/C15 freeze.
- D2/D8 portions of R015 and R022: only through the cited tests.
- Not run: the full `tests/qualification/pdc_x` tree, `compat_x`, `agent_distribution`, `memory` suites, `ruff`, `check_intergrax_no_applications_imports.py` (D6 files were checked by grep: no import of `agents` or `applications` from the D6 modules).
- Tests ran on Windows with one worktree environment; no network.
- The P7-00 mutation findings (F-29) were used as leads only.
- Nothing here certifies, promotes a FRZ item or claims production readiness.

## 6. Repair routing (not done by the auditor)

Per OD-PDC-IIAC-SCOPE a different session repairs in a D6 ticket, then a fresh auditor re-audits (loop 1 of 2). Recommended order: F-D6-01 and F-D6-02 (audit trail), F-D6-05 (provenance binding), F-D6-06 and F-D6-07 (test-side gates), then the orchestrator decision on F-D6-04.
