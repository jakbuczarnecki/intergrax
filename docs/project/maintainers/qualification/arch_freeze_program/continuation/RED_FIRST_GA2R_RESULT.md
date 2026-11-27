# Red-first re-measurement (GA.2R)

Ticket: T-IGX-078-B. Machine-readable record: `RED_FIRST_GA2R_RESULT.json` (same content).
Earlier RED_FIRST records are not edited. This record is not a waiver and adds no operator disposition.

- Read pin: `aabd3925c48bce40bfa5357f39e81bdc7858f8cc`
- Method: MEP-AF2 Appendix V5.3: detached worktree at the prefix SHA, only the test file overlaid from its fix commit, one pytest process at a time under the shared lock. RED_AT_PREFIX only if the node collects, fails on a behavioural assertion about the declared FRZ mechanism and passes at the read pin.
- Environment: Windows 11, CPython 3.12.11; uv sync --frozen with extras dev-ci and dev-unit-cert (pin), dev-ci and mcp (prefix worktrees)
- Pin control run over 31 test files: exit 1, 1 failed, 248 passed, 1 warning in 144.05s (0:02:24); log sha256 `2686c4dea8f442ce8075b1149a09a48a16877ad4ce5ba9572d2fbbabab170590`.
- Pin failures: `tests/unit/architecture/test_rw_bnd05_provider_switch_gate.py::test_nt_bnd_05_2_provider_imports_stay_on_the_closed_list`
- Logs are session artifacts (.tmp/session/ga2r/red/) and are not tracked; the assertion lines below are copied from them, with the replay worktree path shown as `<replay-worktree>` and the account home as `<user-home>`.
- Totals: 21 rows measured; 8 RED_AT_PREFIX, 12 EVIDENCE_MISSING, 1 operator disposition unchanged.

## Summary

| FRZ row | Status | Prefix | Red-first due at |
|---|---|---|---|
| FRZ-BND-04 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-BND-05 | RED_AT_PREFIX | 13b3789c60910ccd3882bfd39641fb5f8d3530fc | T-IGX-043-H |
| FRZ-BND-06 | EVIDENCE_MISSING | - | T-IGX-043-J |
| FRZ-CFG-06 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-CTR-02 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-CTR-06 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-DEBT-02 | EVIDENCE_MISSING | - | T-IGX-051-H |
| FRZ-DEBT-03 | EVIDENCE_MISSING | - | T-IGX-051-H |
| FRZ-DEBT-04 | EVIDENCE_MISSING | - | T-IGX-051-F |
| FRZ-EXE-02 | EVIDENCE_MISSING | - | T-IGX-043-A |
| FRZ-OWN-05 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-PLG-05 | EVIDENCE_MISSING | - | T-IGX-043-A |
| FRZ-PRD-05 | RED_AT_PREFIX | 754ee000675d | - |
| FRZ-SEC-04 | RED_AT_PREFIX | ccc7287c4067 | - |
| FRZ-SEC-06 | EVIDENCE_MISSING | - | T-IGX-051-B |
| FRZ-TEN-02 | EVIDENCE_MISSING | - | T-IGX-078-D |
| FRZ-TEN-07 | RED_AT_PREFIX | 0cee41a88274 | - |
| FRZ-TYP-01 | EVIDENCE_MISSING | - | T-IGX-078-C |
| FRZ-TYP-03 | EVIDENCE_MISSING | - | T-IGX-078-C |
| FRZ-TYP-05 | EVIDENCE_MISSING | - | T-IGX-051-G |
| FRZ-RPL-04 | RED_FIRST_OUT (unchanged) | - | - |

## Rows

<a id="frz-bnd-04"></a>
### FRZ-BND-04 - RED_AT_PREFIX

Mechanism: private implementation leakage = 0: a production module must not import intergrax.integrations._shared.

Behavioural failure on the offending production imports at the prefix; passes at the pin. Coverage not shown: the node is not seeded and carries documented exceptions.

- `tests/unit/architecture/test_wa_integrations_shared_leakage.py::test_nt_rw_wa_01_production_modules_do_not_import_integrations_shared` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `987b203cdfaf56e22c9e143ce7cdabc1cb76bbe0` from `a40e7291a66f`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert ['agents/web_...actory.py:10'] == []
  E         Left contains 4 more items, first extra item: 'agents/web_search_qualifier/search_provider_resolver.py:10'
  E         Full diff:
  E         - []
  ```


<a id="frz-bnd-05"></a>
### FRZ-BND-05 - RED_AT_PREFIX

Mechanism: cross-layer construction only through sanctioned composition (NT-BND-05-1, NT-BND-05-2).

Mechanism-aligned prefix FAILED on four unauthorized Tier-2 `RuntimeContext.build` sites; pin PASSED at identical `test_blob_sha`. Scope is direct Tier-2 construction removal only; STRICT host authority residual remains open.

- `tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_scan_flags_a_rogue_agent` - counted; prefix `13b3789c60910ccd3882bfd39641fb5f8d3530fc` FAILED (run exit 1), pin `7b7ec4488c504f1c90dba599d5742197dbbf0c1e` PASSED (run exit 0); test blob `06412ff4c544adc2b514c85184af434695b4daaa` unchanged; prefix log sha256 `62c5330515fbcac2e0b20400597267dbe5090fbb0ac1804e98992b5877f22f40`; pin log sha256 `caf8db4b1099d71b7ce500857d5ca1117a88990ac7fe3fbf7c673163b523e04b`

  ```text
  E       AssertionError: agents/boundary_demo/boundary_demo_agent.py:96: unauthorized RuntimeContext.build call
  E         agents/external_contractor_adapter/external_contractor_adapter_agent.py:112: unauthorized RuntimeContext.build call
  E         agents/organization_worker/organization_worker_agent.py:118: unauthorized RuntimeContext.build call
  E         intergrax/agents/authoring/acp_stub_reflex.py:69: unauthorized RuntimeContext.build call
  E       assert violations == []
  ```

Historical non-counted nodes retained below.

<a id="frz-bnd-06"></a>
### FRZ-BND-06 - EVIDENCE_MISSING

Mechanism: transport adapters do not own runtime semantics (NT-BND-06-1).

The W5 nodes fail on OTLP export payload typing. None shows a gate turning red because a transport adapter constructs or interprets a RuntimeEvent or envelope (NT-BND-06-1).

- `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py::test_g1_otlp_transport_port_export_parameter` - not counted; prefix `307a871c059f` FAILED (run exit 1), pin PASSED (run exit 1); test blob `50d9dab5bca0fe954c73a6b7328af968392658e2` from `5a87046ba6b5`; log sha256 `b8c158e653827ca31f7ae2a552d1848ad7571aee3ec7f34ba990bd6562799772`

  ```text
  E       AssertionError: assert 'object' == 'ObservabilityExportPayload'
  E         - ObservabilityExportPayload
  E         + object
  ```

- `tests/unit/runtime/architecture/test_harness_w5_typed_observability_transport_gate.py::test_g4_transport_export_bodies_no_dynamic_probing` - not counted; prefix `307a871c059f` FAILED (run exit 1), pin PASSED (run exit 1); test blob `50d9dab5bca0fe954c73a6b7328af968392658e2` from `5a87046ba6b5`; log sha256 `b8c158e653827ca31f7ae2a552d1848ad7571aee3ec7f34ba990bd6562799772`

  ```text
  E       AssertionError: assert ['OtlpTranspo...s isinstance'] == []
  E         Left contains one more item: 'OtlpTransport.export uses isinstance'
  E         Full diff:
  E         - []
  ```


<a id="frz-cfg-06"></a>
### FRZ-CFG-06 - RED_AT_PREFIX

Mechanism: an unconfigured vendor cannot be reached around the factory: no vendor SDK import outside the provider boundary (NT-CFG-06-1, as FRZ-CTR-02).

Behavioural failure on two modules importing the ollama SDK outside the provider package; passes at the pin. Coverage not shown: the scan does not include applications/ (one `import ollama` remains at applications/local_workspace_application/benchmarks/local_model_qualification/environment.py:108 at the pin), the node has no seeded case, and the vendor list in scripts/maintenance/check_integration_vendor_imports.py does not name ollama. The FC-3 selector files do not collect at the prefix (ImportError) and are not counted, so the selector fail-closed findings A4-017 and A4-019 have no red-first evidence.

- `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py::test_no_module_outside_the_provider_package_imports_the_ollama_sdk` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `3eb00a5a0d3eba413d54399b02142e44384ebf62` from `a27671e784a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert ['intergrax/m...ion_types.py'] == []
  E         Left contains 2 more items, first extra item: 'intergrax/multimedia/images_loader.py'
  E         Full diff:
  E         - []
  ```

- `tests/unit/model_inference/test_fc3_vision_fail_closed.py` - not counted; prefix `0cee41a88274` COLLECTION_ERROR (run exit 1), pin 11 nodes: 11 PASSED (run exit 1); test blob `b341e36164c43219b1e2102d72deeba499a471dd` from `500f136d86fe`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E   ImportError: cannot import name 'VisionAdapterCapabilityError' from 'intergrax.model_inference.contracts' (<replay-worktree>\intergrax\model_inference\contracts.py)
  ```

- `tests/unit/rag/embedding/test_fc3_embedding_selector_fail_closed.py` - not counted; prefix `0cee41a88274` COLLECTION_ERROR (run exit 1), pin 6 nodes: 6 PASSED (run exit 1); test blob `b885473947a9c7a6bfb53416201f45d60df44d04` from `66664e2bffa1`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E   ImportError: cannot import name 'LabSelectorOptIn' from 'intergrax.integrations.contracts.base' (<replay-worktree>\intergrax\integrations\contracts\base.py)
  ```

- `tests/unit/rag/vectorstore/test_fc3_vectorstore_selector_fail_closed.py` - not counted; prefix `0cee41a88274` COLLECTION_ERROR (run exit 1), pin 4 nodes: 4 PASSED (run exit 1); test blob `2a281e831edbc45273dded4b0cff583196cfbd0c` from `71e69cad061e`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E   ImportError: cannot import name 'LabSelectorOptIn' from 'intergrax.integrations.contracts.base' (<replay-worktree>\intergrax\integrations\contracts\base.py)
  ```

- `tests/unit/runtime/notifications/test_fc3_notification_selector_fail_closed.py` - not counted; prefix `0cee41a88274` COLLECTION_ERROR (run exit 1), pin 16 nodes: 16 PASSED (run exit 1); test blob `023592604849b3daed99b9cfc8a77e6d0621e9cf` from `c038b95f133f`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E   ImportError: cannot import name 'resolve_host_notification_adapter' from 'intergrax.applications._shared.notification_wiring' (<replay-worktree>\intergrax\applications\_shared\notification_wiring.py)
  ```


<a id="frz-ctr-02"></a>
### FRZ-CTR-02 - RED_AT_PREFIX

Mechanism: concrete provider leakage = 0: `import ollama` outside the provider boundary fails (NT-CTR-02-1).

Behavioural failure on two modules importing the ollama SDK outside the provider package; passes at the pin. Coverage not shown: the scan does not include applications/ (one `import ollama` remains at applications/local_workspace_application/benchmarks/local_model_qualification/environment.py:108 at the pin), the node has no seeded case, and the vendor list in scripts/maintenance/check_integration_vendor_imports.py does not name ollama.

- `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py::test_no_module_outside_the_provider_package_imports_the_ollama_sdk` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `3eb00a5a0d3eba413d54399b02142e44384ebf62` from `a27671e784a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert ['intergrax/m...ion_types.py'] == []
  E         Left contains 2 more items, first extra item: 'intergrax/multimedia/images_loader.py'
  E         Full diff:
  E         - []
  ```


<a id="frz-ctr-06"></a>
### FRZ-CTR-06 - RED_AT_PREFIX

Mechanism: as FRZ-CTR-02: vendor SDK import outside the provider boundary fails (NT-CTR-06-1).

Same node as FRZ-CTR-02 (the matrix defines NT-CTR-06-1 as that gate). Coverage not shown: the scan does not include applications/ (one `import ollama` remains at applications/local_workspace_application/benchmarks/local_model_qualification/environment.py:108 at the pin), the node has no seeded case, and the vendor list in scripts/maintenance/check_integration_vendor_imports.py does not name ollama.

- `tests/unit/multimedia/test_rw_runtime_no_vendor_sdk_outside_providers.py::test_no_module_outside_the_provider_package_imports_the_ollama_sdk` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `3eb00a5a0d3eba413d54399b02142e44384ebf62` from `a27671e784a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert ['intergrax/m...ion_types.py'] == []
  E         Left contains 2 more items, first extra item: 'intergrax/multimedia/images_loader.py'
  E         Full diff:
  E         - []
  ```


<a id="frz-debt-02"></a>
### FRZ-DEBT-02 - EVIDENCE_MISSING

Mechanism: unresolved transitional authority = 0: dependency-ownership gate (NT-DEBT-02-1).

01a[jsonschema], 01a[tzdata] and 01b fail at the prefix with CORE_OWNERSHIP_VIOLATION and pass at the pin, but the dependency-ownership gate was already red at that baseline (EQOA-A4-005) and the prefix commit only adds the test; MEP red-first rule: "a gate already red" does not count. The seeded negative case NT-RW-DEP-01c passes at the prefix, so no seeded red exists.

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01a_core_requirement_has_declared_owner[jsonschema]` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert 'CORE_OWNERSHIP_VIOLATION: jsonschema' not in ['CORE_OWNERSHIP_VIOLATION: jsonschema', 'CORE_OWNERSHIP_VIOLATION: tzdata']
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01a_core_requirement_has_declared_owner[tzdata]` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert 'CORE_OWNERSHIP_VIOLATION: tzdata' not in ['CORE_OWNERSHIP_VIOLATION: jsonschema', 'CORE_OWNERSHIP_VIOLATION: tzdata']
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01b_project_reports_no_core_ownership_violation` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert ['CORE_OWNERS...TION: tzdata'] == []
  E         Left contains 2 more items, first extra item: 'CORE_OWNERSHIP_VIOLATION: jsonschema'
  E         Full diff:
  E         - []
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement` - not counted; prefix `6c9c8155da62` PASSED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

<a id="frz-debt-03"></a>
### FRZ-DEBT-03 - EVIDENCE_MISSING

Mechanism: temporary compatibility seam in frozen scope = 0 (NT-DEBT-03-1, as FRZ-DEBT-02).

01a[jsonschema], 01a[tzdata] and 01b fail at the prefix with CORE_OWNERSHIP_VIOLATION and pass at the pin, but the dependency-ownership gate was already red at that baseline (EQOA-A4-005) and the prefix commit only adds the test; MEP red-first rule: "a gate already red" does not count. The seeded negative case NT-RW-DEP-01c passes at the prefix, so no seeded red exists. No node inventories compatibility seams.

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01a_core_requirement_has_declared_owner[jsonschema]` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert 'CORE_OWNERSHIP_VIOLATION: jsonschema' not in ['CORE_OWNERSHIP_VIOLATION: jsonschema', 'CORE_OWNERSHIP_VIOLATION: tzdata']
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01a_core_requirement_has_declared_owner[tzdata]` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert 'CORE_OWNERSHIP_VIOLATION: tzdata' not in ['CORE_OWNERSHIP_VIOLATION: jsonschema', 'CORE_OWNERSHIP_VIOLATION: tzdata']
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01b_project_reports_no_core_ownership_violation` - not counted; prefix `6c9c8155da62` FAILED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

  ```text
  E       AssertionError: assert ['CORE_OWNERS...TION: tzdata'] == []
  E         Left contains 2 more items, first extra item: 'CORE_OWNERSHIP_VIOLATION: jsonschema'
  E         Full diff:
  E         - []
  ```

- `tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement` - not counted; prefix `6c9c8155da62` PASSED (run exit 1), pin PASSED (run exit 1); test blob `79d975b93716f4ba3f669cb6be16ed9963923290` from `6c9c8155da62`; log sha256 `ecd370df75e2bbd512501997fea8816f37341c46e8e0c7274482ee6314b9aa5b`

<a id="frz-debt-04"></a>
### FRZ-DEBT-04 - EVIDENCE_MISSING

Mechanism: PARTIAL inside frozen scope = 0: a PARTIAL entry without a disposition fails (NT-DEBT-04-1).

The cited node does not collect at the prefix (ImportError) and it tests the evaluation digest guard, not an inventory of PARTIAL entries.

- `tests/unit/architecture/test_rw_gates_eval_digest_denied.py::test_nt_debt_04_enforce_eval_run_digest_guard` - not counted; prefix `8bdb0038d7a8` COLLECTION_ERROR (run exit 1), pin PASSED (run exit 1); test blob `f38ed9330481c1ca1a3063d912576ade549f1122` from `866025070547`; log sha256 `847eef1e2e05dd486c21096817a05f8d1bc8e4f797e144557a03c49d6cdd5c55`

  ```text
  E   ImportError: cannot import name 'enforce_eval_run_digests' from 'intergrax.contracts.evaluation_contracts' (<replay-worktree>\intergrax\contracts\evaluation_contracts.py)
  ```


<a id="frz-exe-02"></a>
### FRZ-EXE-02 - EVIDENCE_MISSING

Mechanism: all meaningful executable work enters the canonical boundary: construction outside the registry fails (NT-EXE-02-1).

No node fails because an agent is constructed outside the registry. The agent import-rule nodes test an import boundary; the routing-posture nodes fail on a missing keyword or attribute; the construction gate fails on missing production_mode posture at router and loop sites.

- `tests/unit/agents/test_wa_agent_import_rule.py::test_wa_13_agents_tree_uses_only_integration_contract_leaves` - not counted; prefix `b7855b83e98e` FAILED (run exit 1), pin PASSED (run exit 1); test blob `6bb73f92cb6a59e84d11c0dbbc2ee67e7f871212` from `fab086f37f36`; log sha256 `624c04a6ec2a80037556eff46635307fe38edc89ac5ce894f7a3320852e263ea`

  ```text
  E       AssertionError: assert ['agents/web_...p3.factories'] == []
  E         Left contains 2 more items, first extra item: 'agents/web_search_qualifier/search_provider_resolver.py:10 -> intergrax.integrations._shared.p2.factories'
  E         Full diff:
  E         - []
  ```

- `tests/unit/runtime/execution/test_fc1_host_task_routing_posture.py::test_host_task_execution_uses_the_loops_router` - not counted; prefix `b7855b83e98e` FAILED (run exit 1), pin PASSED (run exit 1); test blob `6aeb19344ac4e3607858ac5e46796846ff4d2b8f` from `d81f21a0c78a`; log sha256 `624c04a6ec2a80037556eff46635307fe38edc89ac5ce894f7a3320852e263ea`

  ```text
  E           AttributeError: 'NexusLoop' object has no attribute 'agent_router'
  ```

- `tests/unit/architecture/test_fc1_router_and_loop_construction_gate.py::test_no_product_construction_without_explicit_posture` - not counted; prefix `b7855b83e98e` FAILED (run exit 1), pin PASSED (run exit 1); test blob `67a97368c3823eb6beb6069693abea6551c8680d` from `d81f21a0c78a`; log sha256 `624c04a6ec2a80037556eff46635307fe38edc89ac5ce894f7a3320852e263ea`

  ```text
  E       AssertionError: intergrax/runtime/execution/debug_lab_nexus_loop.py:26: NexusLoop without production_mode
  E         intergrax/runtime/execution/debug_lab_nexus_loop.py:40: NexusLoop without production_mode
  E         intergrax/runtime/execution/nexus_host_execution.py:57: AgentRouter without production_mode
  E         intergrax/runtime/execution/nexus_host_execution.py:57: AgentRouter outside the sanctioned sites
  ```


<a id="frz-own-05"></a>
### FRZ-OWN-05 - RED_AT_PREFIX

Mechanism: no duplicated semantic mechanism: one host LLM resolver (NT-OWN-05-1, finding A4-016).

Behavioural failure on two host LLM resolver modules at the prefix; passes at the pin. Coverage not shown: the node covers one of the five listed responsibilities (plugin discovery, context-window tables, tool-input guard and evaluator contract have no red-first evidence here).

- `tests/unit/architecture/test_rw_own05_duplicate_mechanisms.py::test_nt_own_05_1_one_host_llm_resolver` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `182c5f1e18ad658d40efed729392ec4eed14a503` from `83896a79cc71`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert ['intergrax/a..._resolver.py'] == ['intergrax/r..._resolver.py']
  E         At index 0 diff: 'intergrax/applications/_shared/llm_resolver.py' != 'intergrax/runtime/wiring/llm_resolver.py'
  E         Left contains one more item: 'intergrax/runtime/wiring/llm_resolver.py'
  E         Full diff:
  ```


<a id="frz-plg-05"></a>
### FRZ-PLG-05 - EVIDENCE_MISSING

Mechanism: activation/admission explicit: an unsigned or digest-mismatched plugin entry point is refused at admission (NT-PLG-05-1).

The nodes that fail at the prefix fail on a missing digest function, attribute or parameter. The unsigned and tampered-digest rejection nodes already pass at the prefix.

- `tests/unit/core/plugins/test_rw_supply_installed_file_digest.py::test_sup_06a_altered_installed_file_refused` - not counted; prefix `459dfebc712f` FAILED (run exit 1), pin PASSED (run exit 1); test blob `45f49aef1f6c4b52a5f78674ffd36930d29e823e` from `2cd836a0c4a7`; log sha256 `5a8b4986ad835da87b665fcc5a1087b03987d147182ebab6940a0b8bbd62e23c`

  ```text
  E       AssertionError: no installed-tree digest function exists (EX-D6-03)
  E       assert None is not None
  ```

- `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py::test_d6_r016_bootstrap_rejects_unsigned_plugin_before_it_is_imported` - not counted; prefix `459dfebc712f` FAILED (run exit 1), pin PASSED (run exit 1); test blob `5a8148d245c9996a2b07caded6e90bd29a59376c` from `adaaa374e465`; log sha256 `5a8b4986ad835da87b665fcc5a1087b03987d147182ebab6940a0b8bbd62e23c`

  ```text
  E       AttributeError: <module 'intergrax.core.catalog_bootstrap' from '<replay-worktree>\\intergrax\\core\\catalog_bootstrap.py'> has no attribute 'iter_entry_point_specs'
  ```

- `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py::test_d6_r016_unsigned_plugin_is_rejected_in_strict_profile` - not counted; prefix `459dfebc712f` PASSED (run exit 1), pin PASSED (run exit 1); test blob `5a8148d245c9996a2b07caded6e90bd29a59376c` from `adaaa374e465`; log sha256 `5a8b4986ad835da87b665fcc5a1087b03987d147182ebab6940a0b8bbd62e23c`
- `tests/qualification/pdc_x/d6/test_d6_plugin_provenance.py::test_d6_r016_tampered_digest_is_rejected` - not counted; prefix `459dfebc712f` PASSED (run exit 1), pin PASSED (run exit 1); test blob `5a8148d245c9996a2b07caded6e90bd29a59376c` from `adaaa374e465`; log sha256 `5a8b4986ad835da87b665fcc5a1087b03987d147182ebab6940a0b8bbd62e23c`

<a id="frz-prd-05"></a>
### FRZ-PRD-05 - RED_AT_PREFIX

Mechanism: shutdown/resource cleanup qualified: a Tier-3 production entrypoint serves through the hosting lifecycle engine (NT-PRD-05).

Both nodes fail on their assertion at the prefix and pass at the pin. Coverage not shown: NT-PRD-05 is a source-text check of one serving command and NT-WB-15 covers one reference application; the declared entrypoint census gate does not exist, and direct uvicorn.run calls remain in other applications at the pin.

- `tests/unit/architecture/test_nt_prd_red_first_probes.py::test_nt_prd_05_entrypoint_uses_hosting_lifecycle_engine` - counted; prefix `754ee000675d` FAILED (run exit 1), pin PASSED (run exit 1); test blob `ab4ba59fb2acf76f8dd14bd75109d97d8e3abab3` from `dff5449c97bf`; log sha256 `0e402073e427fddaaaff7c24a9807ffb69b7edbded715e1914d708bf833c9faf`

  ```text
  E       AssertionError: NT-PRD-05 missing hosting lifecycle entry: run_governed_contractor_hosted_strict_production_application
  E       assert 'run_governed_contractor_hosted_strict_production_application' in '# � Artur Czarnecki. All rights reserved.\n\n"""Sanctioned Package B strict production serve command (profile v1 + D4 uvicorn factory)."""\n\nfrom __future__ import annotations\n\nimport os\nimport sys\n\nfrom governed_contractor_application.host.settings import GovernedContractorBackendSettings\nfrom governed_contractor_application.host.single_node_profile_v1_strict_startup import (\n    open_gca_strict_production_serving_session_from_single_node_production_profile_v1,\n)\nfrom governed_contractor_application.host.strict_production_serving import (\n    build_governed_contractor_strict_production_uvicorn_factory,\n)\nfrom intergrax.agent_distribution.venv_bundle_colocated_runtime_deployment_adapter import (\n    VenvBundleColocatedRuntimeDeploymentAdapter,\n)\nfrom intergrax.applications._shared.package_b_operator_production_governance_configuration import (\n    PackageBOperatorProductionGovernanceConfigurationError,\n    load_package_b_operator_production_control_plane_governance_from_env,\n)\nfrom intergrax.applications._shared.package_b_strict_production_serving_environment import (\n    PackageBStrictProductionServingEnvironmentError,\n    load_strict_production_serving_en...xc:\n        close_exc = _close_strict_session_when_command_still_owns_shutdown(session)\n        _reraise_with_command_owned_close_failure(exc, close_exc)\n    except BaseException as exc:\n        close_exc = _close_strict_session_when_command_still_owns_shutdown(session)\n        wrapped = PackageBStrictProductionServingCommandError(str(exc))\n        if close_exc is None:\n            raise wrapped from exc\n        raise_primary_with_cleanup_errors(\n            wrapped,\n            [close_exc],\n            message=(\n                "package B strict production serving command failed; "\n                "errors during command-owned session close"\n            ),\n        )\n    else:\n        close_exc = _close_strict_session_when_command_still_owns_shutdown(session)\n        if close_exc is not None:\n            raise PackageBStrictProductionServingCommandError(str(close_exc)) from close_exc\n\n\ndef main() -> None:\n    try:\n        run_package_b_strict_production_serving_command()\n    except PackageBStrictProductionServingCommandError as exc:\n        print(str(exc), file=sys.stderr)\n        raise SystemExit(2) from exc\n\n\nif __name__ == "__main__":\n    main()\n'
  ```

- `tests/unit/applications/test_wb_serving_bind_defaults.py::test_nt_wb_15_lkw_reference_production_has_no_uvicorn_run` - counted; prefix `754ee000675d` FAILED (run exit 1), pin PASSED (run exit 1); test blob `805f5cf4bd681ea833f8c447fc2f3f41797d3922` from `923c80addf8e`; log sha256 `0e402073e427fddaaaff7c24a9807ffb69b7edbded715e1914d708bf833c9faf`

  ```text
  E       assert not True
  E        +  where True = any(<generator object test_nt_wb_15_lkw_reference_production_has_no_uvicorn_run.<locals>.<genexpr> at 0x000001D7AB775A40>)
  ```


<a id="frz-sec-04"></a>
### FRZ-SEC-04 - RED_AT_PREFIX

Mechanism: production auth bypass = 0: unset or invalid inbound verification mode, or a mode without its secret, refuses to build (NT-SEC-04-1, A4-026).

The counted nodes fail with DID NOT RAISE SecurityConfigurationError at the prefix and pass at the pin. Coverage not shown: INT7 and the STRICT/PRODUCT host wiring nodes INT9 to INT12 fail at the prefix with TypeError on a keyword argument that did not exist and are not counted.

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int1_create_inbound_verifier_env_unset` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int2_invalid_mode_raises_with_and_without_opt_in` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int3_slack_mode_no_secret` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int4_slack_secret_verify_disabled` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int5_teams_mode_no_token` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int6_explicit_none_without_opt_in` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int8_intake_service_verifier_none_raises` - counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E       Failed: DID NOT RAISE <class 'intergrax.core.security.security_configuration.SecurityConfigurationError'>
  ```

- `tests/unit/runtime/interactions/test_fc3_inbound_verifier_fail_closed.py::test_nt_fc3_int9_wire_strict_without_verifier_config` - not counted; prefix `ccc7287c4067` FAILED (run exit 1), pin PASSED (run exit 1); test blob `b3e8e63adfe12c798b9ff02b30eb9b36f3cf6e35` from `00bcb13f3a22`; log sha256 `78625ba0f1aad2aa7b5fe9024f52a0ae0aee741e4253fc8809be635c9f873a23`

  ```text
  E           TypeError: wire_interaction_intake_service() got an unexpected keyword argument 'env'
  ```


<a id="frz-sec-06"></a>
### FRZ-SEC-06 - EVIDENCE_MISSING

Mechanism: provider/plugin trust qualified: a strict root refuses unattested defence plugins (NT-SEC-06-1).

The node fails at the prefix because the require_provenance setting did not exist, not because an unattested defence plugin was admitted.

- `tests/unit/runtime/security/test_rw_supply_defense_plugin_provenance_required.py::test_sec_06_1_strict_root_refuses_unattested_defense_plugins` - not counted; prefix `459dfebc712f` FAILED (run exit 1), pin PASSED (run exit 1); test blob `811a35adf21b928ac7bfe69d89e523593a915c00` from `2cd836a0c4a7`; log sha256 `5a8b4986ad835da87b665fcc5a1087b03987d147182ebab6940a0b8bbd62e23c`

  ```text
  E       AssertionError: a strict root cannot require provenance for security-defense plugins (FRZ-SEC-06)
  E       assert 'require_provenance' in {'ep_name_conflict', 'on_load_failure', 'plugin_id_conflict', 'provenance', 'shipped_id_override'}
  ```


<a id="frz-ten-02"></a>
### FRZ-TEN-02 - EVIDENCE_MISSING

Mechanism: end-to-end tenant propagation across semantic and runtime boundaries.

The failing node asserts the production_mode posture of a strict host, not tenant propagation. No end-to-end tenant propagation node exists.

- `tests/unit/campaign_b/test_pt_t01_nt02_posture_contract.py::test_nt02_strict_host_injection_sets_production_mode` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin NOT_PRESENT_UNDER_THIS_NAME (run exit 1); test blob `ca647a4091646db0e970dc6c7687c82ac3313b3e` from `6fdc2b5c22b8`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: assert False is True
  E        +  where False = RuntimeConfig(llm_adapter=<external_contractor_adapter.external_contractor_adapter_agent._ExternalContractorAdapterStubLLM object at 0x00000239B8376ED0>, embedding_manager=None, vectorstore_manager=None, retriever_manager=None, reranker_manager=None, rag_profile=None, retrieval_service=None, enable_rag=False, enable_websearch=True, tenant_id='t1', workspace_id=None, llm_routing_context=None, llm_routing_snapshot=None, max_docs_per_query=8, max_rag_tokens=4096, rag_score_threshold=None, max_longterm_entries_per_query=8, max_longterm_tokens=4096, longterm_score_threshold=None, websearch_executor=None, websearch_config=None, tool_planner=None, tools_mode='auto', tool_planner_prompt_id='tools_agent_planner', engine_planner_prompt_id='planner_default', tool_selection_mode=<ToolSelectionMode.STATIC: 'static'>, tool_selection_top_k=20, tool_selection_max_hierarchy_passes=2, tool_selection_hierarchical_llm_pass=False, tool_selection_strategy=None, tool_selection_strategy_id=None, execution_authority_policy=None, execution_authority_policy_id=None, execution_budget_allocation_policy=None, execution_budget_allocation_policy_id=None, tools_context_scope=<ToolsContextScope.CURRENT_MESSAGE_ON...text_budget_policy=None, task_context_assembly_options=None, context_decision_profile=None, metadata={}, execution_boundary_export=None, boundary_event_buffer=None, enable_llm_usage_collection=True, llm_usage_tracker=None, runtime_timeout_ms=None, max_run_retries=0, retry_run_on=frozenset({<RuntimeErrorCode.LLM_ERROR: 'llm_error'>, <RuntimeErrorCode.TOOL_ERROR: 'tool_error'>}), runtime_policies=RuntimePolicies(timeout=TimeoutPolicy(llm_seconds=30.0, tool_seconds=30.0, retrieval_seconds=10.0, storage_seconds=5.0), retry=RetryPolicy(max_attempts=3, backoff_seconds=0.0), fallback=FallbackPolicy(escalate_to_hitl=True), hitl=HitlPolicy(enabled=True, stop_reason='needs_user_input'), data_compliance=DataCompliancePolicy(api_trace_export='redacted', redact_tool_calls_in_api=True)), execution_slot_warn_threshold_ms=None, hitl_default_message=None, run_budget=None, budget_policy=None, trace_db_path=None, integration_profile=None, runtime_event_bus=None, context_engine=None, policy_bundle=None, production_mode=False, security_profile=None, evaluation_profile=None, evaluation_registry=None, signal_collector=None, adaptive_profile=None, resolved_profile_versions=None, prompt_catalog_path=None).production_mode
  E        +    where RuntimeConfig(llm_adapter=<external_contractor_adapter.external_contractor_adapter_agent._ExternalContractorAdapterStubLLM object at 0x00000239B8376ED0>, embedding_manager=None, vectorstore_manager=None, retriever_manager=None, reranker_manager=None, rag_profile=None, retrieval_service=None, enable_rag=False, enable_websearch=True, tenant_id='t1', workspace_id=None, llm_routing_context=None, llm_routing_snapshot=None, max_docs_per_query=8, max_rag_tokens=4096, rag_score_threshold=None, max_longterm_entries_per_query=8, max_longterm_tokens=4096, longterm_score_threshold=None, websearch_executor=None, websearch_config=None, tool_planner=None, tools_mode='auto', tool_planner_prompt_id='tools_agent_planner', engine_planner_prompt_id='planner_default', tool_selection_mode=<ToolSelectionMode.STATIC: 'static'>, tool_selection_top_k=20, tool_selection_max_hierarchy_passes=2, tool_selection_hierarchical_llm_pass=False, tool_selection_strategy=None, tool_selection_strategy_id=None, execution_authority_policy=None, execution_authority_policy_id=None, execution_budget_allocation_policy=None, execution_budget_allocation_policy_id=None, tools_context_scope=<ToolsContextScope.CURRENT_MESSAGE_ON...text_budget_policy=None, task_context_assembly_options=None, context_decision_profile=None, metadata={}, execution_boundary_export=None, boundary_event_buffer=None, enable_llm_usage_collection=True, llm_usage_tracker=None, runtime_timeout_ms=None, max_run_retries=0, retry_run_on=frozenset({<RuntimeErrorCode.LLM_ERROR: 'llm_error'>, <RuntimeErrorCode.TOOL_ERROR: 'tool_error'>}), runtime_policies=RuntimePolicies(timeout=TimeoutPolicy(llm_seconds=30.0, tool_seconds=30.0, retrieval_seconds=10.0, storage_seconds=5.0), retry=RetryPolicy(max_attempts=3, backoff_seconds=0.0), fallback=FallbackPolicy(escalate_to_hitl=True), hitl=HitlPolicy(enabled=True, stop_reason='needs_user_input'), data_compliance=DataCompliancePolicy(api_trace_export='redacted', redact_tool_calls_in_api=True)), execution_slot_warn_threshold_ms=None, hitl_default_message=None, run_budget=None, budget_policy=None, trace_db_path=None, integration_profile=None, runtime_event_bus=None, context_engine=None, policy_bundle=None, production_mode=False, security_profile=None, evaluation_profile=None, evaluation_registry=None, signal_collector=None, adaptive_profile=None, resolved_profile_versions=None, prompt_catalog_path=None) = RuntimeContext(config=RuntimeConfig(llm_adapter=<external_contractor_adapter.external_contractor_adapter_agent._ExternalContractorAdapterStubLLM object at 0x00000239B8376ED0>, embedding_manager=None, vectorstore_manager=None, retriever_manager=None, reranker_manager=None, rag_profile=None, retrieval_service=None, enable_rag=False, enable_websearch=True, tenant_id='t1', workspace_id=None, llm_routing_context=None, llm_routing_snapshot=None, max_docs_per_query=8, max_rag_tokens=4096, rag_score_threshold=None, max_longterm_entries_per_query=8, max_longterm_tokens=4096, longterm_score_threshold=None, websearch_executor=None, websearch_config=None, tool_planner=None, tools_mode='auto', tool_planner_prompt_id='tools_agent_planner', engine_planner_prompt_id='planner_default', tool_selection_mode=<ToolSelectionMode.STATIC: 'static'>, tool_selection_top_k=20, tool_selection_max_hierarchy_passes=2, tool_selection_hierarchical_llm_pass=False, tool_selection_strategy=None, tool_selection_strategy_id=None, execution_authority_policy=None, execution_authority_policy_id=None, execution_budget_allocation_policy=None, execution_budget_allocation_policy_id=None, tools_context_scope=<ToolsContextSc...sion_manager=<intergrax.runtime.nexus.session.session_manager.SessionManager object at 0x00000239B67F9F40>, replay_service=None, ingestion_service=None, context_builder=None, rag_prompt_builder=<intergrax.runtime.nexus.prompts.rag_prompt_builder.DefaultRagPromptBuilder object at 0x00000239B32B9370>, user_longterm_memory_prompt_builder=<intergrax.runtime.nexus.prompts.user_longterm_memory_prompt_builder.DefaultUserLongTermMemoryPromptBuilder object at 0x00000239B32138F0>, websearch_executor=None, websearch_prompt_builder=<intergrax.runtime.nexus.prompts.websearch_prompt_builder.DefaultWebSearchPromptBuilder object at 0x00000239B83CD220>, history_layer=<intergrax.runtime.nexus.context.engine_history_layer.HistoryLayer object at 0x00000239B83CC3B0>, llm_usage_run_seq=0, llm_usage_runs=[], llm_usage_lock=<asyncio.locks.Lock object at 0x00000239B32917F0 [unlocked]>, trace_writer=None, governance_service=None, governance_evidence_recorder=None, prompt_registry=<intergrax.prompts.registry.yaml_registry.YamlPromptRegistry object at 0x00000239B32915B0>, artifact_store=None, execution_semaphore=None, policy_bundle=None, max_parallel_per_tenant=None, tool_invoker_close_on_context_close=True).config
  ```


<a id="frz-ten-07"></a>
### FRZ-TEN-07 - RED_AT_PREFIX

Mechanism: evidence and diagnostics preserve tenant identity and cannot establish truth for another tenant: MCP client-asserted tenant fails closed.

Behavioural failure at the prefix (a client-asserted tenant was accepted); passes at the pin. Coverage not shown: only the MCP ingress half of the row; tenant preservation in events, traces and diagnostics has no red-first evidence (the event-id persistence file does not collect at its prefix, ImportError).

- `tests/unit/applications/test_nt_ten_red_first_probes.py::test_nt_ten_07_mcp_client_asserted_tenant_identity_must_fail_closed` - counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `0e03f805961a1e7052990c0838409c5c6ec984f4` from `ce0b63fab74f`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E           AssertionError: NT-TEN-07 MCP client-asserted tenant identity must fail closed: client-asserted tenant must not establish truth without verified harness principal
  E           assert 'victim' != 'victim'
  E            +  where 'victim' = McpRunAgentIdentityResolution(tenant_id='victim', user_id='mcp-user', principal=None, request_identity=None).tenant_id
  ```

- `tests/unit/runtime/events/test_event_id_persistence_semantics.py` - not counted; prefix `5fe00c793c5e` COLLECTION_ERROR (run exit 1), pin 62 nodes: 62 PASSED (run exit 1); test blob `0a09a36eb44be475d47b3057eb7942e74a5fd38c` from `6d1670b230fe`; log sha256 `fc074dc01df388e1f08eda4005bfd89f56b65b95c2debbc061b230519dbba33c`

  ```text
  E   ImportError: cannot import name '_global_event_id_partition' from 'intergrax.runtime.events.stores.document_backed_runtime_event_store' (<replay-worktree>\intergrax\runtime\events\stores\document_backed_runtime_event_store.py)
  ```


<a id="frz-typ-01"></a>
### FRZ-TYP-01 - EVIDENCE_MISSING

Mechanism: semantic boundaries strongly typed (NT-WF-B01..B06 are the planned negative tests).

The typed-ratchet nodes fail at the prefix because the baseline file or the checker script does not exist (FileNotFoundError, missing script). That is not a behavioural failure on the declared mechanism. No node seeds Any or object on a semantic boundary.

- `tests/unit/architecture/test_rw_typ_ratchet_new_any_fails.py::test_rw_typ_ratchet_new_any_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `56856d8a41daa1689148aa90278ba678d4f1a9cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_new_object_fails.py::test_rw_typ_ratchet_new_object_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `14bb2d620d6c6376d1e49140456da0fb10d940cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_baseline_unchanged_passes.py::test_rw_typ_ratchet_baseline_unchanged_passes` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `5a675e1974e0fff6e1360fece61990b29c282ada` from `4060193d4fe9`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: <user-home>\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe: can't open file '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory
  E       assert 2 == 0
  E        +  where 2 = CompletedProcess(args=['<replay-worktree>\\.venv\\Scripts\\python.exe', '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py'], returncode=2, stdout='', stderr="<user-home>\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\python.exe: can't open file '<replay-worktree>\\\\scripts\\\\maintenance\\\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory\n").returncode
  ```


<a id="frz-typ-03"></a>
### FRZ-TYP-03 - EVIDENCE_MISSING

Mechanism: typed semantic payloads instead of object (NT-WF-B01..B06 are the planned negative tests).

The typed-ratchet nodes fail at the prefix because the baseline file or the checker script does not exist (FileNotFoundError, missing script). That is not a behavioural failure on the declared mechanism. No node seeds Any or object on a semantic boundary.

- `tests/unit/architecture/test_rw_typ_ratchet_new_any_fails.py::test_rw_typ_ratchet_new_any_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `56856d8a41daa1689148aa90278ba678d4f1a9cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_new_object_fails.py::test_rw_typ_ratchet_new_object_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `14bb2d620d6c6376d1e49140456da0fb10d940cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_baseline_unchanged_passes.py::test_rw_typ_ratchet_baseline_unchanged_passes` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `5a675e1974e0fff6e1360fece61990b29c282ada` from `4060193d4fe9`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: <user-home>\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe: can't open file '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory
  E       assert 2 == 0
  E        +  where 2 = CompletedProcess(args=['<replay-worktree>\\.venv\\Scripts\\python.exe', '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py'], returncode=2, stdout='', stderr="<user-home>\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\python.exe: can't open file '<replay-worktree>\\\\scripts\\\\maintenance\\\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory\n").returncode
  ```


<a id="frz-typ-05"></a>
### FRZ-TYP-05 - EVIDENCE_MISSING

Mechanism: string-dispatch substitute for typed contract = 0: ratchet on new string-keyed dispatch (NT-TYP-05-1).

The typed-ratchet nodes fail at the prefix because the baseline file or the checker script does not exist (FileNotFoundError, missing script). That is not a behavioural failure on the declared mechanism.

- `tests/unit/architecture/test_rw_typ_ratchet_new_any_fails.py::test_rw_typ_ratchet_new_any_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `56856d8a41daa1689148aa90278ba678d4f1a9cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_new_object_fails.py::test_rw_typ_ratchet_new_object_fails` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `14bb2d620d6c6376d1e49140456da0fb10d940cf` from `12ca63a7e3a2`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       FileNotFoundError: [Errno 2] No such file or directory: '<replay-worktree>\\governance\\qualification\\typ_ratchet_baseline.v1.json'
  ```

- `tests/unit/architecture/test_rw_typ_ratchet_baseline_unchanged_passes.py::test_rw_typ_ratchet_baseline_unchanged_passes` - not counted; prefix `0cee41a88274` FAILED (run exit 1), pin PASSED (run exit 1); test blob `5a675e1974e0fff6e1360fece61990b29c282ada` from `4060193d4fe9`; log sha256 `aa48380e8a550661cc3b986b4a3298879bf3513aacf344229f0431a7edf83a65`

  ```text
  E       AssertionError: <user-home>\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe: can't open file '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory
  E       assert 2 == 0
  E        +  where 2 = CompletedProcess(args=['<replay-worktree>\\.venv\\Scripts\\python.exe', '<replay-worktree>\\scripts\\maintenance\\check_frozen_contract_typ_ratchet.py'], returncode=2, stdout='', stderr="<user-home>\\AppData\\Roaming\\uv\\python\\cpython-3.12-windows-x86_64-none\\python.exe: can't open file '<replay-worktree>\\\\scripts\\\\maintenance\\\\check_frozen_contract_typ_ratchet.py': [Errno 2] No such file or directory\n").returncode
  ```


## Operator dispositions (unchanged)

### FRZ-RPL-04 - RED_FIRST_OUT

Operator disposition unchanged (RED-CLASSES). Replayed for information only: the probe node passes at the prefix, so it gives no red.

- `tests/unit/architecture/test_nt_rpl_red_first_probes.py::test_neg_p_rpl_04_queue_worker_call_sites_bind_message_consumer_contract` - not counted; prefix `754ee000675d` PASSED (run exit 1), pin PASSED (run exit 1); test blob `d3d664359e600e4dda70b1a73284428e8b57d24c` from `1bc28033cf0f`; log sha256 `0e402073e427fddaaaff7c24a9807ffb69b7edbded715e1914d708bf833c9faf`
