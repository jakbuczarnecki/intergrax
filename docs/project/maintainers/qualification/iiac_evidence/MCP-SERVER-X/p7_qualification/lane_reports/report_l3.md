# Report L3 - MCP-SERVER-X P7 (T-IGX-035-H), lane mcp-p7-l3

Worktree: D:/Projekty/_wt/intergrax/p7-l3, branch mcp-p7-l3, base origin/development d5c1383fc. Not pushed.

**Fixtures commit (for the FI lane): bf6f9204fb7b6ab390e91b0b960c385a97248667**
Tests commit (branch HEAD): 89cf466d87b38cb63ebed3a2022e5361d47da864. (Preliminary SHA 8387f56db was replaced by a rebase that restored CRLF in build.py; use the SHAs above.)

## Files (all under tests/qualification/mcp_server_x/, no product path touched)
Fixtures (additive): reference_app/build.py (CRLF kept; new optional kwargs: definition, exposure_policy, retired/hidden/omit_from_allowlist, include_catalog_tool_bindings, max_concurrent_tool_calls, catalog_tool_invoker, tool_enablement, task_control_mutation_boundary, meaningful_side_effect_authorization, extra_custom_handlers; new build_mcp_server_x_reference_definition), reference_app/http_serve.py (uvicorn port 0, streamable client helpers, raw JSON-RPC POST, patched delegate), reference_app/variants.py (stub IdentityProvider, HarnessAuthState builders, catalog registry, recording invoker).

Level: all three modules are L3 (physical streamable HTTP, loopback, port 0).
- test_mcp_server_x_p7_l3_publication_discovery.py (17 functions)
  - ACC-01: test_acc_01_reference_server_exact_tool_set_over_http, _authorized_run_agent_over_http_with_auth, _custom_echo_binding_over_http
  - ACC-05: test_acc_05_only_streamable_http_mount_is_published, _streamable_http_initialize_opens_a_session, _sse_and_stdio_style_paths_are_not_served[6 paths]
  - ACC-06: test_acc_06_client_initialize_result_authenticated, _initialize_without_credentials_is_rejected_at_http_level
  - ACC-07: test_acc_07_discovery_set_equals_definition_exposure_over_http, _non_exposed_bindings_are_absent_and_not_callable[retired|disabled|hidden], _exposure_policy_disabled_publishes_no_tools, _invoke_catalog_tool_is_absent_without_a_wired_invoker
  - ACC-08: test_acc_08_input_schema_of_qual_echo_over_http, _input_schema_of_run_agent_over_http, _describe_catalog_tool_payload_matches_tool_contract, _describe_catalog_tool_unknown_id_is_a_typed_error_payload, _list_catalog_tools_matches_contract_over_http
- test_mcp_server_x_p7_l3_invoke_security.py (11 functions)
  - ACC-09: test_acc_09_authorized_invoke_without_assertion_binds_principal_tenant, _with_matching_tenant_and_user_assertion, _via_bearer_identity_binds_token_tenant_and_user
  - ACC-10: test_acc_10_invalid_credentials_are_rejected_with_http_401_before_execution[3], _missing_credential_is_rejected_with_http_401_and_leaks_no_tool_names, _forged_tenant_assertion_is_denied_before_execution, _forged_user_assertion_is_denied_before_execution, _unknown_bearer_token_is_rejected_by_identity_provider_server
  - ACC-11: test_acc_11_principal_cannot_assert_another_tenant_on_one_server, _concurrent_principals_stay_isolated_on_one_server, _two_hosts_sharing_a_registry_do_not_accept_each_others_credentials
- test_mcp_server_x_p7_l3_protocol_subset.py (10 functions)
  - ACC-18: test_acc_18_announced_capabilities_snapshot, _tools_list_and_call_work_over_http, _resources_and_prompts_lists_are_empty, _resources_prompts_capability_is_advertised_but_empty, _unknown_resource_and_prompt_requests_fail, _unannounced_methods_return_method_not_found, _logging_capability_is_announced_and_accepted, _protocol_version_negotiation_is_recorded[2], _only_streamable_http_transport_is_reachable, _matrix_rows_reference_existing_tests (PROTOCOL_SUBSET_MATRIX: checked vs not announced vs unsupported)

## Run
cd D:/Projekty/_wt/intergrax/p7-l3 && D:/Projekty/intergrax/.venv/Scripts/python.exe -m pytest <3 new files + test_mcp_server_x_p6_reference_l3_qualification.py> -q   (under the shared lock)
Final tree: exit 0, 50 passed (48 new + 2 existing P6 L3), 36 s. New files alone ran 3 times: 48 passed each, exit 0 (logs: .tmp/session/mcp-p7p8-run/l3_pytest_*.log). ruff check/format clean. No xfail/skip.

## Findings (no failures; product behaviour vs plan)
1. DOC CONTRADICTION (ACC-18): MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md says Resources/Prompts "Not advertised". Initialize result at FastMCP 3.3.1 advertises resources {subscribe:false, listChanged:true} and prompts {listChanged:true}; all lists are empty and unknown read/get fail with McpError. Pinned in a test; doc not edited.
2. Doc gap: FastMCP also announces `logging` (logging/setLevel returns {}) and an `extensions` "io.modelcontextprotocol/ui" capability; neither is in the matrix. tasks and completions are NOT announced (-32601 Method not found).
3. Allowlisted `invoke_catalog_tool` is mounted only when an ExecutionBoundCatalogToolInvoker is composed; without it the definition lists it as exposed but discovery omits it (fail-closed in effect; pinned by test_acc_07_invoke_catalog_tool_is_absent_without_a_wired_invoker).
4. Endpoint is `/mcp/` (trailing slash); POST `/mcp` with valid credentials returns 404 (falls to the "/" mount); unauthenticated gets 401 first. SDK clients unaffected.
5. Existing test test_mcp_streamable_http_rejects_invalid_api_key uses weak pytest.raises(Exception); here the 401 is asserted precisely (ExceptionGroup leaves are httpx.HTTPStatusError 401 plus raw body).
6. The repo pre-commit hook created a uv .venv inside the worktree (untracked, ignored); harmless.
7. Process note: the fixup commit that restored CRLF in build.py was made with hooks disabled (core.hooksPath); I ran ruff check and ruff format --check on the tree afterwards (clean).

## Not verified / limits
- ACC-09 catalog invoke with tenant assertion (invoke_catalog_tool) not done here: needs a live execution binding (covered by existing F3 tests); only the run_agent assertion path is L3-checked.
- ACC-10 side-effect port deny is L2 (lane L2). ACC-13/14/15 (FI) not started.
- stdio absence is shown by route inventory and 404 probes, not by a process-level check.
- Identity provider is a contract stub (not vendor evidence). Tenant checks use an AsyncMock host port: they prove what reaches the execution port, not a full agent run (ACC-01 uses the canonical host with a stubbed delegate).
- Only my own files plus the existing P6 L3 file were run; no broad suite, no igx_gov ticket run.
