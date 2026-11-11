# MCP-SERVER-X P5 — security requirement-to-test matrix

Plan digest: `5dd3c72e497c72764ccc0c08d3df2dddfdbe19b3f36550ba5d87c4091f9833e7`

| Req ID | Mechanism | Mandatory | Test evidence | Task |
| --- | --- | --- | --- | --- |
| SEC-MCP-01 | Harness API key auth | yes | `test_mcp_http_ingress_still_rejects_missing_credentials`, L3 streamable HTTP auth negatives | MCP-P5-01 |
| SEC-MCP-02 | Principal tenant binding | yes | `test_resolve_mcp_identity_api_key_binds_authoritative_tenant`, streamable HTTP bind | MCP-P5-01 |
| SEC-MCP-03 | Forged tenant/user rejected pre-Execution | yes | `test_resolve_mcp_identity_rejects_forged_*`, `test_mcp_streamable_http_rejects_forged_tenant_before_execution` | MCP-P5-01 |
| SEC-MCP-04 | Cross-tenant catalog deny | yes | `test_gate_b_wrong_tenant`, `test_catalog_child_http_path_denied_without_parent_context` | MCP-P5-02 |
| SEC-MCP-05 | Cross-tenant cancel isolation | yes | `test_cancel_a_cross_tenant_governed_cancel_not_applied` | MCP-P5-02 |
| SEC-MCP-06 | Side-effect port deny | yes | `test_gate_b_side_effect_port_deny_blocks_invoke`, `test_gate_b_unauthorized_side_effect_tool` | MCP-P5-03 |
| SEC-MCP-07 | Governance principal mismatch | yes | `test_catalog_child_denied_on_governance_principal_mismatch` | MCP-P5-03 |
| SEC-MCP-08 | Canonical Execution facade | yes | `test_shared_mcp_run_agent_uses_canonical_execution_facade`, research/gca strategy paths | MCP-P5-04 |
| SEC-MCP-09 | host_custom via engine (no bypass) | yes | `test_host_custom_tool_compose_discover_and_invoke`, kwargs rejection | MCP-P5-04 |
| SEC-MCP-10 | MSE side-effect enforcement | yes | `test_catalog_child_mse_deny_prevents_invoker` | MCP-P5-05 |
| SEC-MCP-11 | Transport cancel / disconnect | yes | cancel-a L2/L3 suite, F3 HTTP disconnect | MCP-P5-06 |
| SEC-MCP-12 | Unknown outcome not success | yes | `test_f3_streamable_http_unknown_task_id_is_missing_execution_binding` | MCP-P5-06 |
| SEC-MCP-13 | Concurrency admission bound | yes | `test_concurrent_admission_rejects_when_saturated` | MCP-P5-06 |
| SEC-MCP-14 | Retired / disabled exposure | yes | P4 lifecycle + discovery mount disabled | MCP-P5-07 |
| SEC-MCP-15 | Oversized / invalid schema deny | yes | discovery oversize + invalid tool id tests | MCP-P5-07 |

**MemoryProvider (B-P4-TEST-01):** reference-app MCP mount uses test harness qualification evidence (`testing_support/mcp_ac3_strict_product_memory_test_wiring.py`) — distinct from production `sqlite.user_profile` vendor qualification (PROD-Q / MEM campaign).
