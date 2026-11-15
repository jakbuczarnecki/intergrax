<!-- Campaign: EQO-A | BASELINE_SHA: 258bbf6abb6fdbb92ef0185d058527fbb1e47f04 | generated_utc: 2026-11-15T18:58:02Z | source_registry_sha256: 557e1a8981e8d37f2431249e363afd45844edc06d651527689871da08f0bb264 -->

# Static boundary sweep (A4-T04)

Allowed read-only scanners run from the audit worktree at the baseline (see `TOOL_RUN_LOG.md`). Exclusions: `build/`, `applications/*/docker/runtime-context/` (SX-06). Dynamic-import grep: see EQOA-A4-007.

| Run | Exit | Duration (s) | Log |
|---|---|---|---|
| check_no_upward_application_imports | 1 | 0.7 | `logs/A4-check_no_upward_application_imports.log` |
| check_agents_no_pseudo_shared_packages | 0 | 0.1 | `logs/A4-check_agents_no_pseudo_shared_packages.log` |
| check_integration_vendor_imports | 0 | 1.1 | `logs/A4-check_integration_vendor_imports.log` |
| check_intergrax_no_applications_imports | 1 | 0.6 | `logs/A4-check_intergrax_no_applications_imports.log` |
| check_agents_no_tier3_imports | 0 | 0.2 | `logs/A4-check_agents_no_tier3_imports.log` |
| check_agents_no_vendor_sdk_imports | 0 | 0.2 | `logs/A4-check_agents_no_vendor_sdk_imports.log` |
| check_agents_vendor_imports | 0 | 0.2 | `logs/A4-check_agents_vendor_imports.log` |
| check_context_tier0_import_boundary | 0 | 0.1 | `logs/A4-check_context_tier0_import_boundary.log` |
| check_llm_routing_tier_boundary | 0 | 0.1 | `logs/A4-check_llm_routing_tier_boundary.log` |
| check_dependency_ownership | 1 | 0.2 | `logs/A4-check_dependency_ownership.log` |
| check_agent_registry_bypass | 1 | 0.1 | `logs/A4-check_agent_registry_bypass.log` |
| check_architecture_debt_register | 0 | 0.1 | `logs/A4-check_architecture_debt_register.log` |
| check_tenant_storage_isolation | 0 | 4.6 | `logs/A4-check_tenant_storage_isolation.log` |
| check_runtime_event_tenant_propagation | 1 | 0.1 | `logs/A4-check_runtime_event_tenant_propagation.log` |
| check_legacy_modules_removed | 0 | 0.6 | `logs/A4-check_legacy_modules_removed.log` |
