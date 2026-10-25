# TOOL-LIFE-X R4 — Production E2E RCA & Final Qualification

| Field | Value |
| --- | --- |
| **START_HEAD** | `9f060bde3d3f1c0f83da34eaf1148607b0ed5edf` |
| **Publication** | See git `IMPLEMENTATION_SHA` / `PUBLICATION_HEAD` in terminal report after push |
| **Disposition** | Ready for independent EXT-L4 production-E2E audit (not parent CLOSED) |

## 1. Production call graph (mandatory TOOL-LIFE-X scope)

| # | Path | ENTRY → CALL SITE → OWNER → CONTRACT → STATE → EFFECT → TERMINAL |
| --- | --- | --- |
| 1 | Host construction | `build_harness_host_runtime` → `resolve_tool_life_host_bootstrap_wiring` → `wire_tool_life_durable_single_host_execution_recovery` → G-A07 env gate → `ToolLifeDurableSingleHostComposition` (default-off) |
| 2 | Tool invoker composition | `build_declarative_invoker_for_application_host` → `resolve_external_operation_state_store_for_tool_invoker` → `build_production_runtime_tool_invoker(external_operation_store=…)` → `RuntimeToolInvoker` |
| 3 | Tool admission | `RuntimeToolInvoker.invoke` → `_prepare_invocation` / MSE gate → `_execute_once` |
| 4 | External operation attempt | `_build_tool_external_operation_attempt` → `ToolExternalOperationAttempt` → `ExternalOperationStateStore` + `ProcessLocalExternalOperationOwner` |
| 5 | **Authoritative bind (RCA fix)** | `_execute_once` → `before_physical_submit(catalog_tool_id, physical_idempotency_key)` → `bind_sanctioned_redrive_effect_before_tool_physical_execution` → `bind_authoritative_redrive_effect_to_operation` → trusted registry (`resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool`) |
| 6 | Physical tool effect | `RegistryToolExecutor` / provider handlers (HTTP/MCP/DB per tool) |
| 7 | HTTP provider | `intergrax.tools.providers.http.service.http_request` (qual registry: `http_rest` + `http_request` / `http_blind_post`) |
| 8 | Durable store | `SqliteToolLifeDurableStoresBundle.external_operation_state` (CAS, binding fields, worker epoch) |
| 9 | Revocation / certified loss | `worker_execution_ownership_revocation` + `build_worker_execution_certified_owner_loss_certifier` (composition; activation default-off) |
| 10 | UNKNOWN / reconciliation | `recover_running_operation_after_certified_owner_loss` → `admit_unknown_external_operation_for_reconciliation` |
| 11 | Redrive authority | `transfer_redrive_authority_after_reconciliation` (Class B only; bound digest match) |
| 12 | LLM external ops | `BaseLLMAdapter` optional `ExternalOperationStateStore` (parallel spine; out of R4 HTTP E2E acceptance unit) |
| 13 | `execute_contained_provider_call` | Enterprise provider admission spine (R2 diagnostic); not the tool-catalog HTTP redrive owner |

**Bypass closed by this change:** durable redrive binding was implemented only in qualification helpers/tests; production `RuntimeToolInvoker` did not call `bind_authoritative_redrive_effect_to_operation`, and host bootstrap did not forward `external_operation_state` into the declarative invoker.

## 2. Systemic root causes

1. **Module vs path:** `bind_authoritative_redrive_effect_to_operation` had zero production call sites (only tests).
2. **Composition vs execution:** G-A07 materialized `tool_life_execution_recovery` on `HarnessHostRuntime` but did not connect `bundle.external_operation_state` to `RuntimeToolInvoker`.
3. **Acceptance unit drift:** EXT-L4 physical proofs invoked store/redrive APIs directly instead of canonical host + invoker entry.

## 3. Implemented corrections

| Area | Change |
| --- | --- |
| Contracts | `resolve_sanctioned_trusted_redrive_effect_binding_for_catalog_tool` |
| Runtime | `tool_external_operation_redrive_binding.bind_sanctioned_redrive_effect_before_tool_physical_execution` |
| Tool attempt | `ToolExternalOperationAttempt.before_physical_submit` records trusted bind before `mark_running` |
| Invoker | `_execute_once` passes `catalog_tool_id` + `physical_idempotency_key` |
| Applications | `tool_life_declarative_tool_invoker_wiring` + `declarative_tool_wiring` / `harness_host_runtime` / ACP / LKW factory forward store |

## 4. Production call-site proof

- **Bind:** `intergrax/runtime/nexus/tools/invoker.py` (`before_physical_submit` with tool identity).
- **Store wire:** `intergrax/applications/_shared/harness_host_runtime.py` (bootstrap before declarative invoker).
- **Tests:** `tests/unit/applications/test_tool_life_x_r4_production_entry_e2e_qualification.py` (harness bootstrap + invoker binding observable on store).

## 5–9. Physical / revoke / redrive evidence

| Evidence | Location | Notes |
| --- | --- | --- |
| HTTP Class B lost-ack + redrive | `test_tool_life_x_r4_ext_l4_integrated_http_redrive_physical.py` | Physical loopback; bind step may use test helper — invoker bind proven separately |
| Binding adversarial | `test_tool_life_x_r4_ext_l4_binding_adversarial.py` | 37 tests consolidated PASS |
| Scenario 24 | `test_tool_life_x_r4_f6_scenario_24_recovery.py` | Recovery chain |
| L3 R8–R10 | Prior L3 modules | Reused; applicability unchanged |

## 10. Consolidated test results (this closure)

```text
uv run pytest tests/unit/applications/test_tool_life_x_r4_production_entry_e2e_qualification.py \
  tests/unit/applications/test_tool_life_host_bootstrap_wiring.py \
  tests/unit/runtime/external_operations/test_tool_life_x_r4_*.py
→ 37 passed (L2 consolidated, 2026-10-25 session)
```

Log: `.tmp/session/TOOL-LIFE-X-R4/pytest-ext-l4-l2-consolidated-final.log`

## 11–14. EC / Scenarios / G-A / FRZ-ECAP

Prior bundles remain authoritative for per-ID narrative: `TOOL_LIFE_X_R4_FINAL_PARENT_QUALIFICATION_AND_EXT_L4_BUNDLE.md`, `TOOL_LIFE_X_R4_EXT_L4_CORRECTIVE_EVIDENCE_BUNDLE.md`. **Delta for this RCA:** production integration PASS for bind + store wire; component-only PASS is no longer sufficient for those two gaps.

| Gate | Status |
| --- | --- |
| G-A07 | Qualified default-off composition + **execution-path store forward** (activation still operator) |
| G-A02 | Trusted registry enforced on invoker bind path |
| Production recovery activation | **Not authorized** — separate OD |
| MCP HTTP catalog Option C | Remains disabled |
| PostgreSQL Docker L3 | Not re-run in this session (Windows C10 scope) |

## 15. Operator decisions

1. **OD-R4-G-A07-PROD-ACTIVATE** before default-on durable wiring in production.
2. Independent EXT-L4 audit of full HTTP redrive through **live** catalog HTTP tool + harness strict mode (stretch beyond invoker bind proof).
3. Whether parent TOOL-LIFE-X reconciliation accepts Path B with this integration delta.

## 16. EXT-L4 readiness

**TOOL-LIFE-X R4 — READY FOR INDEPENDENT PRODUCTION-E2E EXT-L4 AUDIT** (integration gaps addressed; no self-CLOSED).
