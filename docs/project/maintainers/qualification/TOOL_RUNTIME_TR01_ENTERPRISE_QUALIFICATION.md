# TR-01 — ToolRuntime Enterprise Qualification (FINAL)

**Status:** **TR-01 = CLOSED / ENTERPRISE QUALIFIED**  
**TASK:** TR-01-RQ-FINAL — ToolRuntime Boundary Qualification  
**QUALIFICATION SHA:** `94c0abde805f3da244bd1eb3e9d5362e0ec2fdcc`  
**C1C baseline SHA:** `51df4755a0ae791c5ccc2a88b19780aa56233688`  
**Reconciliation:** no commits on `origin/development` between C1C and qualification HEAD touched `intergrax/runtime/nexus/tools`, `intergrax/runtime/tools`, `intergrax/tools`, or `intergrax/agents` tool seams.

Downstream subsystem findings are tracked in their owning roadmap items and do not reopen ToolRuntime unless its own boundary is violated.

## Executive decision

**TR-01-RQ-FINAL:** **APPROVED** — one canonical production tool invocation path; `RuntimeToolInvoker` is the atomic enforcement boundary; supported production tool bypass count **0** (static gates + U5 inventory).

Historical UAEP bypasses (sandbox `session.execute`, `runtime_bound_catalog` `service()` dispatch) are **removed** on the qualified branch. `BoundToolGateway` forwards all tools through `RuntimeToolGateway` / `invoke_catalog_tool_request` with `UAEPToolInvocationWiringResolver`.

**Next recommended workstream:** **GV-01 — Governance Adoption Sweep** (test/fixture drift on provider invoker tests reflects governance adoption, not ToolRuntime boundary gaps).

## Canonical ownership map

| Responsibility | Owner |
| --- | --- |
| Tool declaration | Tool contracts / Skills / Agent contract |
| Tool availability | Composition / `ToolProfile` |
| Tool authorization | Governance |
| Invocation enforcement | `RuntimeToolInvoker` |
| Physical handler dispatch | `ToolExecutor` / handler |
| Execution identity | Execution Engine |
| Sandbox isolation | Sandbox subsystem (contracts + attestation) |
| Persistence | Owning subsystem / provider |
| Provider integration | Integrations / domain |
| Evidence | Evidence |
| Telemetry | Observability |

## Canonical invocation spine

```text
caller / agent / planner / protocol
  → tool intent (ToolRequest / plan tool_ids)
  → tool availability / scope (ToolProfile, allowed_tools, ToolScopePolicy)
  → governance (declarative policy, fresh meaningful side-effect authorization)
  → RuntimeToolGateway / catalog_dispatch / ToolRuntime plan paths
  → BoundToolGateway (UAEP adapter — bind context, ToolRequest, forward)
  → ToolExecutionRequest
  → RuntimeToolInvoker (lookup, scope, auth, wiring, validation, timeout/retry/idempotency, dispatch)
  → ToolExecutor / handler
  → domain / provider contract
  → output validation, evidence / observability hooks
```

**Code anchors:** `intergrax/runtime/nexus/tools/tool_gateway.py`, `catalog_dispatch.py`, `invoker.py`, `uaep_tool_gateway.py`, `intergrax/tools/invocation_wiring.py`, `registry_tool_executor.py`.

## Entry path matrix (qualified HEAD)

| Entry path | Invoker | Bypass |
| --- | ---: | ---: |
| `RuntimeToolGateway` catalog id | yes | no |
| `catalog_dispatch` / plan | yes | no |
| `ToolRuntime.invoke` RAG/websearch/tools | yes | no |
| `ctx.invoke_tool` → `BoundToolGateway` | yes | no |
| MCP catalog tools (host-wired invoker) | yes | no |
| ACP `invoke_tool` → bound gateway | yes | no |
| UAEP declarative catalog | yes | no |

**SUPPORTED PRODUCTION TOOL BYPASS COUNT:** 0  
**UNKNOWN PATH COUNT:** 0

## Typed invocation wiring ABI (C2 + RX)

```text
ToolRegistrationWiringView + ToolInvocationContext
  → ToolInvocationWiringResolver
  → ToolInvocationWiring (immutable)
  → RuntimeToolInvoker._apply_invocation_wiring
  → legacy ToolWiringContext adapter (private — not canonical plugin ABI)
  → handler
```

Static gates: `tests/unit/runtime/tools/test_tool_eng_rx_invocation_wiring.py` (RX static gates, C1/C2 reflection-free canonical module).

## Sandbox / Memory / Workspace boundaries (ToolRuntime-only)

- **Sandbox:** `sandbox.exec` handler may call `session.execute` **inside** admitted handler after invoker governance; UAEP gateway does not call `session.execute`. Isolation authority via `RuntimeSandboxIsolationAuthority` + attestation (C1A/C1B/C1C).
- **Memory / Workspace:** runtime-bound tool IDs route through invoker; `WorkspaceExecutionPort` on wiring overlay; no `runtime_bound_catalog` service dispatch.

## Regression evidence (TR-01-RQ-FINAL session)

| Suite | Result |
| --- | --- |
| `tests/unit/runtime/nexus/tools` | pass (in combined run) |
| `tests/unit/runtime/tools` | pass (in combined run) |
| `tests/integration/runtime/test_tool_loop_integration.py` | pass |
| `tests/integration/runtime/test_sandbox_uaep.py` | pass |
| `tests/unit/runtime/tools/test_fresh_side_effect_authorization.py` | pass |
| `tests/unit/runtime/tools/test_tool_eng_rx_invocation_wiring.py` | pass |
| `tests/unit/runtime/architecture/test_platform_execution_unification_u5_final_zero_bypass.py` | pass |
| `tests/unit/runtime/sandbox/test_auth_c1a_*` + `test_cap_c1c_*` | pass |
| Combined TR-01 batch | **837 passed**, 9 failed, 2 skipped |

### Failures not blocking TR-01

| Test / area | Owner | Reason |
| --- | --- | --- |
| Provider invoker tests (workspace, websearch, RAG, Jira, Confluence, sandbox builder) | GV-01 / test fixtures | `MeaningfulSideEffectAuthorizationRequiredError` — invoker fail-closed; fixtures lack governed execution scope |
| `test_tool_planning_prompts_yaml` | Skills / prompts | prompt contract drift |
| `test_p0_frozen_child_execution_runner_import_surface` | Execution Engine | import inventory gate — not tool spine |
| `test_mcp_canonical_execution` (2 tests) | Execution Engine / host tests | mock `Execution.execute` signature drift (`held_root_capacity_permit`) |

## TR-FINAL scenario mapping

| ID | Evidence |
| --- | --- |
| TR-FINAL-1..5 | `test_tools_side_effect_safety`, `test_fresh_side_effect_authorization`, nexus invoker policy tests |
| TR-FINAL-6..9 | `test_tool_eng_rx_invocation_wiring`, provider tests (governance deny path), `test_sandbox_uaep` |
| TR-FINAL-10..12 | `test_mcp_canonical_execution` (partial), ACP/U5 gates, `test_sandbox_uaep`, declarative invoker |
| TR-FINAL-13..14 | RX/C2 wiring + custom resolver tests in `test_tool_eng_rx_invocation_wiring.py` |
| TR-FINAL-15..17 | `test_tool_runtime_scope_*`, `test_tool_runtime_authority_closure`, dependency admission tests |
| TR-FINAL-18..19 | `test_runtime_tool_invoker_duration`, idempotency / retry tests in runtime/tools |
| TR-FINAL-20 | RX static gates + U5 zero bypass |

## Historical closeout chain (retained)

**TOOL-ENG-RX / C1 / C2:** CLOSED — per-invocation wiring, typed ABI, UAEP convergence.  
**TR-01-RQ-C1A / C1B / C1C:** CLOSED — sandbox isolation authority transport and contract purity.

Prior audit at `12d5b7d50` documented pre-RX bypasses; superseded by implementation above.

## Non-goals

SBX-01 full qualification, PLUG-02 dynamic mount, per-provider product certification, full Governance policy model audit (GV-01).
