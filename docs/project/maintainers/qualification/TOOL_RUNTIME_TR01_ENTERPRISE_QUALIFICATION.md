# TR-01 — ToolRuntime Enterprise Closure (audit closeout)

**Status:** TOOL ENGINE REOPEN REQUIRED (TR-01 not closed)  
**TASK SHA:** audit at `12d5b7d50fc5f76943cc5c3fdd71534fb40c9f16` (no TR-01 implementation commit)  
**Baseline cited in task:** `1642ff65c5402264101477c1b2dd16b0d0dfe5cc`  
**Audited HEAD:** `12d5b7d50fc5f76943cc5c3fdd71534fb40c9f16`  
**Concurrent tool-engine delta vs baseline:** none (`git log 1642ff65c..HEAD -- intergrax/runtime/nexus/tools` empty)

## Executive decision

**TOOL_ENGINE_FOUNDATION_STATUS:** structurally sound **core invoker + catalog spine**, but **not enterprise-closed** until UAEP/runtime-bound and sandbox UAEP fast paths are removed or folded into a single physical enforcement contract.

**TR-01 STATUS:** TOOL ENGINE REOPEN REQUIRED — convergence without engine contract change would mask alternate execution authorities.

## Canonical ownership map (as-built)

| Responsibility | Canonical owner |
| --- | --- |
| Tool availability (host) | `ToolProfile` + composition `ToolRegistry` bootstrap |
| Agent declaration | `AgentContract.allowed_tools` / skill `tool_ids` |
| Runtime permission | `RuntimePolicyBundle.tool_access`, declarative governance |
| Per-call enforcement | `RuntimeToolInvoker` (+ `ToolScopePolicy`) |
| Physical handler execution | `ToolExecutor` / `RegistryToolExecutor` → handler |
| Vendor implementation | Integration providers behind handlers |
| Sandbox isolation | Sandbox subsystem (`SandboxSession`, contracts) |
| Orchestration / ordering | `ToolInvocationPattern`, planners (propose only) |

## Canonical invocation spine (production-shaped, registry tools)

```text
tool intent (ToolRequest / plan tool_ids)
  → RuntimeToolGateway / catalog_dispatch / ToolRuntime plan paths
  → ToolExecutionRequest
  → RuntimeToolInvoker (scope, declarative policy, fresh side-effect auth, validation, timeout/retry/idempotency, evidence diag)
  → ToolExecutor / handler
  → integration / RAG / websearch backend
```

**Code anchors:** `intergrax/runtime/nexus/tools/tool_gateway.py`, `catalog_dispatch.py`, `invoker.py`, `registry_tool_executor.py`.

## Alternate physical execution authorities (blockers)

### 1. `BoundToolGateway` — `sandbox.exec` (P1)

`intergrax/runtime/nexus/tools/uaep_tool_gateway.py` routes `SANDBOX_TOOL_NAME` to `_invoke_sandbox`, which calls `session.execute` directly after `ToolAccessPolicy` only.

Catalog path `intergrax/tools/providers/sandbox/service.py` (`sandbox_exec`) uses the same session surface but is reached via `RuntimeToolInvoker` when invoked as a registered catalog tool.

**Gap:** UAEP `ctx.invoke_tool(sandbox.exec)` does not cross `RuntimeToolInvoker` → missing unified side-effect authorization, idempotency, declarative policy stack, and invoker evidence taxonomy on that entry.

**Classification:** BYPASS (production — `tests/integration/runtime/test_sandbox_uaep.py`).

**Not** intentional sandbox isolation ownership transfer; isolation remains in Sandbox. Missing piece is **ToolRuntime enforcement**, not isolation.

### 2. `runtime_bound_catalog` — workspace/memory/harness/cost tools (P1)

`intergrax/runtime/nexus/tools/runtime_bound_catalog.py` invokes provider `service(ctx, params)` directly from `BoundToolGateway` **before** `runtime_state` routing.

**Root cause:** `ServiceToolHandler` binds `ToolWiringContext` at **registration** time (`intergrax/tools/core/handler.py`), while UAEP steps inject per-step dependencies (`shadow_workspace`, `run_budget`, trace reader) via `exec_ctx.metadata`. `build_runtime_bound_context` exists to supply dynamic wiring.

**Gap:** No contract seam on `RuntimeToolInvoker` / `ToolExecutionRequest` for per-call wiring resolution. Converging by adapter without that seam duplicates enforcement or uses stale bootstrap wiring.

**Classification:** BYPASS (production UAEP path).

## BoundToolGateway verdict

**Facade over ToolRuntime for catalog + capability tools when `runtime_state` is bound.**  
**Owns alternate execution** for `sandbox.exec` and runtime-bound tool IDs today.

## Entry path matrix (summary)

| Entry path | ToolRuntime | Gateway | Invoker | Governance (full) | Bypass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RuntimeToolGateway` catalog id | partial | yes | yes | yes | no |
| `catalog_dispatch` / plan | via invoker | — | yes | yes | no |
| `ToolRuntime.invoke` RAG/websearch/tools | yes | internal | yes (catalog branch) | yes | no |
| `ctx.invoke_tool` → BoundToolGateway (catalog) | via gateway | yes | yes | yes | no |
| `ctx.invoke_tool` → sandbox | no | yes | **no** | partial | **yes** |
| `ctx.invoke_tool` → runtime-bound ids | no | yes | **no** | partial | **yes** |
| MCP / Nexus planned tools | via gateway/runtime | yes | yes | wired | no* |

\*Assumes host wired `tool_invoker`; unwired hosts fail closed on catalog paths.

**SUPPORTED_BYPASS_COUNT:** 2 (sandbox UAEP fast path, runtime-bound catalog dispatch).

## Tool Engine enterprise matrix (high level)

| Concern | Status |
| --- | --- |
| Stable contracts (`ToolContract`, execution models) | PASS |
| `RuntimeToolInvoker` atomic enforcement | PASS (on registry path) |
| Provider neutrality in core | PASS |
| Plugin handlers via registry | PASS |
| No global mutable registry authority | PASS (composition bootstrap) |
| Permission narrowing | PASS (documented + tests in nexus/tools) |
| Fresh side-effect auth | PASS on invoker path; **FAIL** on bypass paths |
| Retry / idempotency / timeout | PASS on invoker path |
| Sandbox boundary ownership | PASS (contract); **FAIL** enforcement parity on UAEP sandbox |
| No execution bypass | **FAIL** (see above) |

## Required return task

**TOOL-ENG-RX — Per-invocation wiring resolution and UAEP gateway convergence**

1. Add contract-driven **per-call `ToolWiringContext` resolution** at the invoker boundary (without vendor leakage).
2. Route `BoundToolGateway` sandbox + runtime-bound tools through `invoke_catalog_tool_request` / invoker when dependencies resolvable; fail closed otherwise.
3. Retire direct `service(ctx, params)` dispatch from `runtime_bound_catalog` for production.
4. Re-run TR-01 qualification suite (TR-Q1–TR-Q20) after RX.

## TOOL-ENG-RX (implementation evidence)

**Status:** CLOSED on branch `development` after TOOL-ENG-RX-C1 (awaiting independent GitHub audit; TR-01 not closed).

- **Contract:** `ToolInvocationWiringResolver`, `ToolInvocationContext`, `ToolWiringOverlay`, `ToolInvocationWiringRequirements` (`intergrax/tools/invocation_wiring.py`, `invocation_wiring_requirements.py`).
- **Resolution site:** `RuntimeToolInvoker._apply_invocation_wiring` (invoker-owned; read-only; no tool-id branching).
- **UAEP adapter:** `UAEPToolInvocationWiringResolver` + `BoundToolGateway` routes all tools via `RuntimeToolGateway` / `invoke_catalog_tool_request`.
- **Bypass removal:** `uaep_tool_gateway` no longer calls `session.execute` or `invoke_runtime_bound_tool`; `runtime_bound_catalog` is ID metadata only.
- **Static gates:** `tests/unit/runtime/tools/test_tool_eng_rx_invocation_wiring.py` (RX-T1–T4, T6, static bypass gates).
- **TR-01-RQ:** required before TR-01 closeout.

## TOOL-ENG-RX-C1 — Typed wiring ABI and encapsulation

**Status:** CLOSED on branch `development` (awaiting independent GitHub audit).

- **Typed `ToolWiringOverlay`:** explicit platform contracts (`ShadowWorkspace`, `TaskMemoryViewBinding`, `RunTraceReaderBinding`, `RunBudget`, `BudgetEnvelope`, `ResourceQuota`, `SandboxExecCapable`); no `Any` / `object` on public overlay fields.
- **Registration wiring seam:** `WiringContextToolHandler.registration_wiring` (read-only); `registration_wiring_for_handler` does not read `handler._ctx`.
- **Resolver trust:** `ensure_tool_wiring_overlay` fail-closed at invoker boundary.
- **Provider-neutral runtime-bound IDs:** `runtime_bound_catalog` imports `tool_ids` modules only (no `*.service` imports).
- **Tests:** C1-T1–T4, T6, T11, T14 in `test_tool_eng_rx_invocation_wiring.py` plus existing RX gates.

## Tests executed (audit session)

- `tests/unit/runtime/nexus/tools` — **275 passed**, 1 skipped (UE-8B)
- `tests/unit/runtime/tools` + invoker governance matrix — **29 failed** on HEAD (fixture `Cfg` lacks `production_mode`; invoker `_require_agent_runtime_governance`); treat as **environment/HEAD regression**, not TR-01 scope fix
- RI foundation tests included in combined run — passed except where bundled with failing invoker tests

## Non-goals (unchanged)

SBX-01 full qualification, PLUG-02 dynamic mount, individual provider product qualification.

## Known non-goals of this document

Does not certify every catalog tool/provider — only engine/spine audit at cited SHA.
