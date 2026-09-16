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

**Status:** CLOSED (RX + C1 + C2 on branch `development`; awaiting independent GitHub audit; TR-01 not closed).

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

## TOOL-ENG-RX-C2 — Canonical invocation wiring ABI

**Status:** CLOSED on branch `development` (awaiting independent GitHub audit; TR-01 not closed).

```text
ToolWiringContext (legacy static composition)
        ↓ private adapter only (invocation_wiring_adapter.py)

ToolInvocationContext + ToolRegistrationWiringView
        ↓ ToolInvocationWiringResolver
        ↓ ToolInvocationWiring (immutable)
        ↓ RuntimeToolInvoker (compose + validate + adapter)
        ↓ handler (ToolWiringContext effective context)
```

- **Canonical ABI:** `ToolInvocationWiring`, `ToolRegistrationWiringView`, `ToolInvocationWiringResolver` — no `ToolWiringContext` on resolver seam; no `ShadowWorkspace` in `invocation_wiring.py`.
- **Workspace port:** `WorkspaceExecutionPort` (`intergrax/runtime/workspace/execution_port.py`); catalog workspace tools consume port via effective handler context.
- **Typed bindings:** `TaskMemoryViewBinding` (`JsonObject` / `TaskMemoryRecord`); `RunTraceReaderBinding` (`PersistedRun`, `RunSummary`).
- **Requirements:** `ToolInvocationWiringRequirements` validated against composed `ToolInvocationWiring`, not legacy bag fields.
- **Tests:** RX/C1 gates + C2-T1–T5 in `test_tool_eng_rx_invocation_wiring.py`; `tests/unit/runtime/nexus/tools` regression.

## TR-01-RQ-C1A — Explicit sandbox isolation authority wiring

**Status:** CLOSED on branch `development` (awaiting independent GitHub audit; TR-01 not closed).

- **Invariant:** `configured != available != authorized != effective`; session availability must not mint sandbox isolation authority.
- **Removed:** `runtime_host_sandbox_isolation_profile()` / session-derived `effective_environment_profile` synthesis in `invocation_wiring_adapter.py`.
- **Authority contract:** `RuntimeSandboxIsolationAuthority` (`intergrax/contracts/runtime_sandbox_isolation_authority.py`) — explicit, immutable, provider-neutral; Tier-3 `ApplicationEnvironmentProfile` / pinned `effective_profile_revision` (legacy extras) or typed `ToolWiringContext.sandbox_isolation_authority` (C1B).
- **UAEP composition:** optional `UAEPExecutor.sandbox_isolation_authority` merges explicit authority into registration wiring when no profile authority is already present (never from `sandbox_session`).
- **Availability:** `ToolInvocationWiring.sandbox_session` and `overlay_invocation_sandbox_availability` remain provider/session capability only.
- **Separation:** tool registration / `ToolProfile` / agent `allowed_tools` declare scope or availability — not environment isolation authority; governance answers authorization to proceed — not isolation authority.
- **Tests:** `tests/unit/runtime/sandbox/test_auth_c1a_sandbox_isolation_authority.py` (AUTH-C1A-1..13, static gates).

**TOOL-ENG-RX:** CLOSED (RX + C1 + C2). **TR-01-RQ-C1A:** CLOSED (await audit). **TR-01-RQ:** NEXT.

## TR-01-RQ-C1B — Typed authority transport & provider capability attestation

**Status:** CLOSED on branch `development` (awaiting independent GitHub audit; TR-01 not closed).

- **Authority transport:** `ToolWiringContext.sandbox_isolation_authority` (`ProfileSandboxIsolationSource`) — no `extras["runtime_sandbox_isolation_authority"]`; UAEP uses `apply_runtime_sandbox_isolation_authority`.
- **Precedence:** pinned `effective_profile_revision` → legacy `effective_environment_profile` → explicit runtime host authority (typed field); runtime host cannot widen pinned profile.
- **Availability vs capability vs authority vs governance:** `sandbox_session` / `SandboxExecCapable` = execution availability only; isolation authority remains composition-owned; governance unchanged; resolver = authority ∩ requirement ∩ attested provider capabilities.
- **Provider trust:** `SandboxExecCapable` alone does not attest filesystem/process/network guarantees; `SandboxSecurityCapable.security_capabilities()` is the trusted evidence surface; plain exec-only providers fail closed for isolation-sensitive tools.
- **Adapters:** `capabilities_from_attested_exec_session` / `capabilities_from_security_attestation` — no fabricated LOCAL kind or workspace/sandbox flags for unattested exec endpoints.
- **Tests:** `test_auth_c1b_sandbox_authority_transport.py` (AUTH-C1B-1..6), `test_cap_c1b_provider_capability_trust.py` (CAP-C1B-1..8); C1A suite updated for typed transport.

**TR-01-RQ-C1B:** CLOSED (await audit). **TR-01:** BLOCKED until C1C audit. **TR-01-RQ-C1C:** in flight.

## TR-01-RQ-C1C — Contract layer purity & complete capability evidence

**Status:** CLOSED on branch `development` (awaiting independent GitHub audit; TR-01 not closed).

```text
explicit authority
       ∩
tool isolation requirement
       ∩
complete provider attestation
       ∩
governance
       ↓
effective sandbox execution environment
```

- **Contract dependency direction:** `intergrax/contracts/runtime_sandbox_isolation_authority.py` is a pure platform contract (no `intergrax.tools` / runtime / agents imports). Wiring helpers live in `intergrax/tools/registry/sandbox_isolation_wiring.py` (`tools → contracts`).
- **Attestation provenance:** `SandboxSecurityCapabilities` attests security-sensitive facts (`supports_sandboxed_exec`, `supports_workspace_write`, `filesystem_access`, `process_execution`, network egress evidence, `provider_id`, `isolation_tier`). `project_provider_capabilities_from_security` normalizes only — never upgrades unknown to supported.
- **Unattested providers:** plain `SandboxExecCapable`, unattested `SandboxHostBackend`, and incomplete `SandboxSecurityCapable` evidence omit provider capability projection (fail closed for isolation-sensitive tools).
- **Tests:** `test_runtime_sandbox_isolation_authority_import_gate.py`, `test_cap_c1c_sandbox_evidence_and_contract_purity.py`; C1A/C1B suites retained.

**TR-01-RQ-C1C:** CLOSED (await audit). **TR-01:** READY FOR REQUALIFICATION. **TR-01-RQ:** NEXT.

## Tests executed (audit session)

- `tests/unit/runtime/nexus/tools` — **275 passed**, 1 skipped (UE-8B)
- `tests/unit/runtime/tools` + invoker governance matrix — **29 failed** on HEAD (fixture `Cfg` lacks `production_mode`; invoker `_require_agent_runtime_governance`); treat as **environment/HEAD regression**, not TR-01 scope fix
- RI foundation tests included in combined run — passed except where bundled with failing invoker tests

## Non-goals (unchanged)

SBX-01 full qualification, PLUG-02 dynamic mount, individual provider product qualification.

## Known non-goals of this document

Does not certify every catalog tool/provider — only engine/spine audit at cited SHA.
