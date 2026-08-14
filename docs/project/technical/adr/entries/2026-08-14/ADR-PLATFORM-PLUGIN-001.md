# ADR-PLATFORM-PLUGIN-001: Declarative Policy REQUIRE_HITL -> Canonical Nexus HITL Bridge

| Field | Value |
|-------|-------|
| **Status** | Proposed / Ready for Review |
| **Date** | 2026-08-14 |
| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 |
| **Deciders** | Platform / Nexus runtime |
| **Related** | [`PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md`](../../../../maintainers/plans/PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md) BLOCK B · ENTERPRISE-4 review-fix `7154d29d` · [`ADR-POLICY-SIDE-EFFECT-001`](../2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md) · [`ADR-GOVERNED-CONTINUATION-001`](../2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) |

## Context

ENTERPRISE-4 (BLOCK B / CAND-007) wires typed `DeclarativePolicyEnforcer` at `RuntimeToolInvoker.invoke()` for tool invocations. `DENY` and `AUDIT_ONLY` behave correctly. `REQUIRE_HITL` in `ENFORCE` mode raises `DeclarativePolicyHitlRequiredError` before the tool handler executes — correct synchronous block, but **not** canonical HITL.

Canonical Nexus HITL already exists as a single system:

`AgentDecision` / `ExecutionInterruptHandler` -> `GovernanceResolution` -> `AgentExecutionStatus.NEEDS_INPUT` -> `NexusGraphRunner._handle_needs_input` -> `HumanPauseCoordinator.apply_pause` -> `TaskState.WAITING_FOR_HUMAN` -> checkpoint -> `NexusIntakeRunner` resume/reject/escalate.

This ADR freezes how declarative `REQUIRE_HITL` joins that chain without a second pause coordinator, queue, or Tier-3 callback injection.

## Current architecture

### Standard production execution chain (Nexus graph path)

| Step | Owner | Input -> Output |
|------|-------|----------------|
| 1 | `NexusGraphRunner.run` | `Task` + `NexusPlan` + `ExecutionGraph` -> `GraphPhaseOutcome` |
| 2 | `GraphExecutor.execute` / `_execute_node` | node -> `AgentExecutionResult` list |
| 3 | `GraphExecutor.execute_fn` | builds `RuntimeRequest`, calls `AgentEngine.run_agent_with_result` |
| 4 | `AgentEngine.run_agent_with_result` | `RuntimeRequest` -> `AgentExecutionResult` via `runtime_answer_to_agent_result(governance=...)` |
| 5 | `UAEPExecutor` (`intergrax/agents/uaep.py`) | per-step: execute step -> `AgentDecision` -> `ExecutionInterruptHandler.resolve_decision` -> `GovernanceResolution` |
| 6 | `runtime_mapping.runtime_answer_to_agent_result` | `GovernanceResolution.should_pause` -> `AgentExecutionStatus.NEEDS_INPUT` + attaches `agent_decision`, `human_request`, `execution_interrupt` |
| 7 | `GraphExecutor._execute_node` | `NEEDS_INPUT` -> node `PENDING`, `metadata["governance_pause"]=True` |
| 8 | `NexusGraphRunner._handle_needs_input` | last `AgentExecutionResult` -> `HumanPauseCoordinator.apply_pause` -> `TaskState.WAITING_FOR_HUMAN` -> `maybe_checkpoint` |
| 9 | `NexusIntakeRunner.run` (on resume) | `HumanPauseCoordinator.verdict_from_task` -> approve/reject/escalate branches |
| 10 | `GraphExecutor.execute_fn` (resume) | `task.options.human.is_resumed` -> `request.metadata["human_approved"]=True` |

### Tool invocation sub-paths (all converge on `RuntimeToolInvoker`)

| Path | Caller of `RuntimeToolInvoker.invoke` | Current HITL behavior |
|------|--------------------------------------|------------------------|
| LLM tool loop | `tool_loop._invoke_planned_call` | Exception propagates uncaught |
| Catalog gateway | `catalog_dispatch.invoke_catalog_tool_request` | `except Exception` -> `ToolResponseStatus.FAILED` (incorrect for HITL) |
| ACP declarative | `CatalogDeclarativeToolInvoker` -> `catalog_dispatch` | Same swallow as gateway |
| Step kernel declarative | `execute_declarative_actions` -> `DeclarativeToolInvoker` | No `hitl_required` status; failures map to `TOOL_FAILED` |

### Canonical HITL contract (verified)

`ExecutionInterruptHandler.resolve_decision()` returns `GovernanceResolution`:

- `should_pause == True` when `agent_decision.type == REQUEST_HUMAN` **or** `policy_decision.action == REQUIRE_HUMAN`.
- Creates `HumanRequest` when missing and pause is required.
- `HumanPauseCoordinator` owns persisted pause state (`TaskGovernanceState.pause_record`, `human_request`, `paused`).

**Consumer:** Nexus orchestration consumes `GovernanceResolution` **indirectly** via `AgentExecutionResult` produced by `runtime_answer_to_agent_result`. `NexusGraphRunner` checks `AgentExecutionStatus.NEEDS_INPUT`, not `GovernanceResolution` directly.

**`ExecutionInterrupt`:** Used for `AgentDecisionType.INTERRUPT` only. Declarative `REQUIRE_HITL` is **not** an interrupt; `GovernanceResolution.interrupt` remains `None` for this bridge.

### Meaningful-side-effect precedent

`MeaningfulSideEffectAuthorizationBoundary.authorize()` returns `MeaningfulSideEffectAuthorizationResult` with `requires_governed_continuation: bool` when `PolicyAction.REQUIRE_HUMAN`. It does **not** call Nexus HITL directly; higher layers (e.g. external-work adapter) surface `GovernedContinuationRequest` and compose with `ExecutionInterruptHandler`.

**Reuse:** typed **control outcome** at a low boundary + **orchestration-owned** conversion to `AgentDecision` / governed continuation. **Do not** duplicate `GovernedContinuationRequest` for catalog tools; use existing UAEP -> `GovernanceResolution` path.

**Shortcoming:** meaningful-side-effect path is collaborative-work scoped; it does not define tool `idempotency_key` / step checkpoint semantics. This ADR extends the Nexus tool path instead.

## Problem

`DeclarativePolicyHitlRequiredError` blocks execution (handler call count = 0) but:

1. `catalog_dispatch` maps it to generic tool failure.
2. No `AgentDecision.REQUEST_HUMAN` / `HumanRequest` / `WAITING_FOR_HUMAN` transition occurs.
3. No typed approval scope for safe resume + policy re-evaluation.

## Invariants

- **One canonical HITL system** — `HumanPauseCoordinator` + existing intake/graph runners only.
- `NEW_DYNAMIC_ATTRIBUTE_WIRING = 0` — no `getattr`/`setattr`/magic exception-name routing; explicit typed contracts only.
- `runtime/policy` and `runtime/tools` **must not** import `Task`, `NexusHitlRunner`, or `HumanPauseCoordinator`.
- HITL is **not** `TOOL_ERROR` / `ToolExecutionResult.fail`.
- `AUDIT_ONLY` + `REQUIRE_HITL` must never pause (`enforced=False` -> no block, no exception).
- `DENY` precedence is absolute; human approval never bypasses `DENY`.

## Considered options

### Option A — Typed exception bridge at tool-invocation aggregate boundary (chosen)

`RuntimeToolInvoker` keeps raising `DeclarativePolicyHitlRequiredError`. Direct callers catch it and convert via a new Tier-1 bridge module into orchestration-legal signals that terminate in `AgentDecision.REQUEST_HUMAN`.

| Aspect | Assessment |
|--------|------------|
| Ownership | Bridge module in `runtime/nexus/tools/`; governance translation in existing UAEP/`runtime_mapping` |
| Sync/async | Exception unwinds sync invoke; async Nexus loop unchanged |
| `ToolExecutionResult` | Unchanged — represents completed execution only |
| DENY symmetry | Matches existing `DeclarativePolicyViolationError` pattern |
| Risk | Multiple catch sites — mitigated by allowlisted callers + shared bridge function |
| Migration | Low churn on `ToolInvokerProtocol` |

### Option B — `RuntimeToolInvoker` returns suspended control outcome

Extend `invoke()` to return `ToolExecutionResult | ToolInvocationSuspended`.

| Aspect | Assessment |
|--------|------------|
| Ownership | Invoker owns suspension typing |
| Cons | Breaks `ToolInvokerProtocol`; every caller must branch; suspension easily mistaken for failure; high API churn |
| Security | Weaker — callers may treat suspension as failure and retry |

**Rejected.**

### Option C — Move declarative HITL evaluation above `RuntimeToolInvoker`

Pre-flight policy in orchestration before invoke.

| Aspect | Assessment |
|--------|------------|
| Cons | Splits enforcement plane; duplicate evaluation; race between check and handler; contradicts ENTERPRISE-4 invoker-boundary design |
| Security | Weaker guarantee handler call count = 0 |

**Rejected.**

## Decision

**Choose Option A.**

### Frozen decisions

| # | Decision |
|---|----------|
| 1 | **Canonical bridge owner (translation):** `intergrax/runtime/nexus/tools/declarative_policy_hitl_bridge.py` — sole module converting `DeclarativePolicyHitlRequiredError` + invocation context -> `DeclarativePolicyHitlSignal` -> `AgentDecision` + optional pre-built `HumanRequest`. |
| 2 | **Canonical bridge owner (orchestration consumption):** existing `UAEPExecutor` step loop + `runtime_answer_to_agent_result` + `NexusGraphRunner._handle_needs_input` — no new pause owner. |
| 3 | **Low-level HITL signal type:** keep `DeclarativePolicyHitlRequiredError` (sync unwind). Add `DeclarativePolicyHitlSignal` (frozen dataclass) as the cross-layer typed payload extracted by the bridge. |
| 4 | **`DeclarativePolicyHitlRequiredError` remains?** **Yes** — internal synchronous signal inside `RuntimeToolInvoker` only. |
| 5 | **Translation to governance:** `DeclarativePolicyHitlSignal` -> `AgentDecision(type=REQUEST_HUMAN, reason=..., human_request=..., payload=DeclarativeHitlDecisionPayload)` -> `ExecutionInterruptHandler.resolve_decision()` -> `GovernanceResolution` -> `runtime_answer_to_agent_result` -> `NEEDS_INPUT`. |
| 6 | **`ExecutionInterrupt` involved?** **No.** |
| 7 | **`HumanRequest` owner:** bridge builds minimal `HumanRequest`; `ExecutionInterruptHandler` may enrich defaults if fields missing. Persisted by `HumanPauseCoordinator.apply_pause`. |
| 8 | **Pause owner:** `HumanPauseCoordinator` (unchanged). |
| 9 | **Approval scope:** one governed invocation: `(task_id, run_id, step_id, tool_id, idempotency_key, matched_rule_ids, human_request_id)`. |
| 10 | **Resume behavior:** graph node stays `PENDING`; step re-executes from checkpoint; same `ToolExecutionRequest` reconstructed; handler call count 0 before approval, exactly 1 after. |
| 11 | **Policy re-evaluation:** always on resume; `REQUIRE_HITL` satisfied only when typed `DeclarativeHitlApprovalGrant` matches invocation context. |
| 12 | **`DENY` after resume:** always blocks; grant never applies to `DENY`. |
| 13 | **Trace IDs:** `run_id`, `task_id`, `step_id`, `tool_id`, `human_request_id`, `pause_id`, `matched_rule_ids` (no tool args/secrets). |
| 14 | **IMPL-1 file allowlist:** see Implementation scope section. |

## Detailed control flow

### First invocation (ENFORCE + REQUIRE_HITL)

```text
RuntimeToolInvoker.invoke(request)
  -> DeclarativePolicyEnforcer.evaluate_tool_invocation
  -> should_block_execution && action==REQUIRE_HITL
  -> raise DeclarativePolicyHitlRequiredError(...)
       handler call count = 0

tool_loop._invoke_planned_call | catalog_dispatch | execute_declarative_actions
  -> catch DeclarativePolicyHitlRequiredError
  -> DeclarativePolicyHitlBridge.to_signal(exc, request)
  -> propagate DeclarativePolicyHitlSignal (not ToolExecutionResult.fail)

ACP / step kernel path:
  -> StepOutcome.pause_hitl(reason, diagnostics=signal.to_step_diagnostics())
  -> agent_decision_from_outcome -> AgentDecision.REQUEST_HUMAN

UAEP path:
  -> DeclarativePolicyHitlBridge.to_agent_decision(signal)
  -> ExecutionInterruptHandler.resolve_decision(...)
  -> GovernanceResolution(should_pause=True)
  -> runtime_answer_to_agent_result -> NEEDS_INPUT

NexusGraphRunner._handle_needs_input
  -> HumanPauseCoordinator.apply_pause
  -> TaskState.WAITING_FOR_HUMAN
  -> maybe_checkpoint
```

### Human APPROVE (existing API)

```text
Task.options.human.verdict = "approve"
NexusIntakeRunner -> clear_pause
GraphExecutor.execute_fn -> human_approved + DeclarativeHitlApprovalGrant injection
RuntimeToolInvoker.invoke (resume)
  -> policy re-evaluates; matching grant satisfies REQUIRE_HITL only
  -> handler executes once
```

## Approval scope

New contracts in `intergrax/contracts/declarative_hitl.py`:

- `DeclarativePolicyHitlSignal` — bridge transport from tool boundary
- `DeclarativeHitlDecisionPayload` — `AgentDecision.payload` (`extra="forbid"`)
- `DeclarativeHitlApprovalGrant` — persisted on approve in `TaskGovernanceState.declarative_hitl_grant`

### FOLLOW_UP_ARCHITECTURE_GAP

`HumanRequest` v2 has no first-class `tool_id` / `rule_ids`. IMPL-1 uses `AgentDecision.payload` + `context_artifacts` refs. Optional `HumanRequest` v3 fields are follow-up FU-001, not blocking.

## Resume and re-evaluation semantics

| Question | Answer |
|----------|--------|
| `ToolExecutionRequest` persisted? | Yes — runtime checkpoint / step cursor + `idempotency_key` |
| Step rerun on resume? | Yes — node `PENDING` + `governance_pause` |
| Duplicate execution prevention? | `idempotency_key` + side-effect ledger; grant single-use |
| Policy re-evaluated on resume? | **Yes** |
| Infinite re-pause avoidance? | Grant satisfies scoped `REQUIRE_HITL` once; cleared after successful invoke |
| Skip policy after resume? | **Forbidden** except typed grant for `REQUIRE_HITL` only |

## Failure / timeout semantics

| Event | Behavior |
|-------|----------|
| APPROVE | Canonical resume (existing intake + graph re-entry) |
| REJECT | `handle_human_rejection` -> `FAILED`; handler calls = 0 |
| ESCALATE | Existing escalation router |
| TIMEOUT | `HumanTimeoutCoordinator` + `LongRunningScheduler` |

## Dependency-direction proof

`declarative_enforcer` and `invoker` have no Task/HITL imports. Bridge imports contracts + error only. Tool callers catch and emit signal. UAEP/graph runners consume existing `GovernanceResolution` path. No edge from `runtime/policy` or `runtime/tools` to `HumanPauseCoordinator`.

## Security consequences

- Handler never runs before approval.
- Grant scoped; cannot approve different tool/rule set.
- `DENY` never overridden.
- `AUDIT_ONLY` never pauses for `REQUIRE_HITL`.
- No secrets in grant/trace.

## Observability / audit correlation

Policy eval (`declarative_policy_evaluation`) -> HITL signal (`declarative_policy_hitl_required`) -> pause (`HUMAN_APPROVAL_REQUESTED`) -> approval -> grant satisfied -> tool invocation end.

Keys: `run_id`, `task_id`, `step_id`, `tool_id`, `human_request_id`, `pause_id`, `matched_rule_ids`.

## Migration / compatibility

- `ToolInvokerProtocol` unchanged.
- `catalog_dispatch` must stop mapping HITL to `FAILED`.
- `AUDIT_ONLY` hosts: no pause behavior change.

## Implementation scope for IMPL-1

### Production allowlist

- `intergrax/contracts/declarative_hitl.py` (NEW)
- `intergrax/runtime/nexus/tools/declarative_policy_hitl_bridge.py` (NEW)
- `intergrax/runtime/nexus/tools/tool_loop.py`
- `intergrax/runtime/nexus/tools/catalog_dispatch.py`
- `intergrax/runtime/nexus/tools/plan_context_invocation.py`
- `intergrax/runtime/nexus/tools/tool_invocation_pattern.py`
- `intergrax/agents/persistence/declarative_tool_executor.py`
- `intergrax/agents/persistence/catalog_declarative_invoker.py`
- `intergrax/runtime/kernel/step_kernel.py`
- `intergrax/agents/authoring/acp_uaep_shim.py`
- `intergrax/runtime/policy/rules/evaluation.py`
- `intergrax/runtime/policy/declarative_enforcer.py`
- `intergrax/runtime/nexus/execution/graph_executor.py`
- `intergrax/runtime/task/task_contract.py`
- `intergrax/runtime/human/pause.py`

### Test allowlist

- `tests/unit/runtime/nexus/tools/test_declarative_policy_hitl_bridge.py` (NEW)
- `tests/unit/runtime/policy/test_declarative_policy_e2e.py`
- `tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py` (NEW)

## E2E acceptance contract

**ENFORCE** + `REQUIRE_HITL` on test tool.

1. First invoke: handler calls = 0; `WAITING_FOR_HUMAN`; checkpoint.
2. Approve via existing API; resume; grant satisfies HITL; handler calls = 1; task completes.
3. REJECT: handler calls = 0.
4. ESCALATE: canonical escalation path.
5. AUDIT_ONLY: no pause for declarative `REQUIRE_HITL` alone.

## Rejected alternatives

Option B, Option C, second pause coordinator, HITL as `TOOL_ERROR`, global `human_approved` as sole grant, `ExecutionInterrupt` for declarative policy.

## Open follow-ups

| ID | Gap |
|----|-----|
| FU-001 | `HumanRequest` v3 tool/rule scope fields |
| FU-002 | `ToolResponseStatus` HITL channel design detail |
| FU-003 | Grant scope in `SQLiteHumanDecisionStore` |

## Status

**Proposed / Ready for Review**
