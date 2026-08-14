# ADR-PLATFORM-PLUGIN-001: Declarative Policy REQUIRE_HITL → Canonical Nexus HITL Bridge

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-08-14 |
| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 (+ REVIEW-FIX-1) |
| **Deciders** | Platform / Nexus runtime |
| **Related** | [`PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md`](../../../../maintainers/plans/PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md) BLOCK B · ENTERPRISE-4 review-fix `7154d29d` · [`ADR-POLICY-SIDE-EFFECT-001`](../2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md) · [`ADR-GOVERNED-CONTINUATION-001`](../2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) |

## Context

ENTERPRISE-4 (BLOCK B / CAND-007) wires typed `DeclarativePolicyEnforcer` at `RuntimeToolInvoker.invoke()` for tool invocations. `DENY` and `AUDIT_ONLY` behave correctly. `REQUIRE_HITL` in `ENFORCE` mode raises `DeclarativePolicyHitlRequiredError` before the tool handler executes — correct synchronous block, but **not** canonical HITL.

Canonical Nexus HITL already exists as a single system:

`AgentDecision` / `ExecutionInterruptHandler` → `GovernanceResolution` → `AgentExecutionStatus.NEEDS_INPUT` → `NexusGraphRunner._handle_needs_input` → `HumanPauseCoordinator.apply_pause` → `TaskState.WAITING_FOR_HUMAN` → checkpoint → `NexusIntakeRunner` resume/reject/escalate.

This ADR freezes how declarative `REQUIRE_HITL` joins that chain without a second pause coordinator, queue, or Tier-3 callback injection.

**REVIEW-FIX-1** closes three decisions required before IMPL-1:

1. authoritative pre-approval invocation scope persistence (`DeclarativeHitlPendingApproval`),
2. mandatory `invocation_scope_id` when `ToolExecutionRequest.idempotency_key` is `None`,
3. typed transport of an approved grant into policy re-evaluation.

## Current architecture

### Standard production execution chain (Nexus graph path)

| Step | Owner | Input → Output |
|------|-------|----------------|
| 1 | `NexusGraphRunner.run` | `Task` + `NexusPlan` + `ExecutionGraph` → `GraphPhaseOutcome` |
| 2 | `GraphExecutor.execute` / `_execute_node` | node → `AgentExecutionResult` list |
| 3 | `GraphExecutor.execute_fn` | builds `RuntimeRequest`, calls `AgentEngine.run_agent_with_result` |
| 4 | `AgentEngine.run_agent_with_result` | `RuntimeRequest` → `AgentExecutionResult` via `runtime_answer_to_agent_result(governance=...)` |
| 5 | `UAEPExecutor` (`intergrax/agents/uaep.py`) | per-step: execute step → `AgentDecision` → `ExecutionInterruptHandler.resolve_decision` → `GovernanceResolution` |
| 6 | `runtime_mapping.runtime_answer_to_agent_result` | `GovernanceResolution.should_pause` → `AgentExecutionStatus.NEEDS_INPUT` + attaches `agent_decision`, `human_request`, `execution_interrupt` |
| 7 | `GraphExecutor._execute_node` | `NEEDS_INPUT` → node `PENDING`, `metadata["governance_pause"]=True` |
| 8 | `NexusGraphRunner._handle_needs_input` | last `AgentExecutionResult` → `HumanPauseCoordinator.apply_pause` → `TaskState.WAITING_FOR_HUMAN` → `maybe_checkpoint` |
| 9 | `NexusIntakeRunner.run` (on resume) | `HumanPauseCoordinator.verdict_from_task` → approve/reject/escalate branches |
| 10 | `GraphExecutor.execute_fn` (resume) | reads `task.runtime.governance.declarative_hitl_grant` → `RuntimeRequest.declarative_hitl_grant` |

### Tool invocation sub-paths (all converge on `RuntimeToolInvoker`)

| Path | Caller of `RuntimeToolInvoker.invoke` | Current HITL behavior |
|------|--------------------------------------|------------------------|
| LLM tool loop | `tool_loop._invoke_planned_call` | Exception propagates uncaught |
| Catalog gateway | `catalog_dispatch.invoke_catalog_tool_request` | `except Exception` → `ToolResponseStatus.FAILED` (incorrect for HITL) |
| ACP declarative | `CatalogDeclarativeToolInvoker` → `catalog_dispatch` | Same swallow as gateway |
| Step kernel declarative | `execute_declarative_actions` → `DeclarativeToolInvoker` | No `hitl_required` status; failures map to `TOOL_FAILED` |

### Canonical HITL contract (verified)

`ExecutionInterruptHandler.resolve_decision()` returns `GovernanceResolution`:

- `should_pause == True` when `agent_decision.type == REQUEST_HUMAN` **or** `policy_decision.action == REQUIRE_HUMAN`.
- Creates `HumanRequest` when missing and pause is required.
- `HumanPauseCoordinator` owns persisted pause state (`TaskGovernanceState.pause_record`, `human_request`, `paused`).

**Consumer:** Nexus orchestration consumes `GovernanceResolution` **indirectly** via `AgentExecutionResult` produced by `runtime_answer_to_agent_result`. `NexusGraphRunner` checks `AgentExecutionStatus.NEEDS_INPUT`, not `GovernanceResolution` directly.

**`ExecutionInterrupt`:** Used for `AgentDecisionType.INTERRUPT` only. Declarative `REQUIRE_HITL` is **not** an interrupt; `GovernanceResolution.interrupt` remains `None` for this bridge.

### Meaningful-side-effect precedent

`MeaningfulSideEffectAuthorizationBoundary.authorize()` returns `MeaningfulSideEffectAuthorizationResult` with `requires_governed_continuation: bool` when `PolicyAction.REQUIRE_HUMAN`. It does **not** call Nexus HITL directly; higher layers surface `GovernedContinuationRequest` and compose with `ExecutionInterruptHandler`.

**Reuse:** typed **control outcome** at a low boundary + **orchestration-owned** conversion to `AgentDecision` / governed continuation. **Do not** duplicate `GovernedContinuationRequest` for catalog tools; use existing UAEP → `GovernanceResolution` path.

**Shortcoming:** meaningful-side-effect path is collaborative-work scoped; it does not define tool `idempotency_key` / step checkpoint semantics. This ADR extends the Nexus tool path instead.

### Evidence: why `idempotency_key` cannot be sole invocation identity

`ToolExecutionRequest` (`intergrax/tools/execution_models.py`):

```python
idempotency_key: Optional[str] = None
```

`run_id` + `step_id` + `tool_id` identify a UAEP step tool slot but **not** repeated invocations within the same step when `idempotency_key` is absent. No existing canonical invocation ID provides per-raise stability across checkpoint/resume without a dedicated scope identifier. Therefore `invocation_scope_id` is mandatory (see §Invocation identity).

## Problem

`DeclarativePolicyHitlRequiredError` blocks execution (handler call count = 0) but:

1. `catalog_dispatch` maps it to generic tool failure.
2. No `AgentDecision.REQUEST_HUMAN` / `HumanRequest` / `WAITING_FOR_HUMAN` transition occurs.
3. No typed approval scope for safe resume + policy re-evaluation.
4. No authoritative pending-approval DTO on `TaskGovernanceState`.
5. No typed grant transport to `DeclarativePolicyEnforcer`.

## Invariants

- **One canonical HITL system** — `HumanPauseCoordinator` + existing intake/graph runners only.
- `NEW_DYNAMIC_ATTRIBUTE_WIRING = 0` — no `getattr`/`setattr`/magic exception-name routing; explicit typed contracts only.
- `runtime/policy` and `runtime/tools` **must not** import `Task`, `NexusHitlRunner`, or `HumanPauseCoordinator`.
- HITL is **not** `TOOL_ERROR` / `ToolExecutionResult.fail`.
- `AUDIT_ONLY` + `REQUIRE_HITL` must never pause (`enforced=False` → no block, no exception).
- `DENY` precedence is absolute; human approval never bypasses `DENY`.
- **`HumanRequest` is presentation/workflow data only** — not the security source of truth for invocation scope.
- **`AgentDecision.payload` / `HumanRequest.context_artifacts` / generic task metadata** may mirror scope for display/audit but **must not** be authoritative authorization state.

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

**Rejected.** Breaks `ToolInvokerProtocol`; suspension mistaken for failure.

### Option C — Move declarative HITL evaluation above `RuntimeToolInvoker`

**Rejected.** Splits enforcement plane; weak handler call-count guarantee.

## Decision

**Choose Option A.**

### Frozen decisions

| # | Decision |
|---|----------|
| 1 | **Canonical bridge owner (translation):** `intergrax/runtime/nexus/tools/declarative_policy_hitl_bridge.py` — sole module converting `DeclarativePolicyHitlRequiredError` + invocation context → `DeclarativePolicyHitlSignal` → `AgentDecision` + optional pre-built `HumanRequest`. |
| 2 | **Canonical bridge owner (orchestration consumption):** existing `UAEPExecutor` step loop + `runtime_answer_to_agent_result` + `NexusGraphRunner._handle_needs_input` — no new pause owner. |
| 3 | **Low-level HITL signal type:** keep `DeclarativePolicyHitlRequiredError` (sync unwind). Add `DeclarativePolicyHitlSignal` (frozen dataclass) as the cross-layer typed payload extracted by the bridge. |
| 4 | **`DeclarativePolicyHitlRequiredError` remains?** **Yes** — internal synchronous signal inside `RuntimeToolInvoker` only. |
| 5 | **Translation to governance:** `DeclarativePolicyHitlSignal` → `AgentDecision(type=REQUEST_HUMAN, ...)` → `ExecutionInterruptHandler.resolve_decision()` → `GovernanceResolution` → `runtime_answer_to_agent_result` → `NEEDS_INPUT`. |
| 6 | **`ExecutionInterrupt` involved?** **No.** |
| 7 | **`HumanRequest` owner:** bridge builds minimal `HumanRequest` for UI/workflow; `ExecutionInterruptHandler` may enrich defaults. **Not** authoritative scope storage. |
| 8 | **Pause owner:** `HumanPauseCoordinator` (unchanged). |
| 9 | **Pending approval owner:** `TaskGovernanceState.declarative_hitl_pending` — authoritative pre-approval scope; persisted by `HumanPauseCoordinator.apply_pause` before checkpoint completion. |
| 10 | **Grant owner:** `TaskGovernanceState.declarative_hitl_grant` — single-use approval artifact; created on canonical APPROVE; consumed per §Grant consumption. |
| 11 | **Invocation identity:** mandatory `invocation_scope_id` on pending approval and grant; stable across checkpoint/resume (see §Invocation identity). |
| 12 | **Resume behavior:** graph node stays `PENDING`; step re-executes from checkpoint; same scoped `ToolExecutionRequest` reconstructed; handler call count 0 before approval, exactly 1 after. |
| 13 | **Policy re-evaluation:** always on resume; `REQUIRE_HITL` satisfied only when typed `DeclarativeHitlApprovalGrant` matches per §Grant matching. |
| 14 | **`DENY` after resume:** always blocks; grant never applies to `DENY`. |
| 15 | **Grant transport:** typed field chain only (see §Typed grant transport) — **no** `request.metadata["declarative_hitl_grant"]`. |
| 16 | **Trace IDs:** `invocation_scope_id`, `run_id`, `task_id`, `step_id`, `tool_id`, `human_request_id`, `pause_id`, `matched_rule_ids`, `grant_id` (no tool args/secrets). |
| 17 | **IMPL-1 file allowlist:** see Implementation scope section. |

## Typed contracts (`intergrax/contracts/declarative_hitl.py`)

All contracts are frozen (`frozen=True` dataclass or Pydantic `extra="forbid"`). IMPL-1 implements exactly these fields.

### `DeclarativePolicyHitlSignal`

Bridge transport from tool boundary to orchestration. Not persisted as authoritative state.

| Field | Type | Notes |
|-------|------|-------|
| `invocation_scope_id` | `str` | Generated once at bridge (see §Invocation identity) |
| `task_id` | `str` | Owning task |
| `run_id` | `str` | From `ToolExecutionRequest.run_id` / `RuntimeState.run_id` |
| `step_id` | `str` | From `ToolExecutionRequest.step_id` |
| `tool_id` | `str` | From request |
| `agent_id` | `str` | Evaluating agent |
| `idempotency_key` | `str \| None` | From request when present |
| `matched_rule_ids` | `tuple[str, ...]` | From policy decision |
| `policy_provenance_digest` | `str \| None` | From `PolicyEnforcementDecision.provenance_digest` |
| `reasons` | `tuple[str, ...]` | Audit copy |

### `DeclarativeHitlPendingApproval`

Authoritative authorization scope for a paused `REQUIRE_HITL` invocation **before** human APPROVE.

| Field | Type | Notes |
|-------|------|-------|
| `invocation_scope_id` | `str` | **Mandatory** correlation key |
| `task_id` | `str` | |
| `run_id` | `str` | |
| `step_id` | `str` | |
| `tool_id` | `str` | |
| `idempotency_key` | `str \| None` | Additional match constraint when non-null |
| `matched_rule_ids` | `tuple[str, ...]` | Exact set frozen at pause |
| `human_request_id` | `str` | Linked `HumanRequest.request_id` |
| `policy_provenance_digest` | `str \| None` | Prevents stale approval after policy bundle change |
| `agent_id` | `str` | Correlation / audit |
| `pause_id` | `str` | From `TaskPauseRecord.pause_id` once pause record exists |
| `created_at` | `str` | ISO-8601 UTC |

**Storage:** `TaskGovernanceState.declarative_hitl_pending: Optional[DeclarativeHitlPendingApproval] = None`

**Producer:** `DeclarativePolicyHitlBridge.build_pending_approval(signal, task, pause_record)` builds the DTO.

**Consumer:** `HumanPauseCoordinator.apply_pause` assigns `task.runtime.governance.declarative_hitl_pending` **before** `task.sync_metadata()` and checkpoint. Bridge passes pending via `AgentExecutionResult` typed attachment field `declarative_hitl_pending` (IMPL-1) or equivalent single hop — coordinator **must** copy to `TaskGovernanceState`; payload mirrors only.

**Clearing:** On APPROVE (after grant creation), REJECT, ESCALATE (terminal), TIMEOUT, or task terminal failure — `declarative_hitl_pending = None`.

### `DeclarativeHitlApprovalGrant`

Single-use approval artifact created **only** from persisted `DeclarativeHitlPendingApproval` after canonical APPROVE. **Never** reconstructed from user response text or `HumanRequest.context_artifacts`.

| Field | Type | Notes |
|-------|------|-------|
| `grant_id` | `str` | `grant_{uuid4().hex[:16]}` — audit identity |
| `invocation_scope_id` | `str` | Copied from pending |
| `task_id` | `str` | |
| `run_id` | `str` | |
| `step_id` | `str` | |
| `tool_id` | `str` | |
| `idempotency_key` | `str \| None` | Copied from pending |
| `matched_rule_ids` | `tuple[str, ...]` | Copied from pending |
| `human_request_id` | `str` | Originating approval identity |
| `policy_provenance_digest` | `str \| None` | Copied from pending |
| `pause_id` | `str` | From pending |
| `approved_at` | `str` | ISO-8601 UTC |

**Storage:** `TaskGovernanceState.declarative_hitl_grant: Optional[DeclarativeHitlApprovalGrant] = None`

**Producer:** `DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)` in `intergrax/runtime/human/declarative_hitl_grant.py` (IMPL-1), invoked from `NexusIntakeRunner.run` immediately after canonical APPROVE verdict validation and **before** `HumanPauseCoordinator.clear_pause`.

**Lifecycle:**

```text
pending approval
    ↓ APPROVE (canonical verdict, not response-text parsing)
approval grant (declarative_hitl_grant set, declarative_hitl_pending cleared)
    ↓ successful matching invocation (§Grant consumption)
consumed / cleared (declarative_hitl_grant = None)

REJECT / TIMEOUT / terminal ESCALATE:
    declarative_hitl_pending cleared
    no grant created
```

### `DeclarativeHitlDecisionPayload`

Optional mirror on `AgentDecision.payload` for UAEP diagnostics. `extra="forbid"`. **Not** security source of truth.

## Invocation identity

| Question | Frozen answer |
|----------|---------------|
| Can `idempotency_key` be sole identity? | **No** — optional on `ToolExecutionRequest`. |
| Mandatory correlation field | `invocation_scope_id: str` on pending approval and grant. |
| When generated | Once when bridge converts `DeclarativePolicyHitlRequiredError` → `DeclarativePolicyHitlSignal` (first REQUIRE_HITL raise for this invocation). |
| Format | `dhr_{uuid4().hex}` — new identifier; no existing step/invocation ID guarantees per-invocation uniqueness when `idempotency_key` is absent. |
| Stability | Copied unchanged: signal → pending approval → grant → resumed evaluation context. Survives checkpoint via `TaskGovernanceState` persistence. |
| `idempotency_key` role | Additional match dimension when non-null on both request and grant; never assumed present. |

## Typed grant transport

**Forbidden:** `request.metadata["declarative_hitl_grant"]`, untyped dict passthrough, or string-key task metadata as canonical cross-layer mechanism.

**Frozen typed path:**

```text
TaskGovernanceState.declarative_hitl_grant
    ↓ GraphExecutor.execute_fn (resume path only)
RuntimeRequest.declarative_hitl_grant          # NEW optional typed field on RuntimeRequest
    ↓ AgentEngine.run_agent_with_result → RuntimeState construction
RuntimeState.declarative_hitl_grant            # NEW optional typed field on RuntimeState
    ↓ RuntimeToolInvoker.invoke(state, request)
PolicyEvaluationContext.approval_grant         # NEW optional field
    ↓ DeclarativePolicyEnforcer.evaluate_tool_invocation
```

| Hop | Owner module | Field |
|-----|--------------|-------|
| Persisted grant | `task_contract.py` | `TaskGovernanceState.declarative_hitl_grant` |
| Resume attach | `graph_executor.py` | `RuntimeRequest.declarative_hitl_grant` ← from task governance |
| Runtime mirror | `runtime_state.py` | `RuntimeState.declarative_hitl_grant` ← from request at engine init |
| Evaluation input | `evaluation.py` | `PolicyEvaluationContext.approval_grant` |
| Consumer | `declarative_enforcer.py` | reads `context.approval_grant` for REQUIRE_HITL satisfaction |

`human_approved` metadata flag remains for existing UAEP step-skip semantics only; it **does not** satisfy declarative policy and **does not** replace the grant.

## Grant matching

A grant satisfies `REQUIRE_HITL` only when **all** predicates pass:

| Dimension | Rule |
|-----------|------|
| `invocation_scope_id` | Exact string equality |
| `task_id` | Exact equality with task at evaluation boundary |
| `run_id` | Exact equality with `ToolExecutionRequest.run_id` |
| `step_id` | Exact equality with `ToolExecutionRequest.step_id` |
| `tool_id` | Exact equality with `ToolExecutionRequest.tool_id` |
| `matched_rule_ids` | Set equality (order-independent) with current decision's matched rules |
| `human_request_id` | Exact equality with originating approval |
| `idempotency_key` | If non-null on **request**, must equal grant value; if null on request, grant field ignored for match |
| `policy_provenance_digest` | If non-null on **current** `PolicyEnforcementDecision`, must equal grant value; mismatch → grant does **not** satisfy (treat as stale approval) |

| Policy action | Grant behavior |
|---------------|----------------|
| `DENY` | **Never** grant-satisfiable — evaluated first; grant ignored |
| `REQUIRE_HITL` | Grant may satisfy when predicate passes |
| `ALLOW` | Unchanged — grant irrelevant |

Mismatch → `DeclarativePolicyHitlRequiredError` (re-pause) or orchestration failure per existing HITL path; grant not consumed.

## Grant consumption

| Event | Grant behavior |
|-------|----------------|
| Policy eval: DENY | Grant **not** consumed; invocation blocked |
| Policy eval: REQUIRE_HITL + mismatch | Grant **not** consumed |
| Policy eval: REQUIRE_HITL + match | Grant **consumed immediately** before handler invocation begins (`TaskGovernanceState.declarative_hitl_grant = None` + `sync_metadata`) — single-use |
| Handler throws / `ToolExecutionResult.success=False` | Grant already consumed; retry requires new HITL cycle |
| Invocation fails before handler (registry/scope error after grant consumption) | Grant already consumed — no reuse |
| Graph retries step without new approval | No grant → REQUIRE_HITL blocks again |
| Policy bundle changes (digest mismatch) | Grant does not satisfy; not consumed unless match attempted with stale grant |
| Tool request changes (tool_id/step_id/scope_id) | No match; grant not consumed if evaluation never accepts it |
| APPROVE | Pending → grant; pending cleared |
| REJECT | Pending cleared; no grant |
| TIMEOUT | Pending cleared; no grant |
| ESCALATE (terminal) | Pending cleared; no grant |

**Security balance:** consumption at grant acceptance (pre-handler) prevents persistent reusable approval. Handler failure after consumption requires re-approval — aligns with handler call-count invariant and existing idempotency ledger when `idempotency_key` is set.

**Clearing owner (IMPL-1):** `DeclarativeHitlGrantCoordinator.consume_grant(task)` called from `RuntimeToolInvoker` immediately after enforcer accepts grant, before `ToolExecutor` call.

## HumanRequest role

| Artifact | Role |
|----------|------|
| `HumanRequest` | Presentation, workflow, timeout, notification — **not** authoritative authorization scope |
| `HumanRequest.context_artifacts` | May reference/display tool, rules, scope for operators |
| `DeclarativeHitlPendingApproval` | **Authoritative** pre-approval scope on `TaskGovernanceState` |
| `DeclarativeHitlApprovalGrant` | **Authoritative** post-approval scope for one evaluation |
| `AgentDecision.payload` | Diagnostic mirror (`DeclarativeHitlDecisionPayload`) — audit/display only |

FU-001 (`HumanRequest` v3 first-class tool/rule fields) remains **non-blocking** for IMPL-1.

## Detailed control flow

### First invocation (ENFORCE + REQUIRE_HITL)

```text
RuntimeToolInvoker.invoke(request, state)
  → DeclarativePolicyEnforcer.evaluate_tool_invocation(context)
  → should_block_execution && action==REQUIRE_HITL
  → raise DeclarativePolicyHitlRequiredError(...)
       handler call count = 0

tool_loop | catalog_dispatch | execute_declarative_actions
  → catch DeclarativePolicyHitlRequiredError
  → DeclarativePolicyHitlBridge.to_signal(exc, request, task_context)
       generates invocation_scope_id (once)
  → DeclarativePolicyHitlBridge.build_pending_approval(signal, ...)
  → propagate signal / pause_hitl / to_agent_decision

UAEP path:
  → AgentDecision.REQUEST_HUMAN (+ optional payload mirror)
  → ExecutionInterruptHandler.resolve_decision
  → GovernanceResolution(should_pause=True)
  → runtime_answer_to_agent_result → NEEDS_INPUT
  → AgentExecutionResult.declarative_hitl_pending = pending DTO

NexusGraphRunner._handle_needs_input
  → HumanPauseCoordinator.apply_pause
       task.runtime.governance.declarative_hitl_pending = pending
       task.runtime.governance.human_request = HumanRequest (workflow)
  → TaskState.WAITING_FOR_HUMAN
  → maybe_checkpoint
```

### Human APPROVE

```text
NexusIntakeRunner.run
  → verdict == APPROVE (canonical HumanResponseVerdict, not free-text)
  → DeclarativeHitlGrantCoordinator.create_grant_from_pending(task)
       reads declarative_hitl_pending
       sets declarative_hitl_grant
       clears declarative_hitl_pending
  → HumanPauseCoordinator.clear_pause
  → HUMAN_APPROVAL_RECEIVED event

GraphExecutor.execute_fn (resume)
  → RuntimeRequest.declarative_hitl_grant = task.runtime.governance.declarative_hitl_grant
  → request.metadata["human_approved"] = True  (UAEP skip only; not policy grant)

RuntimeToolInvoker.invoke (resume)
  → PolicyEvaluationContext(approval_grant=state.declarative_hitl_grant, ...)
  → DeclarativePolicyEnforcer: DENY still DENY
  → REQUIRE_HITL + matching grant → satisfied
  → DeclarativeHitlGrantCoordinator.consume_grant(task)
  → handler executes once
```

### REJECT / ESCALATE / TIMEOUT

```text
REJECT:
  → handle_human_rejection → FAILED
  → declarative_hitl_pending cleared; no grant; handler calls = 0

ESCALATE:
  → handle_human_escalation (existing router)
  → on terminal escalation: pending cleared; no grant

TIMEOUT:
  → HumanTimeoutCoordinator + LongRunningScheduler
  → pending cleared; no grant
```

## Resume and re-evaluation semantics

| Question | Answer |
|----------|--------|
| `ToolExecutionRequest` persisted? | Yes — runtime checkpoint / step cursor + `idempotency_key` when set |
| `invocation_scope_id` persisted? | Yes — `TaskGovernanceState.declarative_hitl_pending` then grant |
| Step rerun on resume? | Yes — node `PENDING` + `governance_pause` |
| Duplicate execution prevention? | Grant single-use + `idempotency_key` + side-effect ledger |
| Policy re-evaluated on resume? | **Yes** |
| Infinite re-pause avoidance? | Grant satisfies scoped `REQUIRE_HITL` once per approval |
| Skip policy after resume? | **Forbidden** except typed grant for `REQUIRE_HITL` only |

## Dependency-direction proof

`declarative_enforcer` and `invoker` have no Task/HITL imports. Bridge imports contracts + error only. Grant coordinator lives in `runtime/human/` (orchestration tier). UAEP/graph runners consume existing `GovernanceResolution` path. No edge from `runtime/policy` or `runtime/tools` to `HumanPauseCoordinator`.

## Security consequences

- Handler never runs before approval.
- Grant scoped to frozen pending approval; cannot approve different tool/rule set.
- `policy_provenance_digest` blocks stale policy reuse.
- `DENY` never overridden.
- `AUDIT_ONLY` never pauses for `REQUIRE_HITL`.
- No secrets in grant/trace.
- User response text never constructs grant.

## Observability / audit correlation

Policy eval (`declarative_policy_evaluation`) → HITL signal (`declarative_policy_hitl_required`) → pause (`HUMAN_APPROVAL_REQUESTED`) → approval → grant created (`declarative_hitl_grant_created`) → grant satisfied (`declarative_hitl_grant_consumed`) → tool invocation end.

Keys: `invocation_scope_id`, `grant_id`, `run_id`, `task_id`, `step_id`, `tool_id`, `human_request_id`, `pause_id`, `matched_rule_ids`.

## Migration / compatibility

- `ToolInvokerProtocol` unchanged.
- `catalog_dispatch` must stop mapping HITL to `FAILED`.
- `AUDIT_ONLY` hosts: no pause behavior change.
- New typed fields on `RuntimeRequest`, `RuntimeState`, `PolicyEvaluationContext`, `TaskGovernanceState` — no metadata key addition.

## Implementation scope for IMPL-1

### Production allowlist

- `intergrax/contracts/declarative_hitl.py` (NEW)
- `intergrax/runtime/human/declarative_hitl_grant.py` (NEW)
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
- `intergrax/runtime/nexus/tools/invoker.py`
- `intergrax/runtime/nexus/execution/graph_executor.py`
- `intergrax/runtime/nexus/responses/response_schema.py`
- `intergrax/runtime/nexus/engine/runtime_state.py`
- `intergrax/runtime/task/task_contract.py`
- `intergrax/runtime/human/pause.py`
- `intergrax/runtime/nexus/orchestration/intake_runner.py`
- `intergrax/contracts/agent_execution_result.py`

### Test allowlist

- `tests/unit/runtime/nexus/tools/test_declarative_policy_hitl_bridge.py` (NEW)
- `tests/unit/runtime/human/test_declarative_hitl_grant.py` (NEW)
- `tests/unit/runtime/policy/test_declarative_policy_e2e.py`
- `tests/integration/runtime/test_declarative_policy_hitl_nexus_e2e.py` (NEW)

## E2E acceptance contract

**ENFORCE** + `REQUIRE_HITL` on test tool.

1. First invoke: handler calls = 0; `WAITING_FOR_HUMAN`; `declarative_hitl_pending` persisted; checkpoint.
2. Approve via existing API; `declarative_hitl_grant` created from pending; resume; grant satisfies HITL; handler calls = 1; grant consumed; task completes.
3. REJECT: pending cleared; no grant; handler calls = 0.
4. ESCALATE: canonical escalation path; terminal escalation clears pending without grant.
5. AUDIT_ONLY: no pause for declarative `REQUIRE_HITL` alone.
6. Policy digest change after pause: grant does not satisfy without re-approval.

## Rejected alternatives

Option B, Option C, second pause coordinator, HITL as `TOOL_ERROR`, global `human_approved` as sole grant, `ExecutionInterrupt` for declarative policy, metadata-dict grant transport, reconstructing grant from `HumanRequest` or user response text, `idempotency_key` as sole invocation identity.

## Open follow-ups

| ID | Gap |
|----|-----|
| FU-001 | `HumanRequest` v3 tool/rule scope display fields (non-blocking) |
| FU-002 | `ToolResponseStatus` HITL channel design detail |
| FU-003 | Grant/pending persistence in `SQLiteHumanDecisionStore` for cross-process audit |

## Status

**Accepted** — pending approval DTO, grant DTO, invocation identity, typed transport, matching predicate, and consumption lifecycle frozen for IMPL-1.
