# ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION: Agent governance approval vs canonical Execution HITL (UCA-6C-ADR3)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture) |
| **Date** | 2026-09-23 |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | [ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION](../2026-09-22/ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) · [ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY](../2026-09-22/ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md) · [ADR-HARNESS-003](../2026-09-22/ADR-HARNESS-003.md) · [ADR-GR-10-002](../2026-09-19/ADR-GR-10-002.md) |

## 1. Context

UCA-6C removed legacy **TIGAE** (tool-invocation governance approval evidence) and AW/UCA pre-derived approval scopes. The canonical tool path now relies on **RuntimeToolInvoker** governance ordering plus **Execution-owned** continuation (`ExecutionContinuationPort`, `SuspendedExecutionOperationStore`, `ContinuationAwareCatalogToolHost`).

Independent audit at `3401d0074a825218f0202dc4c90e770fb9bb6af3` reported **PARTIAL PASS — ARCHITECTURAL DECISION REQUIRED**: **Agent Runtime Governance** can emit `ToolGovernanceApprovalRequiredError` for HIGH/CRITICAL tool invocations **before** declarative policy `REQUIRE_HITL` is evaluated, so the invocation never reaches the existing declarative HITL bridge. There is no typed, durable, execution-bound pause/re-entry for **Agent Governance authority** on the execution-bound catalog path.

This ADR selects one canonical model. **No production implementation** in this change set (R6-R5 gates only).

## 2. Current runtime ordering

Verified in `RuntimeToolInvoker._prepare_invocation` and `invoke` at `3401d0074a825218f0202dc4c90e770fb9bb6af3`.

### 2.1 `_prepare_invocation` (per physical tool attempt)

| Step | Boundary |
|------|----------|
| 1 | Registry lookup + contract bind |
| 2 | **Canonical inner governance** (`_require_canonical_inner_execution_guard`) |
| 3 | **Agent Runtime Governance** (`_require_agent_runtime_governance` → `authorize_tool`) |
| 4 | **Scope policy** (`_scope_policy.is_allowed`) |
| 5 | **Sandbox isolation** (`require_sandbox_isolation` when `contract.requires_sandbox_isolation`) |
| 6 | **Legacy MSE configuration gate** (`require_meaningful_side_effect_authorization` — fails closed when MSE port missing in production) |
| 7 | **Declarative policy** (`declarative_enforcer.evaluate_tool_invocation` → `DeclarativePolicyHitlRequiredError` on `REQUIRE_HITL`) |
| 8 | **Canonical MSE** (`_require_canonical_meaningful_side_effect_authorization` → may raise `ToolGovernanceApprovalRequiredError` **with** `governed_continuation_request`) |
| 9 | Input schema validation (after all governance gates above) |

### 2.2 `invoke` (after preparation)

| Step | Boundary |
|------|----------|
| 10 | Protected-work admission (when configured) |
| 11 | **Pre-effect idempotency** claim/replay (side-effect tools with `idempotency_key`) |
| 12 | **ToolExecutor** (`_execute_external_effect`) |

**Note:** Idempotency runs **after** the full governance stack in `_prepare_invocation`, immediately before backend execution. The idempotency key stays stable across sequential human pauses.

## 3. Exact blocking scenario

```text
ToolExecutionRequest
  → build_tool_authorization_request (four-ID + tool/step/idempotency)
  → AgentRuntimeGovernancePipeline.evaluate
  → HighRiskApprovalPolicyProvider (or equivalent) → REQUIRE_APPROVAL
  → AgentRuntimeGovernanceBoundary.authorize_tool
  → ToolGovernanceApprovalRequiredError
  → RuntimeToolInvoker traces and re-raises (no bridge)
```

On **execution-bound L3** (`ContinuationAwareCatalogToolHost`), only `DeclarativePolicyHitlRequiredError` is translated to durable pause. **Agent Governance** errors propagate (see `test_uca6c_r5_r2_strict_governance_composition`: MSE `calls == 0`).

### 3.1 `governed_continuation_request` on Agent Governance path

| Field | Agent Governance (HIGH/CRITICAL) | Canonical MSE HITL |
|-------|----------------------------------|--------------------|
| Present on error? | **Absent** (`authorization_boundary` does not set it) | **Present** (`gate.governed_continuation_request`) |
| Creator | N/A | `mse_hitl_effect_gate` + MSE boundary |
| EE bridge | None today | `raise_mse_governed_continuation_hitl_pause` when `catalog_dispatch` catches error |

### 3.2 Correlation on governance request today

`build_tool_authorization_request` supplies typed **TaskId, RunId, AttemptId, ExecutionId** (when active execution context is bound). **Gap:** `approval_evidence_ref` is populated from **declarative** grant/`dhr_*` scope only — not from an Agent Governance grant. Policy providers treat any non-empty `approval_evidence_ref` as satisfied (`high_risk_approval_evidence_present`) without typed grant verification — **unsafe for production** once declarative grants could satisfy Agent Governance (must be fixed in R6-R5).

## 4. Existing authorities

| Authority | Owner | Signal | Human decision belongs to |
|-----------|-------|--------|---------------------------|
| **A. Agent Runtime Governance** | `runtime/agent_governance/**` | `ToolGovernanceApprovalRequiredError` (no continuation payload) | Agent Runtime Governance |
| **B. Declarative policy** | `runtime/policy/**` | `DeclarativePolicyHitlRequiredError` | Declarative policy rules |
| **C. Meaningful side effect (canonical MSE)** | MSE port + `mse_hitl_effect_gate` | `ToolGovernanceApprovalRequiredError` + `governed_continuation_request` | MSE / side-effect authorization |
| **D. Execution continuation** | `ExecutionContinuationPort` | Lifecycle only (`WAITING_FOR_HUMAN` → `RESUME_AUTHORIZED` → `RESUMED`) | **Not** an authority |

## 5. Existing continuation / HITL mechanisms

### 5.1 Declarative HITL (reference)

```text
DeclarativePolicyHitlRequiredError
  → raise_hitl_pause_from_tool_invocation
  → DeclarativePolicyHitlSignal (dhr_* invocation_scope_id)
  → DeclarativeHitlPendingApproval
  → DeclarativePolicyHitlPauseRequired
  → (L3) compose_governed_continuation_from_declarative_hitl_pause
  → SuspendedExecutionOperationStore.prepare/block
  → establish_canonical_hitl_pause → ExecutionContinuationPort
  → ExecutionSuspendedWorkPauseRequired
```

Authority evidence: `DeclarativeHitlPendingApproval` / `DeclarativeHitlApprovalGrant` (matched_rule_ids, policy provenance). Lifecycle correlation: `GovernedContinuationRequest` + `continuation_id`. Resumable work: `SuspendedExecutionOperationDescriptor` (execution-bound catalog payload).

### 5.2 MSE governed continuation (reference)

```text
ToolGovernanceApprovalRequiredError (governed_continuation_request set)
  → raise_mse_governed_continuation_hitl_pause
  → GovernedContinuationHitlPauseRequired
  → Task WAITING_FOR_HUMAN + governed continuation pause projection
```

Legal because MSE attaches **typed** `GovernedContinuationRequest` with side-effect scope correlation; grant consumption is via `GovernedContinuationApprovalGrant` on task state and `mse_hitl_effect_gate` — separate from declarative grants.

### 5.3 Legacy `ApprovalRequest` store

`AgentRuntimeApprovalBoundary` + `InMemoryApprovalStore` create `ApprovalRequest` records when `REQUIRE_APPROVAL` is evaluated. This is **not** wired to `ExecutionContinuationPort`, not invocation-durable across EE re-entry, and **not** the canonical human pause model. R6-R5 must **not** treat `approval_id` from pipeline reason strings as execution resume evidence.

## 6. Rejected shortcuts

| Shortcut | Verdict |
|----------|---------|
| Restore TIGAE / AW pre-approval transport | **Rejected** — ARCH-R1 REMOVE |
| Worker / CodeCraft / UCA minting approval | **Rejected** |
| Pre-derived `uca6c-scope:*` or dispatch-time scope | **Rejected** |
| `except ToolGovernanceApprovalRequiredError` → declarative pending | **Rejected** — authority confusion |
| Move declarative policy before Agent Governance | **Rejected** — does not remove first blocker; weakens ordering intent |
| Generic `approval_evidence_ref` string without typed grant | **Rejected** |
| Universal approval megacontract | **Rejected** — HARNESS-003 |
| Second continuation or suspended-work store | **Rejected** |
| `DEFER_TO_HITL` without enforcement | **Rejected** — bypass risk |
| Local `except` swallow in RuntimeToolInvoker | **Rejected** — EE host must materialize pause |
| CodeCraft / Worker replay instead of suspended-work reentry | **Rejected** |

## 7. Options considered

### Option A — Authority-specific governed continuation (Agent Governance)

Agent Governance produces **typed** pending/grant + `GovernedContinuationRequest` with **Agent-specific** `operation_id` / correlation; EE L3 bridge mirrors declarative/MSE patterns.

**Fits** HARNESS-003, ADR1/ADR2, security invariants. **Chosen** (with Option C envelope).

### Option B — Map to declarative HITL

Reuse `DeclarativeHitlPendingApproval` / `DeclarativeHitlApprovalGrant`.

**Rejected:** `matched_rule_ids` and policy provenance are **declarative authority evidence**. They cannot legally authorize `HighRiskApprovalPolicyProvider`. `DeclarativeHitlApprovalGrant` must not satisfy Agent Governance (`approval_evidence_ref` rewiring required).

### Option C — Common human continuation envelope

Shared: `HumanRequest` projection, `ExecutionContinuationPort`, `SuspendedExecutionOperationStore`, `establish_canonical_hitl_pause`, execution identity.

Separate: authority pending/grant types and verification functions.

**Accepted** as the integration shape; combines with Option A contracts.

### Option D — Reorder governance (declarative before Agent Governance)

**Rejected:** Agent Governance remains the capability/risk gate for HIGH/CRITICAL; declarative HITL cannot subsume that authority. Policy HITL approval ≠ Agent Governance approval.

### Option E — Agent Governance `DEFER_TO_HITL`

**Rejected** unless a fail-closed deferral contract exists (none today). Default stance: bypass risk.

## 8. Decision

Adopt **authority-specific Agent Runtime Governance human approval artifacts** integrated through the **existing Execution-owned continuation and suspended-operation infrastructure** (Option A + C).

Agent Governance **evaluates** and **validates grants**; it does **not** own continuation lifecycle. RuntimeToolInvoker **emits** `ToolGovernanceApprovalRequiredError` unchanged; **EE L3** (`ContinuationAwareCatalogToolHost` and equivalent execution hosts) gain a **typed branch** for Agent Governance pause materialization (parallel to declarative, not generic exception swallowing).

## 9. Ownership model

| Concern | Owner |
|---------|-------|
| REQUIRE_APPROVAL decision | Agent Runtime Governance pipeline |
| Grant issuance & verification | Agent Runtime Governance (typed grant contract) |
| Pause lifecycle | `ExecutionContinuationPort` |
| Durable resumable invocation | `SuspendedExecutionOperationStore` + descriptor codec |
| Human prompt projection | `HumanRequest` (UI/lifecycle only) |
| Tool path enforcement | `RuntimeToolInvoker` (signals only) |
| Bridge / materialization | EE L3 orchestration (Nexus-internal, not public) |

## 10. Contracts (define only — R6-R5 implementation)

New **immutable, authority-specific** contracts:

| Contract | Role |
|----------|------|
| `AgentGovernanceHumanApprovalRequirement` | Snapshot of exact `ToolAuthorizationRequest` + policy_results + `approval_id` at block time |
| `AgentGovernanceHumanApprovalPending` | Persisted pause record (four-ID, tool_id, step_id, agent_id, tenant_id, idempotency_key, **agent_governance_invocation_scope_id** `agr_*`, human_request_id, pause_id, policy_results digest) |
| `AgentGovernanceHumanApprovalGrant` | One-shot grant bound to pending; includes `grant_id`, scope id, expiry, decided_by; **not** interchangeable with `DeclarativeHitlApprovalGrant` or `GovernedContinuationApprovalGrant` |

Composition:

- `compose_governed_continuation_from_agent_governance_pause(...)` → `GovernedContinuationRequest` with `ContinuationReason.SECURITY` (or enum extension `AGENT_RUNTIME_GOVERNANCE` in R6-R5) and `operation_id = agent_governance_invocation_scope_id`.
- `GovernedContinuationCorrelation` carries authority via `reason` + typed pending id — **no** universal L2 bag on `ExecutionBoundCatalogToolInvokeRequest`.

`ToolAuthorizationRequest.approval_evidence_ref` must accept **only** verified `AgentGovernanceHumanApprovalGrant.grant_id` (prefer typed grant on runtime/task governance state over declarative leakage).

## 11. Persistence model

| Artifact | Store |
|----------|-------|
| Continuation lifecycle | `ExecutionContinuationPort` backing store (existing) |
| Suspended invocation payload | `SuspendedExecutionOperationStore` (existing) |
| Agent Governance pending/grant | Task governance typed fields (parallel to `declarative_hitl_pending` / `declarative_hitl_grant`) |
| HumanRequest | Projected on task; not sole authority evidence |

Grants: **one-shot**, stale-safe consumption, survive process restart and host switch.

## 12. Invocation correlation

Minimum binding:

- `TaskId`, `RunId`, `AttemptId`, `ExecutionId`
- `tool_id`, `step_id`, `agent_id`, `tenant_id`
- `idempotency_key`
- `agent_governance_invocation_scope_id` (`agr_*`) — **do not** reuse `dhr_*` as Agent Governance evidence

Grant verification must reject sibling invocations.

## 13. Pause / resume sequence (single authority)

```text
initial invoke → Agent Governance REQUIRE_APPROVAL
  → EE L3 bridge → pending + GovernedContinuationRequest
  → suspended operation PREPARED → BLOCKED
  → ExecutionContinuation WAITING_FOR_HUMAN
  → human APPROVE → AgentGovernanceHumanApprovalGrant
  → RESUME_AUTHORIZED → execution-owned reentry (same descriptor)
  → RuntimeToolInvoker full stack (fresh gates; grant consumed)
  → idempotency (unchanged key) → ToolExecutor
```

## 14. Multiple authority sequence

Default: **sequential, separate human decisions**:

```text
Agent Governance (agr_*) → Declarative HITL (dhr_*) → MSE HITL (GCG)
```

No approval collapsing without a future ADR proving one combined authority.

## 15. Restart semantics

Reload pending, grant, continuation, and suspended descriptor by four-ID + `continuation_id`. Reentry via suspended-work host only. Expired grant → fail closed.

## 16. Multi-host semantics

Single-owner continuation transitions and one-shot grant consumption (existing port semantics).

## 17. Security invariants

- Agent Governance approval ≠ declarative ALLOW/HITL.
- Declarative HITL ≠ Agent Governance approval.
- MSE grant ≠ other authorities.
- No pre-approval outside exact admitted invocation.
- Nexus remains internal.

## 18. Compatibility

Frozen: execution identity, continuation lifecycle, ToolRuntime path, L2 catalog invoker shape.

## 19. Migration impact

Remove TIGAE and declarative grant as Agent Governance evidence; rewire EE L3 and task governance fields; unblocks strict UCA-6C tests.

## 20. Implementation gates for R6-R5

| Gate | Deliverable |
|------|-------------|
| R6-R5.1 | Requirement, pending, grant contracts |
| R6-R5.2 | Typed grant verification in policy engine |
| R6-R5.3 | Agent governance HITL bridge + L3 materialization |
| R6-R5.4 | Reentry grant propagation |
| R6-R5.5 | Sequential multi-authority pauses |
| R6-R5.6 | E2E execution-bound HIGH-risk |
| R6-R5.7 | Restart / adversarial / multi-host |

## 21. Test matrix (R6-R5)

Covers: initial HIGH-risk pause; approve/deny; wrong grant; sibling; expiry; policy drift; declarative+MSE sequencing; restart; multi-host.

## 22. Permanent architecture gates

Assert: single continuation + suspended-work store; no declarative grant satisfies Agent Governance; ordering frozen unless new ADR; no backend without bridge on execution-bound path.

## Compliance

- Tier boundaries preserved
- HARNESS-003: shared lifecycle, separate authority contracts
- ADR1/ADR2 remain authoritative for continuation + suspended work

## Consequences

### Positive

- Unblocks UCA-6C-R6-R5 without TIGAE
- Clear authority matrix

### Negative

- Additional typed contracts; possible multiple human prompts (by design)
