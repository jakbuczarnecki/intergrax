# ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION: Agent governance approval vs canonical Execution HITL (UCA-6C-ADR3)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture); **R1** sequential pause + typed grant (2026-09-23) |
| **Date** | 2026-09-23 |
| **Deciders** | Execution Engine / UCA-6C architecture |
| **Related** | [ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION](../2026-09-22/ADR-UCA-6C-EXECUTION-CONTINUATION-INTEGRATION.md) Â· [ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY](../2026-09-22/ADR-UCA-6C-DURABLE-SUSPENDED-OPERATION-REENTRY.md) Â· [ADR-HARNESS-003](../2026-09-22/ADR-HARNESS-003.md) Â· [ADR-GR-10-002](../2026-09-19/ADR-GR-10-002.md) |

## 1. Context

UCA-6C removed legacy **TIGAE** (tool-invocation governance approval evidence) and AW/UCA pre-derived approval scopes. The canonical tool path now relies on **RuntimeToolInvoker** governance ordering plus **Execution-owned** continuation (`ExecutionContinuationPort`, `SuspendedExecutionOperationStore`, `ContinuationAwareCatalogToolHost`).

Independent audit at `3401d0074a825218f0202dc4c90e770fb9bb6af3` reported **PARTIAL PASS â€” ARCHITECTURAL DECISION REQUIRED**: **Agent Runtime Governance** can emit `ToolGovernanceApprovalRequiredError` for HIGH/CRITICAL tool invocations **before** declarative policy `REQUIRE_HITL` is evaluated, so the invocation never reaches the existing declarative HITL bridge. There is no typed, durable, execution-bound pause/re-entry for **Agent Governance authority** on the execution-bound catalog path.

This ADR selects one canonical model. **No production implementation** in this change set (R6-R5 gates only).

## 2. Current runtime ordering

Verified in `RuntimeToolInvoker._prepare_invocation` and `invoke` at `3401d0074a825218f0202dc4c90e770fb9bb6af3`.

### 2.1 `_prepare_invocation` (per physical tool attempt)

| Step | Boundary |
|------|----------|
| 1 | Registry lookup + contract bind |
| 2 | **Canonical inner governance** (`_require_canonical_inner_execution_guard`) |
| 3 | **Agent Runtime Governance** (`_require_agent_runtime_governance` â†’ `authorize_tool`) |
| 4 | **Scope policy** (`_scope_policy.is_allowed`) |
| 5 | **Sandbox isolation** (`require_sandbox_isolation` when `contract.requires_sandbox_isolation`) |
| 6 | **Legacy MSE configuration gate** (`require_meaningful_side_effect_authorization` â€” fails closed when MSE port missing in production) |
| 7 | **Declarative policy** (`declarative_enforcer.evaluate_tool_invocation` â†’ `DeclarativePolicyHitlRequiredError` on `REQUIRE_HITL`) |
| 8 | **Canonical MSE** (`_require_canonical_meaningful_side_effect_authorization` â†’ may raise `ToolGovernanceApprovalRequiredError` **with** `governed_continuation_request`) |
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
  â†’ build_tool_authorization_request (four-ID + tool/step/idempotency)
  â†’ AgentRuntimeGovernancePipeline.evaluate
  â†’ HighRiskApprovalPolicyProvider (or equivalent) â†’ REQUIRE_APPROVAL
  â†’ AgentRuntimeGovernanceBoundary.authorize_tool
  â†’ ToolGovernanceApprovalRequiredError
  â†’ RuntimeToolInvoker traces and re-raises (no bridge)
```

On **execution-bound L3** (`ContinuationAwareCatalogToolHost`), only `DeclarativePolicyHitlRequiredError` is translated to durable pause. **Agent Governance** errors propagate (see `test_uca6c_r5_r2_strict_governance_composition`: MSE `calls == 0`).

### 3.1 `governed_continuation_request` on Agent Governance path

| Field | Agent Governance (HIGH/CRITICAL) | Canonical MSE HITL |
|-------|----------------------------------|--------------------|
| Present on error? | **Absent** (`authorization_boundary` does not set it) | **Present** (`gate.governed_continuation_request`) |
| Creator | N/A | `mse_hitl_effect_gate` + MSE boundary |
| EE bridge | None today | `raise_mse_governed_continuation_hitl_pause` when `catalog_dispatch` catches error |

### 3.2 Correlation on governance request today

`build_tool_authorization_request` supplies typed **TaskId, RunId, AttemptId, ExecutionId** (when active execution context is bound). **Gap:** `approval_evidence_ref` is populated from **declarative** grant/`dhr_*` scope only â€” not from an Agent Governance grant. Policy providers treat any non-empty `approval_evidence_ref` as satisfied (`high_risk_approval_evidence_present`) without typed grant verification â€” **unsafe for production** once declarative grants could satisfy Agent Governance (must be fixed in R6-R5).

## 4. Existing authorities

| Authority | Owner | Signal | Human decision belongs to |
|-----------|-------|--------|---------------------------|
| **A. Agent Runtime Governance** | `runtime/agent_governance/**` | `ToolGovernanceApprovalRequiredError` (no continuation payload) | Agent Runtime Governance |
| **B. Declarative policy** | `runtime/policy/**` | `DeclarativePolicyHitlRequiredError` | Declarative policy rules |
| **C. Meaningful side effect (canonical MSE)** | MSE port + `mse_hitl_effect_gate` | `ToolGovernanceApprovalRequiredError` + `governed_continuation_request` | MSE / side-effect authorization |
| **D. Execution continuation** | `ExecutionContinuationPort` | Lifecycle only (`WAITING_FOR_HUMAN` â†’ `RESUME_AUTHORIZED` â†’ `RESUMED`) | **Not** an authority |

## 5. Existing continuation / HITL mechanisms

### 5.1 Declarative HITL (reference)

```text
DeclarativePolicyHitlRequiredError
  â†’ raise_hitl_pause_from_tool_invocation
  â†’ DeclarativePolicyHitlSignal (dhr_* invocation_scope_id)
  â†’ DeclarativeHitlPendingApproval
  â†’ DeclarativePolicyHitlPauseRequired
  â†’ (L3) compose_governed_continuation_from_declarative_hitl_pause
  â†’ SuspendedExecutionOperationStore.prepare/block
  â†’ establish_canonical_hitl_pause â†’ ExecutionContinuationPort
  â†’ ExecutionSuspendedWorkPauseRequired
```

Authority evidence: `DeclarativeHitlPendingApproval` / `DeclarativeHitlApprovalGrant` (matched_rule_ids, policy provenance). Lifecycle correlation: `GovernedContinuationRequest` + `continuation_id`. Resumable work: `SuspendedExecutionOperationDescriptor` (execution-bound catalog payload).

### 5.2 MSE governed continuation (reference)

```text
ToolGovernanceApprovalRequiredError (governed_continuation_request set)
  â†’ raise_mse_governed_continuation_hitl_pause
  â†’ GovernedContinuationHitlPauseRequired
  â†’ Task WAITING_FOR_HUMAN + governed continuation pause projection
```

Legal because MSE attaches **typed** `GovernedContinuationRequest` with side-effect scope correlation; grant consumption is via `GovernedContinuationApprovalGrant` on task state and `mse_hitl_effect_gate` â€” separate from declarative grants.

### 5.3 Legacy `ApprovalRequest` store

`AgentRuntimeApprovalBoundary` + `InMemoryApprovalStore` create `ApprovalRequest` records when `REQUIRE_APPROVAL` is evaluated. This is **not** wired to `ExecutionContinuationPort`, not invocation-durable across EE re-entry, and **not** the canonical human pause model. R6-R5 must **not** treat `approval_id` from pipeline reason strings as execution resume evidence.

## 6. Rejected shortcuts

| Shortcut | Verdict |
|----------|---------|
| Restore TIGAE / AW pre-approval transport | **Rejected** â€” ARCH-R1 REMOVE |
| Worker / CodeCraft / UCA minting approval | **Rejected** |
| Pre-derived `uca6c-scope:*` or dispatch-time scope | **Rejected** |
| `except ToolGovernanceApprovalRequiredError` â†’ declarative pending | **Rejected** â€” authority confusion |
| Move declarative policy before Agent Governance | **Rejected** â€” does not remove first blocker; weakens ordering intent |
| Generic `approval_evidence_ref` string without typed grant | **Rejected** |
| Universal approval megacontract | **Rejected** â€” HARNESS-003 |
| Second continuation or suspended-work store | **Rejected** |
| `DEFER_TO_HITL` without enforcement | **Rejected** â€” bypass risk |
| Local `except` swallow in RuntimeToolInvoker | **Rejected** â€” EE host must materialize pause |
| CodeCraft / Worker replay instead of suspended-work reentry | **Rejected** |

## 7. Options considered

### Option A â€” Authority-specific governed continuation (Agent Governance)

Agent Governance produces **typed** pending/grant + `GovernedContinuationRequest` with **Agent-specific** `operation_id` / correlation; EE L3 bridge mirrors declarative/MSE patterns.

**Fits** HARNESS-003, ADR1/ADR2, security invariants. **Chosen** (with Option C envelope).

### Option B â€” Map to declarative HITL

Reuse `DeclarativeHitlPendingApproval` / `DeclarativeHitlApprovalGrant`.

**Rejected:** `matched_rule_ids` and policy provenance are **declarative authority evidence**. They cannot legally authorize `HighRiskApprovalPolicyProvider`. `DeclarativeHitlApprovalGrant` must not satisfy Agent Governance (`approval_evidence_ref` rewiring required).

### Option C â€” Common human continuation envelope

Shared: `HumanRequest` projection, `ExecutionContinuationPort`, `SuspendedExecutionOperationStore`, `establish_canonical_hitl_pause`, execution identity.

Separate: authority pending/grant types and verification functions.

**Accepted** as the integration shape; combines with Option A contracts.

### Option D â€” Reorder governance (declarative before Agent Governance)

**Rejected:** Agent Governance remains the capability/risk gate for HIGH/CRITICAL; declarative HITL cannot subsume that authority. Policy HITL approval â‰  Agent Governance approval.

### Option E â€” Agent Governance `DEFER_TO_HITL`

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

## 10. Contracts (define only â€” R6-R5 implementation)

New **immutable, authority-specific** contracts:

| Contract | Role |
|----------|------|
| `AgentGovernanceHumanApprovalRequirement` | Snapshot of exact `ToolAuthorizationRequest` + policy_results + `approval_id` at block time |
| `AgentGovernanceHumanApprovalPending` | Persisted pause record (four-ID, tool_id, step_id, agent_id, tenant_id, idempotency_key, **agent_governance_invocation_scope_id** `agr_*`, human_request_id, pause_id, policy_results digest) |
| `AgentGovernanceHumanApprovalGrant` | One-shot grant bound to pending; includes `grant_id`, scope id, expiry, decided_by; **not** interchangeable with `DeclarativeHitlApprovalGrant` or `GovernedContinuationApprovalGrant` |

Composition:

- `compose_governed_continuation_from_agent_governance_pause(...)` â†’ `GovernedContinuationRequest` with `ContinuationReason.SECURITY` (or enum extension `AGENT_RUNTIME_GOVERNANCE` in R6-R5) and `operation_id = agent_governance_invocation_scope_id`.
- `GovernedContinuationCorrelation` carries authority via `reason` + typed pending id â€” **no** universal L2 bag on `ExecutionBoundCatalogToolInvokeRequest`.

`ToolAuthorizationRequest.approval_evidence_ref` **cannot** satisfy `REQUIRE_APPROVAL` alone (R1 §27.5 / §29). Use `VerifiedAgentGovernanceHumanApproval` from Agent Governance-owned verifier; typed grant on Task/runtime governance state — not declarative leakage.

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
- `agent_governance_invocation_scope_id` (`agr_*`) â€” **do not** reuse `dhr_*` as Agent Governance evidence

Grant verification must reject sibling invocations.

## 13. Pause / resume sequence (single authority)

```text
initial invoke â†’ Agent Governance REQUIRE_APPROVAL
  â†’ EE L3 bridge â†’ pending + GovernedContinuationRequest
  â†’ suspended operation PREPARED â†’ BLOCKED
  â†’ ExecutionContinuation WAITING_FOR_HUMAN
  â†’ human APPROVE â†’ AgentGovernanceHumanApprovalGrant
  â†’ RESUME_AUTHORIZED â†’ execution-owned reentry (same descriptor)
  â†’ RuntimeToolInvoker full stack (fresh gates; grant consumed)
  â†’ idempotency (unchanged key) â†’ ToolExecutor
```

## 14. Multiple authority sequence

Default: **sequential, separate human decisions**:

```text
Agent Governance (agr_*) â†’ Declarative HITL (dhr_*) â†’ MSE HITL (GCG)
```

No approval collapsing without a future ADR proving one combined authority.

## 15. Restart semantics

Reload pending, grant, continuation, and suspended descriptor by four-ID + `continuation_id`. Reentry via suspended-work host only. Expired grant â†’ fail closed.

## 16. Multi-host semantics

Single-owner continuation transitions and one-shot grant consumption (existing port semantics).

## 17. Security invariants

- Agent Governance approval â‰  declarative ALLOW/HITL.
- Declarative HITL â‰  Agent Governance approval.
- MSE grant â‰  other authorities.
- No pre-approval outside exact admitted invocation.
- Nexus remains internal.

## 18. Compatibility

Frozen: execution identity, continuation lifecycle, ToolRuntime path, L2 catalog invoker shape.

## 19. Migration impact

Remove TIGAE and declarative grant as Agent Governance evidence; rewire EE L3 and task governance fields; unblocks strict UCA-6C tests.

## 20. Implementation gates for R6-R5

Superseded by **§31 (Amendment R1)** for gate numbering and dependencies. Summary:

| Gate | Deliverable |
|------|-------------|
| R6-R5.1 | Typed Agent Governance approval contracts + `VerifiedAgentGovernanceHumanApproval` |
| R6-R5.2 | `AgentGovernanceGrantVerifier` + durable grant boundary |
| R6-R5.3 | Sequential pause state machine (`authority_reblock_from_claimed`, `pause_generation`) |
| R6-R5.4 | Agent Governance EE L3 bridge |
| R6-R5.5 | Resume/reentry + grant restore |
| R6-R5.6 | Declarative/MSE sequential transitions |
| R6-R5.7–R6-R5.9 | Worker E2E, restart, multi-host/adversarial |

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

---

## Amendment R1 (UCA-6C-ADR3-R1) â€” Sequential pause lifecycle & typed grant verification

| Field | Value |
|-------|-------|
| **Amendment** | R1 |
| **Date** | 2026-09-23 |
| **Audited code baseline** | `350565a867fba05d7d1c858ec5660a2e3613d921` (store/reentry/L3/governance); prior ADR3 audit pin `c1dbe20367591691e0025ffb9245800352443f28` |
| **Scope** | Architecture hardening only â€” **no** R6-R5 production implementation in this amendment |

R1 closes the two ADR3 gaps: (1) sequential authority pauses on one exact logical invocation while descriptor may be `CLAIMED` during reentry, and (2) typed Agent Governance grant verification so a bare `approval_evidence_ref` string is never security proof.

### Code audit summary (frozen core unchanged)

**Suspended operation materialization state machine (as implemented in `SuspendedOperationBackingStore` / contract `SuspendedExecutionOperationStore`):**

| From | Operation | To | Notes |
|------|-----------|-----|-------|
| â€” | `prepare` | `PREPARED` | insert only |
| `PREPARED` | `block` | `BLOCKED` | CAS on revision; clears `claim_ownership` |
| `BLOCKED` | `claim` | `CLAIMED` | fence starts at `1` |
| `CLAIMED` | `claim` | â€” | `ALREADY_CLAIMED` |
| `CLAIMED` | `reclaim` (expired lease) | `CLAIMED` | new owner; fence `N+1` |
| `CLAIMED` | `mark_consumed` | `CONSUMED` | terminal success |
| `BLOCKED` / `CLAIMED` | `abandon` | `ABANDONED` | terminal deny/cancel |
| `CONSUMED` / `ABANDONED` | any | â€” | terminal |

**`CLAIMED â†’ BLOCKED` does not exist today.** `block()` accepts only `PREPARED`. Sequential pauses therefore require a **new store transition** (R6-R5.3) â€” not reinterpretation of `block()` or `reclaim()`.

**Reentry coordinator (`ExecutionSuspendedWorkReentryCoordinator.reenter_after_resume`):** requires continuation `RESUMED` â†’ `load_active_for_continuation` â†’ `claim` (`BLOCKED`â†’`CLAIMED`) â†’ reconstruct â†’ `catalog_host.invoke` â†’ on success `mark_consumed`. On tool failure/exception, descriptor stays `CLAIMED` (no automatic `abandon`). There is **no** branch for a second human pause while `CLAIMED`; a nested `DeclarativePolicyHitlRequiredError` during invoke would attempt **new** `prepare()` + new `continuation_id` in L3 â€” **duplicate logical work risk** unless R1 model is implemented.

**L3 host (`ContinuationAwareCatalogToolHost._materialize_pause`):** always `mint_suspended_operation_id()`, `prepare(PREPARED)`, new `continuation_id` from governed request, then `block`. It does **not** consult an existing reentry descriptor. During reentry this is the gap R1 addresses.

**Continuation (`ExecutionContinuationPort` / `execution_continuation.py`):** one record is **one episode**. `RESUMED` is terminal (`advance_continuation_lifecycle` rejects further commands). `execution_continuation_lifecycle_permits_successor_episode()` is true only for terminal states (including `RESUMED`). Sequential human pauses therefore use **successive continuation instances** (new `continuation_id` per pause generation), not one immortal continuation record.

**Agent Governance today:** `HighRiskApprovalPolicyProvider` / `FinancialApprovalPolicyProvider` treat non-empty `approval_evidence_ref` as `ALLOW` (`high_risk_approval_evidence_present`). `build_tool_authorization_request` sets `approval_evidence_ref` from **declarative** grant/scope only. `AgentRuntimeApprovalBoundary` + `InMemoryApprovalStore` are legacy parallel approval IDs â€” **not** execution-bound typed grants; R6-R5 migrates to Task governance + verifier (deprecate string-only path).

## 23. Sequential authority pause lifecycle

For one **exact logical invocation** (same admitted Execution, same tool input, same `idempotency_key`), authorities are evaluated **in frozen RuntimeToolInvoker order** across **one or more resume cycles**. Each authority that requires human input produces **pause generation** `G = 1, 2, â€¦` without creating a second active logical work item.

**Permanent invariant:**

```text
ACTIVE_SUSPENDED_OPERATION_COUNT(logical_invocation) <= 1
ACTIVE_LOGICAL_WORK_COUNT = 1
```

At most one descriptor (`suspended_operation_id`) per logical invocation; at most one materialization in `{BLOCKED, CLAIMED}` at any time (`load_active_for_continuation` already fails closed if >1 per continuation_id).

### 23.1 Chosen model: Option C â€” one logical operation + pause generations

| Model | Verdict |
|-------|---------|
| **A â€” same descriptor, re-block (`CLAIMEDâ†’BLOCKED`)** | **Selected mechanism** â€” requires new `authority_reblock` store API (not `block()` on `PREPARED`) |
| **B â€” new descriptor per authority pause** | **Rejected** â€” violates single logical work item; sibling descriptor races |
| **C â€” logical operation + `pause_generation`** | **Selected identity model** â€” one `suspended_operation_id`, monotonic `pause_generation`, payload unchanged |

Implementation shape (R6-R5.3):

- Add to descriptor (contract v2 or optional fields in v1 extension): `pause_generation: int (>=1)`, `logical_invocation_fingerprint: str` (digest of four-ID + tool_id + step_id + agent_id + tenant_id + idempotency_key + payload_digest).
- Add store method e.g. `authority_reblock_from_claimed(...)` **or** `transition_to_authority_pause(...)` with CAS on `materialization_revision`, **only** callable from EE L3/coordinator when propagating typed `ExecutionSuspendedWorkPauseRequired` during active reentry.

**Option A transition (normative):**

```text
CLAIMED(owner=A, fence=F)
  â†’ authority_reblock (CAS revision)
  â†’ BLOCKED(claim_ownership=None, pause_generation+=1, continuation_id=K+1, invocation_scope_id=new authority scope)
```

- **Who releases claim:** the **Execution-owned** pause materialization path (L3 bridge / coordinator hook) that handles the typed pause signal â€” not Worker, not reclaim-for-lease.
- **Ownership after reblock:** `None` (must re-`claim` on next resume).
- **Revision:** `materialization_revision += 1` on every successful mutation (same as today).
- **Fence:** invalidated on reblock; next `claim` mints **new lease** with `fence = 1` for the new generation (do not continue fence `F` across generations).
- **`reclaim`:** remains **only** for expired `CLAIMED` lease recovery on **same** generation â€” **not** for authority pause.

### 23.2 Stale host (generation change)

After reblock to generation `G+1`, host A holding `(owner=A, fence=F, revision=R_old)` **must** fail all terminal mutations:

- `mark_consumed` â†’ `STALE_REVISION` or `STALE_CLAIM`
- `abandon` with A's fence â†’ `STALE_CLAIM` if revision advanced
- backend invocation on stale claim is impossible (coordinator only consumes matching revision/fence)

Host A **must not** create a parallel descriptor or downgrade generation.

### 23.3 Descriptor terminalization

`CONSUMED` only after **all** authority gates pass **and** backend completes successfully (or idempotent replay reports completed effect) on the **final** reentry generation. Approving generation 1 Agent Governance does **not** consume the suspended operation.

### 23.4 Human deny

Any authority `REJECT` / deny â†’ `abandon` on the single descriptor (typed reason e.g. `HUMAN_DENIED` / authority-specific) â†’ `ABANDONED` terminal; no `BLOCKED` orphan. Active continuation episode moves to `REJECTED` / `CANCELLED` per existing port semantics.

### 23.5 Reentry coordinator behavior on second pause (target)

When `catalog_host.invoke` raises typed pause while descriptor is `CLAIMED`:

1. Coordinator **does not** `mark_consumed`.
2. Coordinator invokes EE pause materialization for the **new** authority (updates same descriptor via `authority_reblock_from_claimed` + new continuation episode).
3. Returns disposition `PAUSED_FOR_AUTHORITY` (new enum in R6-R5) instead of `FAILED` where appropriate.

Today: **not implemented** â€” documented gap driving R6-R5.3/R6-R5.6.

## 24. Logical invocation vs pause generation model

| Concept | Definition | Stability |
|---------|------------|-----------|
| **Logical invocation** | One `ToolExecutionRequest` admitted under one four-ID Execution | Fixed for entire flow |
| **Logical invocation fingerprint** | Digest over four-ID, tool_id, step_id, agent_id, tenant_id, idempotency_key, canonical payload digest | Fixed |
| **Suspended operation identity** | `suspended_operation_id` (durable key) | One per logical invocation |
| **Pause generation** | `G` â€” increments on each distinct authority human pause | Monotonic |
| **Authority scope** | `agr_*`, `dhr_*`, MSE governed correlation | One active pending per generation |
| **Continuation instance** | `continuation_id` per episode | **New** `continuation_id` each generation |
| **Materialization state** | PREPARED/BLOCKED/CLAIMED/â€¦ | One active BLOCKED/CLAIMED row per descriptor |

Authority pause generation **does not** change logical invocation fingerprint or payload bytes.

## 25. Descriptor / revision / fencing semantics

| Event | `materialization_revision` | `pause_generation` | `claim_ownership` | `fence` |
|-------|---------------------------|-------------------|-------------------|---------|
| Initial `prepare`+`block` (gen 1) | 0â†’1 block | 1 | None | â€” |
| Resume `claim` | +1 | 1 | owner, lease | 1 |
| Reblock for gen 2 pause | +1 | 2 | None | â€” |
| Resume gen 2 `claim` | +1 | 2 | owner, lease | 1 (reset) |
| `mark_consumed` (final success) | +1 | final | None | â€” |

CAS: every mutator requires `expected_materialization_revision` matching loaded snapshot.

## 26. Continuation instance semantics

- **Mechanism:** single `ExecutionContinuationPort` (unchanged).
- **Instances:** **one new continuation episode per pause generation** because `RESUMED` is terminal and `request_pause` creates a new `continuation_id` (via governed `continuation_request_id`).
- **Correlation:** each episode's `governed_correlation.operation_id` matches the authority scope for that generation (`agr_*`, `dhr_*`, MSE scope).
- **Resume:** reentry coordinator always keys off the **current** generation's `continuation_id` (stored on descriptor after reblock).
- **Not** a second continuation store; **not** one continuation record reused across generations.

## 27. Typed Agent Governance grant verification

### 27.1 Owner

**Agent Runtime Governance** owns `AgentGovernanceGrantVerifier` (name normative; implementation R6-R5.2) â€” not policy providers individually, not Worker/UCA/CodeCraft, not Nexus generic code.

### 27.2 Pipeline order (target)

```text
ToolAuthorizationRequest (approval_evidence_ref MUST NOT satisfy REQUIRE_APPROVAL alone)
  â†’ AgentGovernanceGrantVerifier.verify(...)  [when resuming or when grant handle present on Task/RuntimeState]
       â†’ VerifiedAgentGovernanceHumanApproval (authority-specific; no universal VerifiedApproval)
  â†’ AgentRuntimePolicyEngine.evaluate with VerifiedAgentGovernanceHumanApproval context
  â†’ ALLOW | REQUIRE_APPROVAL | DENY
```

Policy providers receive **verified context** (Option A) or pipeline pre-marks Agent Governance approval satisfied **only after** verifier success (Option B). Providers **must not** read raw `approval_evidence_ref` for ALLOW.

### 27.3 Verifier inputs (minimum)

Grant type `AgentGovernanceHumanApprovalGrant`; authority = Agent Governance; `grant_id`; `agent_governance_invocation_scope_id`; four-ID; tenant_id; agent_id; tool_id; step_id; idempotency_key; expiry; pending linkage; consumption state; policy provenance/digest at pending time (when required).

### 27.4 Cross-authority isolation

| Grant | Agent Governance verifier | Declarative | MSE |
|-------|--------------------------|-------------|-----|
| `AgentGovernanceHumanApprovalGrant` | YES | NO | NO |
| `DeclarativeHitlApprovalGrant` | NO | YES | NO |
| `GovernedContinuationApprovalGrant` | NO | NO | YES |

### 27.5 `approval_evidence_ref` disposition (Â§37)

**Rule C (hardening):** field may remain for **non-security metadata / audit correlation only**. It **cannot** satisfy `REQUIRE_APPROVAL` without `VerifiedAgentGovernanceHumanApproval`. **Invariant:** `non-empty string != approval`. Remove declarative leakage from `build_tool_authorization_request` for Agent Governance path (R6-R5).

## 28. Grant consumption / crash semantics

Separate concerns: **approval grant consumption** â‰  **effect idempotency** (pre-effect idempotency store unchanged).

| Phase | Agent Governance grant |
|-------|------------------------|
| Human APPROVE | Grant persisted on Task governance state; pending marked resolved |
| Resume / reentry start | Verifier **validates**; may **reserve** grant (store-mediated CAS) |
| Fresh governance ALLOW for that generation | Grant **consumed** (one-shot) atomically with reservation |
| Backend success | Suspended op `CONSUMED` (unchanged) |

**Crash window (`verified â†’ crash before backend`):** consumption must occur **no earlier than** successful fresh Agent Governance ALLOW for that generation, and **no later than** first irreversible authorization of backend execution for that attempt. If consumed at ALLOW and crash before backend: **same** grant generation must not authorize a **different** invocation; recovery relies on **same** suspended descriptor + **effect idempotency** for replay. If verify reserved but not consumed: restart retries verify. Multi-host: single winner via durable grant store CAS (`consume_grant(grant_id, expected_revision)`).

Expired / already consumed / wrong scope / sibling step â†’ fail closed, require new human approval.

Policy drift: fresh evaluation mandatory; digest mismatch on pending â†’ grant invalid, new approval required (enterprise default).

## 29. `approval_evidence_ref` â€” final rule (R1)

See Â§27.5. Implementers **must not** implement `if approval_evidence_ref: ALLOW` in Agent Governance policy providers after R6-R5.2.

## 30. Restart / multi-host security

**Restart at generation G BLOCKED:** reload descriptor by `suspended_operation_id` or logical fingerprint index; read `pause_generation`, `continuation_id`, authority pending on Task; resume via `ExecutionContinuationPort` for **that** `continuation_id` only.

**Multi-host:** one active `CLAIMED` owner; reblock clears claim; stale fences rejected; grant consumption atomic in durable store.

## 31. Revised R6-R5 implementation gates

| Gate | Deliverable |
|------|-------------|
| **R6-R5.1** | `AgentGovernanceHumanApprovalRequirement`, `Pending`, `Grant`; `VerifiedAgentGovernanceHumanApproval` |
| **R6-R5.2** | `AgentGovernanceGrantVerifier` + durable grant store; strip string-trust from providers |
| **R6-R5.3** | `authority_reblock_from_claimed` + descriptor `pause_generation` / fingerprint; invariant tests |
| **R6-R5.4** | Agent Governance EE L3 bridge (`compose_governed_continuation_from_agent_governance_pause`) |
| **R6-R5.5** | Resume/reentry + grant restore + verifier before policy |
| **R6-R5.6** | Declarative/MSE sequential transitions via reblock (no second `prepare`) |
| **R6-R5.7** | Worker E2E execution-bound HIGH-risk |
| **R6-R5.8** | True restart across generations |
| **R6-R5.9** | Multi-host / adversarial stale host + double consume |

### 31.1 Test matrix (prescriptive)

**Grant security:** arbitrary string; declarative/MSE grant on Agent verifier; wrong `agr_*`; wrong execution; sibling step; expired; consumed.

**Sequential pauses:** Agentâ†’Declarative; Agentâ†’MSE; Agentâ†’Declarativeâ†’MSE; deny at gen 2; restart at gen 2 BLOCKED; stale host gen 1 cannot `mark_consumed`.

### 31.2 Multi-generation invariants (proof obligations)

```text
one logical invocation Â· one active suspended operation Â· one current pause generation
Â· one active continuation instance for that generation Â· one current authority pending
Â· one valid claim owner (or BLOCKED awaiting claim)
```

## 32. R1 mini-audit checklist

| Check | R1 |
|-------|-----|
| One owner per concern | PASS |
| Typed / no string authority | PASS (target; code gap documented) |
| No bypass / no second store | PASS |
| No duplicate logical work | PASS (Option C + reblock) |
| Fail closed | PASS |
| Restart / multi-host | PASS (specified) |
| Cross-authority isolated | PASS |
| Frozen core unchanged | PASS â€” **no** TaskId/RunId/AttemptId/ExecutionId/continuation port lifecycle change; **additive** store transition only |

## 33. Worked example â€” Agent â†’ Declarative â†’ MSE â†’ backend

| Gen | Authority | Materialization | Rev | Owner | Fence | Continuation |
|-----|-----------|-----------------|-----|-------|-------|--------------|
| 1 | Agent Governance | BLOCKED | 1 | â€” | â€” | `cont-agr-1` |
| 1 | (resume) | CLAIMED | 2 | host-1 | 1 | `cont-agr-1` |
| â€” | reblock | BLOCKED | 3 | â€” | â€” | `cont-dhr-2` |
| 2 | Declarative | (same) | 3 | â€” | â€” | `cont-dhr-2` |
| 2 | (resume) | CLAIMED | 4 | host-2 | 1 | `cont-dhr-2` |
| â€” | reblock | BLOCKED | 5 | â€” | â€” | `cont-mse-3` |
| 3 | MSE | (same) | 5 | â€” | â€” | `cont-mse-3` |
| 3 | (resume) | CLAIMED | 6 | host-2 | 1 | `cont-mse-3` |
| â€” | backend OK | CONSUMED | 7 | â€” | â€” | `cont-mse-3` terminal |

(Single `suspended_operation_id` throughout; `pause_generation` 1â†’2â†’3.)

---

*End of Amendment R1.*
