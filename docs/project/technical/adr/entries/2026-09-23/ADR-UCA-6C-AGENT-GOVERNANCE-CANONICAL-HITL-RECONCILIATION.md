# ADR-UCA-6C-AGENT-GOVERNANCE-CANONICAL-HITL-RECONCILIATION: Agent governance approval vs canonical Execution HITL (UCA-6C-ADR3)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture); **R1** sequential pause + typed grant; **R2** grant recovery + reblock freeze (2026-09-23) |
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

## Amendment R1 (UCA-6C-ADR3-R1) — Sequential pause lifecycle & typed grant verification

| Field | Value |
|-------|-------|
| **Amendment** | R1 |
| **Date** | 2026-09-23 |
| **Audited code baseline** | `350565a867fba05d7d1c858ec5660a2e3613d921` (store/reentry/L3/governance); prior ADR3 audit pin `c1dbe20367591691e0025ffb9245800352443f28` |
| **Scope** | Architecture hardening only — **no** R6-R5 production implementation in this amendment |

R1 closes the two ADR3 gaps: (1) sequential authority pauses on one exact logical invocation while descriptor may be `CLAIMED` during reentry, and (2) typed Agent Governance grant verification so a bare `approval_evidence_ref` string is never security proof.

### Code audit summary (frozen core unchanged)

**Suspended operation materialization state machine (as implemented in `SuspendedOperationBackingStore` / contract `SuspendedExecutionOperationStore`):**

| From | Operation | To | Notes |
|------|-----------|-----|-------|
| — | `prepare` | `PREPARED` | insert only |
| `PREPARED` | `block` | `BLOCKED` | CAS on revision; clears `claim_ownership` |
| `BLOCKED` | `claim` | `CLAIMED` | fence starts at `1` |
| `CLAIMED` | `claim` | — | `ALREADY_CLAIMED` |
| `CLAIMED` | `reclaim` (expired lease) | `CLAIMED` | new owner; fence `N+1` |
| `CLAIMED` | `mark_consumed` | `CONSUMED` | terminal success |
| `BLOCKED` / `CLAIMED` | `abandon` | `ABANDONED` | terminal deny/cancel |
| `CONSUMED` / `ABANDONED` | any | — | terminal |

**`CLAIMED → BLOCKED` does not exist today.** `block()` accepts only `PREPARED`. Sequential pauses therefore require a **new store transition** (R6-R5.3) — not reinterpretation of `block()` or `reclaim()`.

**Reentry coordinator (`ExecutionSuspendedWorkReentryCoordinator.reenter_after_resume`):** requires continuation `RESUMED` → `load_active_for_continuation` → `claim` (`BLOCKED`→`CLAIMED`) → reconstruct → `catalog_host.invoke` → on success `mark_consumed`. On tool failure/exception, descriptor stays `CLAIMED` (no automatic `abandon`). There is **no** branch for a second human pause while `CLAIMED`; a nested `DeclarativePolicyHitlRequiredError` during invoke would attempt **new** `prepare()` + new `continuation_id` in L3 — **duplicate logical work risk** unless R1 model is implemented.

**L3 host (`ContinuationAwareCatalogToolHost._materialize_pause`):** always `mint_suspended_operation_id()`, `prepare(PREPARED)`, new `continuation_id` from governed request, then `block`. It does **not** consult an existing reentry descriptor. During reentry this is the gap R1 addresses.

**Continuation (`ExecutionContinuationPort` / `execution_continuation.py`):** one record is **one episode**. `RESUMED` is terminal (`advance_continuation_lifecycle` rejects further commands). `execution_continuation_lifecycle_permits_successor_episode()` is true only for terminal states (including `RESUMED`). Sequential human pauses therefore use **successive continuation instances** (new `continuation_id` per pause generation), not one immortal continuation record.

**Agent Governance today:** `HighRiskApprovalPolicyProvider` / `FinancialApprovalPolicyProvider` treat non-empty `approval_evidence_ref` as `ALLOW` (`high_risk_approval_evidence_present`). `build_tool_authorization_request` sets `approval_evidence_ref` from **declarative** grant/scope only. `AgentRuntimeApprovalBoundary` + `InMemoryApprovalStore` are legacy parallel approval IDs — **not** execution-bound typed grants; R6-R5 migrates to Task governance + verifier (deprecate string-only path).

## 23. Sequential authority pause lifecycle

For one **exact logical invocation** (same admitted Execution, same tool input, same `idempotency_key`), authorities are evaluated **in frozen RuntimeToolInvoker order** across **one or more resume cycles**. Each authority that requires human input produces **pause generation** `G = 1, 2, …` without creating a second active logical work item.

**Permanent invariant:**

```text
ACTIVE_SUSPENDED_OPERATION_COUNT(logical_invocation) <= 1
ACTIVE_LOGICAL_WORK_COUNT = 1
```

At most one descriptor (`suspended_operation_id`) per logical invocation; at most one materialization in `{BLOCKED, CLAIMED}` at any time (`load_active_for_continuation` already fails closed if >1 per continuation_id).

### 23.1 Chosen model: Option C — one logical operation + pause generations

| Model | Verdict |
|-------|---------|
| **A — same descriptor, re-block (`CLAIMED→BLOCKED`)** | **Selected mechanism** — requires new `authority_reblock` store API (not `block()` on `PREPARED`) |
| **B — new descriptor per authority pause** | **Rejected** — violates single logical work item; sibling descriptor races |
| **C — logical operation + `pause_generation`** | **Selected identity model** — one `suspended_operation_id`, monotonic `pause_generation`, payload unchanged |

Implementation shape (R6-R5.3):

- Add to descriptor (contract v2 or optional fields in v1 extension): `pause_generation: int (>=1)`, `logical_invocation_fingerprint: str` (digest of four-ID + tool_id + step_id + agent_id + tenant_id + idempotency_key + payload_digest).
- Add store method e.g. `authority_reblock_from_claimed(...)` **or** `transition_to_authority_pause(...)` with CAS on `materialization_revision`, **only** callable from EE L3/coordinator when propagating typed `ExecutionSuspendedWorkPauseRequired` during active reentry.

**Option A transition (normative):**

```text
CLAIMED(owner=A, fence=F)
  → authority_reblock (CAS revision)
  → BLOCKED(claim_ownership=None, pause_generation+=1, continuation_id=K+1, invocation_scope_id=new authority scope)
```

- **Who releases claim:** the **Execution-owned** pause materialization path (L3 bridge / coordinator hook) that handles the typed pause signal — not Worker, not reclaim-for-lease.
- **Ownership after reblock:** `None` (must re-`claim` on next resume).
- **Revision:** `materialization_revision += 1` on every successful mutation (same as today).
- **Fence:** invalidated on reblock; next `claim` mints **new lease** with `fence = 1` for the new generation (do not continue fence `F` across generations).
- **`reclaim`:** remains **only** for expired `CLAIMED` lease recovery on **same** generation — **not** for authority pause.

### 23.2 Stale host (generation change)

After reblock to generation `G+1`, host A holding `(owner=A, fence=F, revision=R_old)` **must** fail all terminal mutations:

- `mark_consumed` → `STALE_REVISION` or `STALE_CLAIM`
- `abandon` with A's fence → `STALE_CLAIM` if revision advanced
- backend invocation on stale claim is impossible (coordinator only consumes matching revision/fence)

Host A **must not** create a parallel descriptor or downgrade generation.

### 23.3 Descriptor terminalization

`CONSUMED` only after **all** authority gates pass **and** backend completes successfully (or idempotent replay reports completed effect) on the **final** reentry generation. Approving generation 1 Agent Governance does **not** consume the suspended operation.

### 23.4 Human deny

Any authority `REJECT` / deny → `abandon` on the single descriptor (typed reason e.g. `HUMAN_DENIED` / authority-specific) → `ABANDONED` terminal; no `BLOCKED` orphan. Active continuation episode moves to `REJECTED` / `CANCELLED` per existing port semantics.

### 23.5 Reentry coordinator behavior on second pause (target)

When `catalog_host.invoke` raises typed pause while descriptor is `CLAIMED`:

1. Coordinator **does not** `mark_consumed`.
2. Coordinator invokes EE pause materialization for the **new** authority (updates same descriptor via `authority_reblock_from_claimed` + new continuation episode).
3. Returns disposition `PAUSED_FOR_AUTHORITY` (new enum in R6-R5) instead of `FAILED` where appropriate.

Today: **not implemented** — documented gap driving R6-R5.3/R6-R5.6.

## 24. Logical invocation vs pause generation model

| Concept | Definition | Stability |
|---------|------------|-----------|
| **Logical invocation** | One `ToolExecutionRequest` admitted under one four-ID Execution | Fixed for entire flow |
| **Logical invocation fingerprint** | Digest over four-ID, tool_id, step_id, agent_id, tenant_id, idempotency_key, canonical payload digest | Fixed |
| **Suspended operation identity** | `suspended_operation_id` (durable key) | One per logical invocation |
| **Pause generation** | `G` — increments on each distinct authority human pause | Monotonic |
| **Authority scope** | `agr_*`, `dhr_*`, MSE governed correlation | One active pending per generation |
| **Continuation instance** | `continuation_id` per episode | **New** `continuation_id` each generation |
| **Materialization state** | PREPARED/BLOCKED/CLAIMED/… | One active BLOCKED/CLAIMED row per descriptor |

Authority pause generation **does not** change logical invocation fingerprint or payload bytes.

## 25. Descriptor / revision / fencing semantics

| Event | `materialization_revision` | `pause_generation` | `claim_ownership` | `fence` |
|-------|---------------------------|-------------------|-------------------|---------|
| Initial `prepare`+`block` (gen 1) | 0→1 block | 1 | None | — |
| Resume `claim` | +1 | 1 | owner, lease | 1 |
| Reblock for gen 2 pause | +1 | 2 | None | — |
| Resume gen 2 `claim` | +1 | 2 | owner, lease | 1 (reset) |
| `mark_consumed` (final success) | +1 | final | None | — |

CAS: every mutator requires `expected_materialization_revision` matching loaded snapshot.

## 26. Continuation instance semantics

- **Mechanism:** single `ExecutionContinuationPort` (unchanged).
- **Instances:** **one new continuation episode per pause generation** because `RESUMED` is terminal and `request_pause` creates a new `continuation_id` (via governed `continuation_request_id`).
- **Correlation:** each episode's `governed_correlation.operation_id` matches the authority scope for that generation (`agr_*`, `dhr_*`, MSE scope).
- **Resume:** reentry coordinator always keys off the **current** generation's `continuation_id` (stored on descriptor after reblock).
- **Not** a second continuation store; **not** one continuation record reused across generations.

## 27. Typed Agent Governance grant verification

### 27.1 Owner

**Agent Runtime Governance** owns `AgentGovernanceGrantVerifier` (name normative; implementation R6-R5.2) — not policy providers individually, not Worker/UCA/CodeCraft, not Nexus generic code.

### 27.2 Pipeline order (target)

```text
ToolAuthorizationRequest (approval_evidence_ref MUST NOT satisfy REQUIRE_APPROVAL alone)
  → AgentGovernanceGrantVerifier.verify(...)  [when resuming or when grant handle present on Task/RuntimeState]
       → VerifiedAgentGovernanceHumanApproval (authority-specific; no universal VerifiedApproval)
  → AgentRuntimePolicyEngine.evaluate with VerifiedAgentGovernanceHumanApproval context
  → ALLOW | REQUIRE_APPROVAL | DENY
```

Policy providers receive **only** typed `verified_agent_governance_human_approval` on `AgentRuntimePolicyEvaluationContext` (R2.5). **Option B (pipeline pre-mark) is rejected.** Providers **must not** read raw `approval_evidence_ref` for ALLOW.

### 27.3 Verifier inputs (minimum)

Grant type `AgentGovernanceHumanApprovalGrant`; authority = Agent Governance; `grant_id`; `agent_governance_invocation_scope_id`; four-ID; tenant_id; agent_id; tool_id; step_id; idempotency_key; expiry; pending linkage; consumption state; policy provenance/digest at pending time (when required).

### 27.4 Cross-authority isolation

| Grant | Agent Governance verifier | Declarative | MSE |
|-------|--------------------------|-------------|-----|
| `AgentGovernanceHumanApprovalGrant` | YES | NO | NO |
| `DeclarativeHitlApprovalGrant` | NO | YES | NO |
| `GovernedContinuationApprovalGrant` | NO | NO | YES |

### 27.5 `approval_evidence_ref` disposition (§37)

**Rule C (hardening):** field may remain for **non-security metadata / audit correlation only**. It **cannot** satisfy `REQUIRE_APPROVAL` without `VerifiedAgentGovernanceHumanApproval`. **Invariant:** `non-empty string != approval`. Remove declarative leakage from `build_tool_authorization_request` for Agent Governance path (R6-R5).

## 28. Grant consumption / crash semantics

Separate concerns: **approval grant consumption** ≠ **effect idempotency** (pre-effect idempotency store unchanged).

| Phase | Agent Governance grant |
|-------|------------------------|
| Human APPROVE | Grant persisted on Task governance state; pending marked resolved |
| Resume / reentry start | Verifier **validates**; may **reserve** grant (store-mediated CAS) |
| Fresh governance ALLOW for that generation | Grant → **APPLIED** (R2.1); see Amendment R2 crash matrix |
| Backend success | Suspended op `CONSUMED` (unchanged) |

**Crash window:** superseded by **Amendment R2** (grant **APPLIED** semantics + crash matrix). Summary: same logical invocation may recover after ALLOW without new human approval; siblings remain blocked; effect idempotency handles backend dedupe only.

Expired / already consumed / wrong scope / sibling step → fail closed, require new human approval.

Policy drift: fresh evaluation mandatory; digest mismatch on pending → grant invalid, new approval required (enterprise default).

## 29. `approval_evidence_ref` — final rule (R1)

See §27.5. Implementers **must not** implement `if approval_evidence_ref: ALLOW` in Agent Governance policy providers after R6-R5.2.

## 30. Restart / multi-host security

**Restart at generation G BLOCKED:** reload descriptor by `suspended_operation_id` or logical fingerprint index; read `pause_generation`, `continuation_id`, authority pending on Task; resume via `ExecutionContinuationPort` for **that** `continuation_id` only.

**Multi-host:** one active `CLAIMED` owner; reblock clears claim; stale fences rejected; grant consumption atomic in durable store.

## 31. Revised R6-R5 implementation gates

| Gate | Deliverable |
|------|-------------|
| **R6-R5.1** | Agent Governance contracts + lifecycle states (AVAILABLE…TERMINAL) on canonical Task fields |
| **R6-R5.2** | Canonical durable grant/pending via AgentGovernanceGrantLifecyclePort + checkpoint CAS adapter (R2.4) |
| **R6-R5.3** | AgentGovernanceGrantVerifier + typed AgentRuntimePolicyEvaluationContext |
| **R6-R5.4** | authority_reblock_from_claimed + pause_generation / ClaimAuthority invariant tests |
| **R6-R5.5** | Agent Governance EE L3 pause bridge |
| **R6-R5.6** | Recovery + grant reservation/application + crash matrix behaviors |
| **R6-R5.7** | Sequential Declarative/MSE reblock (no second prepare) |
| **R6-R5.8** | Worker E2E |
| **R6-R5.9** | Restart + multi-host + adversarial |


### 31.1 Test matrix (prescriptive)

**Grant security:** arbitrary string; declarative/MSE grant on Agent verifier; wrong `agr_*`; wrong execution; sibling step; expired; consumed.

**Sequential pauses:** Agent→Declarative; Agent→MSE; Agent→Declarative→MSE; deny at gen 2; restart at gen 2 BLOCKED; stale host gen 1 cannot `mark_consumed`.

### 31.2 Multi-generation invariants (proof obligations)

```text
one logical invocation · one active suspended operation · one current pause generation
· one active continuation instance for that generation · one current authority pending
· one valid claim owner (or BLOCKED awaiting claim)
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
| Frozen core unchanged | PASS — **no** TaskId/RunId/AttemptId/ExecutionId/continuation port lifecycle change; **additive** store transition only |

## 33. Worked example — Agent → Declarative → MSE → backend

| Gen | Authority | Materialization | Rev | Owner | Fence | Continuation |
|-----|-----------|-----------------|-----|-------|-------|--------------|
| 1 | Agent Governance | BLOCKED | 1 | — | — | `cont-agr-1` |
| 1 | (resume) | CLAIMED | 2 | host-1 | 1 | `cont-agr-1` |
| — | reblock | BLOCKED | 3 | — | — | `cont-dhr-2` |
| 2 | Declarative | (same) | 3 | — | — | `cont-dhr-2` |
| 2 | (resume) | CLAIMED | 4 | host-2 | 1 | `cont-dhr-2` |
| — | reblock | BLOCKED | 5 | — | — | `cont-mse-3` |
| 3 | MSE | (same) | 5 | — | — | `cont-mse-3` |
| 3 | (resume) | CLAIMED | 6 | host-2 | 1 | `cont-mse-3` |
| — | backend OK | CONSUMED | 7 | — | — | `cont-mse-3` terminal |

(Single `suspended_operation_id` throughout; `pause_generation` 1→2→3.)

---

*End of Amendment R1.*
## Amendment R2 (UCA-6C-ADR3-R2) — Grant recovery semantics & reblock fencing freeze

| Field | Value |
|-------|-------|
| **Amendment** | R2 |
| **Date** | 2026-09-23 |
| **Scope** | Architecture freeze only — **no** R6-R5 production implementation in this amendment |
| **Supersedes (partial)** | R1 §27.2 integration fork; R1 §28 consumption wording; R1 §31 gate numbering |

R2 closes the four remaining ADR3 gaps: (1) crash after Agent Governance ALLOW with grant already applied, (2) normative `authority_reblock_from_claimed` owner + fence + generation contract, (3) single verified-grant integration model, (4) exactly one canonical source-of-truth for Agent Governance pending/grant.

### R2.1 Grant lifecycle (normative)

#### Semantic states

Agent Governance human approval grant lifecycle is **distinct** from suspended-operation materialization and **distinct** from effect idempotency.

| State | Meaning | Sibling logical invocation | Same logical invocation after crash |
|-------|---------|------------------------------|-------------------------------------|
| **AVAILABLE** | Human approval recorded; grant durable; not yet host-bound for this governance generation | May not use until reserved/applied per rules | May verify + reserve/apply per recovery |
| **RESERVED** | Durable CAS won; grant bound to exact linkage (below) for one in-flight governance attempt | **Rejected** (exclusive) | **May resume** if reservation lease valid or reclaim rules fire |
| **APPLIED** | Fresh Agent Governance evaluation returned **ALLOW** for this exact linkage; grant spent for this governance generation | **Rejected** | **May recover** authorization path without new human approval if linkage + policy still valid |
| **TERMINAL** | Grant cannot authorize any further Agent Governance generation (expiry, superseded requirement, descriptor terminal, explicit revoke) | **Rejected** | **Rejected** |

Names may map to repo enums in R6-R5; semantic partition above is frozen.

#### Reservation linkage (RESERVED)

`reserve` (CAS) must atomically record at minimum:

- `logical_invocation_fingerprint` (stable digest of admitted Execution + tool invocation identity)
- `pause_generation` (current Agent Governance generation on the single suspended descriptor)
- `agent_governance_invocation_scope_id` (`agr_*`)
- four-ID (`TaskId`, `RunId`, `AttemptId`, `ExecutionId` when bound)
- `grant_id` + pending generation / policy provenance digest at approval time

Reservation is **durable**, **multi-host exclusive**, **restart-visible** — not process-local. Reuse **`LeaseOwnership` field shape** (`owner_id`, `lease_expires_at`, `fence`) on the grant lifecycle sub-record for reservation/reclaim only; **do not** reuse suspended-operation `claim_ownership` state.

#### Application (APPLIED)

**Forbidden:** “consumed at ALLOW” without recovery semantics.

**Frozen:** transition to **APPLIED** occurs only when **fresh** `AgentRuntimePolicyEngine.evaluate` returns **ALLOW** for the current physical attempt, after `AgentGovernanceGrantVerifier` produced `VerifiedAgentGovernanceHumanApproval` valid for the exact linkage above.

APPLIED means:

- this grant generation cannot satisfy a **different** logical invocation or sibling step;
- the **same** logical invocation may **re-enter** Agent Governance after crash and, if grant still **AVAILABLE/RESERVED/APPLIED** per matrix below and policy still permits, **must not** require new human approval.

#### Approval authorization recovery ≠ effect idempotency recovery

| Subsystem | Owns |
|-----------|------|
| Agent Governance grant lifecycle | Reuse of human approval for **exact same** logical invocation / generation linkage |
| Effect idempotency store | Backend effect deduplication (`InvocationClaim` / existing semantics unchanged) |

Never satisfy backend replay by bypassing grant state; never treat grant APPLIED as proof backend completed.

#### Policy drift (mandatory fresh evaluation)

Grant proves **human approved this exact invocation requirement at approval time**. It does **not** prove current policy must ALLOW.

On every physical attempt: full Agent Governance evaluation runs. Outcomes:

- current policy **DENY** → **DENY** (grant irrelevant)
- **REQUIRE_APPROVAL** with **same** requirement + valid verified grant for linkage → may **ALLOW** via verified context
- **REQUIRE_APPROVAL** with **different** requirement digest / scope → grant **TERMINAL** for old requirement; new human approval
- **ALLOW** without human gate → proceed (grant not needed)

#### Multi-host grant CAS

```text
Host A: AVAILABLE → reserve(grant_id, expected_lifecycle_revision, linkage…) → RESERVED
Host B: simultaneous reserve → CONFLICT / STALE (fail closed)
```

**Host crash while RESERVED:** reservation carries `lease_expires_at`; after expiry, **reclaim** path (same pattern as suspended-operation `reclaim`, on **grant lifecycle** only) returns grant to **AVAILABLE** with monotonic reservation `fence` increment — never permanent lock. In-flight host with valid lease retains exclusive reserve.

#### Terminal grant

Grant becomes **TERMINAL** when any of:

- suspended descriptor **CONSUMED** or **ABANDONED** for the logical invocation
- approval expiry elapsed
- policy requirement superseded (digest / scope mismatch on fresh eval)
- human **DENY** or cancel (no grant; pending terminal)
- successful **TERMINAL** transition after governance completion for that approval episode (R6-R5 defines exact trigger = descriptor terminal or explicit revoke)

### R2.2 Crash matrix (mandatory)

| Crash point | Grant state | Backend / effect | Same logical invocation may resume? | Sibling may use grant? | Human reapproval? | Effect idempotency role |
|-------------|-------------|------------------|-------------------------------------|------------------------|-------------------|-------------------------|
| Before verify | AVAILABLE | not started | Yes — verify/reserve | No | No | none |
| After verify / reserve | RESERVED | not started | Yes — continue governance | No | No | none |
| After fresh ALLOW (APPLIED), before idempotency claim | APPLIED | not started | **Yes** — recovery per §R2.1 | No | No if linkage+policy valid | none yet |
| After idempotency claim, before backend start | APPLIED | claim held | Yes — same attempt path | No | No if linkage+policy valid | claim prevents duplicate effect |
| After backend started | APPLIED or TERMINAL | uncertain | Resume only via fenced claim + idempotency rules | No | No if linkage+policy valid | dedupe / replay |
| After backend success, before descriptor CONSUMED | APPLIED → terminalize | completed | Complete descriptor terminalization | No | No | replay returns prior effect |
| After descriptor CONSUMED | TERMINAL | completed | N/A (invocation complete) | No | N/A | replay only |

### R2.3 Audit matrix (mandatory)

| Scenario | Grant | Descriptor | Continuation | Backend |
|----------|-------|------------|--------------|---------|
| Approve normal | AVAILABLE → APPLIED → TERMINAL | BLOCKED→CLAIMED→…→CONSUMED | new episode per generation | once |
| Crash before reserve | AVAILABLE | unchanged | unchanged | none |
| Crash after reserve | RESERVED (or reclaimed AVAILABLE) | CLAIMED or BLOCKED per gen | current gen only | none |
| Crash after ALLOW before idempotency | APPLIED | CLAIMED (current gen) | current gen | none |
| Crash after idempotency before effect | APPLIED | CLAIMED | current gen | none / claimed |
| Crash after effect | TERMINAL path | toward CONSUMED | current gen | replay via idempotency |
| Sibling invocation | reject | unchanged | — | none |
| Policy drift | re-evaluate; maybe TERMINAL + new pending | per R1 reblock rules | maybe new pause | none until allowed |

### R2.4 Canonical grant source-of-truth (frozen)

**Decision: Option A — Task Governance persistence is the single canonical store** for Agent Governance pending and grant authority records.

Evidence (code at session HEAD): `TaskGovernanceState` already hosts authority-specific declarative pending/grant (`declarative_hitl_pending`, `declarative_hitl_grant`); durable task checkpoints expose **`checkpoint_revision` CAS** (`StaleCheckpointWriteError` in `SQLiteTaskCheckpointStore`). No separate parallel full-grant store may duplicate authority.

| Artifact | Role |
|----------|------|
| `TaskGovernanceState.agent_governance_hitl_pending` (R6-R5) | **Canonical pending** |
| `TaskGovernanceState.agent_governance_human_approval_grant` + embedded lifecycle revision (R6-R5) | **Canonical grant + AVAILABLE/RESERVED/APPLIED/TERMINAL** |
| `TaskCheckpointPersistence` + `expected_checkpoint_revision` | **CAS / multi-host task write serialization** for reserve/apply/terminalize |
| `HumanRequest` / runtime pause projections | **Projection only** — never proof of approval |
| `ToolAuthorizationRequest.approval_evidence_ref` | **Audit/correlation only** — see R2.6 |

**Forbidden:** full grant in Task governance **and** independent `AgentGrantStore` both acting as authority (`ONE_CANONICAL_AGENT_GRANT_STORE`).

R6-R5.2 implements **`AgentGovernanceGrantLifecyclePort`** (pluginable API) whose **default adapter** mutates canonical Task governance fields through checkpoint CAS — not a second source-of-truth.

Pending canonical location mirrors grant: **`agent_governance_hitl_pending` on TaskGovernanceState**; HumanRequest is not authority.

### R2.5 Verified-grant integration (single model — Option A closed)

R1 §27.2 **Option B is rejected.** Pipeline pre-mark without typed context is forbidden.

**Normative chain (only path):**

```text
AgentGovernanceHumanApprovalGrant (canonical on Task governance)
  → AgentGovernanceGrantVerifier.verify(...)
  → VerifiedAgentGovernanceHumanApproval (immutable; mint authority = verifier only)
  → AgentRuntimePolicyEvaluationContext.verified_agent_governance_human_approval
  → AgentRuntimePolicyEngine.evaluate(...)
  → policy providers (HighRisk, Financial, extensions)
```

Rules:

- `VerifiedAgentGovernanceHumanApproval` **cannot** be constructed by Worker, UCA, CodeCraft, or generic Nexus code.
- Providers **must** use `verified_agent_governance_human_approval` (or repo-equivalent typed field) — **not** `approval_evidence_ref`, generic dict, or metadata bags.
- External plugin providers may inspect verified context **without** grant store or verifier imports.

Post R6-R5, **forbidden:**

```text
if request.approval_evidence_ref:
    ALLOW
```

Required:

```text
if verified_agent_governance_approval satisfies this request linkage:
    ...
```

Migrate existing `HighRiskApprovalPolicyProvider` / `FinancialApprovalPolicyProvider` string-trust (verified at `3401d007…` / HEAD) in R6-R5.3.

### R2.6 `approval_evidence_ref` (final)

```text
AUDIT / CORRELATION METADATA ONLY — NOT SECURITY EVIDENCE
```

Remove declarative grant leakage into Agent `build_tool_authorization_request` during R6-R5.

### R2.7 `authority_reblock_from_claimed` fencing contract

New store API (R6-R5.3) — name frozen: **`authority_reblock_from_claimed`**.

#### Forbidden

Revision-only reblock:

```text
authority_reblock_from_claimed(id, expected_revision)  # REJECTED
```

#### Required inputs (minimum)

| Parameter | Purpose |
|-----------|---------|
| `suspended_operation_id` | Entity key |
| `expected_materialization_revision` | CAS |
| `expected_pause_generation` | Generation fence |
| `expected_owner_id` | Current claim owner |
| `expected_fence` | Current claim fence |
| `next_pause_generation` | Must equal `expected_pause_generation + 1` |
| `next_continuation` | New `PendingExecutionContinuation` episode |
| `next_governed_correlation` | Authority-specific correlation |
| `next_invocation_scope_id` | New authority scope (`agr_*` / `dhr_*` / MSE) |
| `next_authority_scope` | Typed authority discriminator for generation |

(Names may be adapted; semantics frozen.)

#### Composite claim authority (frozen)

```text
ClaimAuthority = (
  materialization_revision,
  pause_generation,
  owner_id,
  fence,
)
```

Only the host holding **current** `ClaimAuthority` for the loaded descriptor may invoke `authority_reblock_from_claimed`. Stale revision **or** stale generation **or** stale owner **or** stale fence → `STALE_CLAIM` / `STALE_REVISION` (fail closed).

#### Effect on success

```text
materialization_state = BLOCKED
claim_ownership = None
pause_generation += 1
materialization_revision += 1
continuation_id = new episode
invocation_scope_id = next authority scope
```

#### Fence reset

Within a generation, `fence` is monotonic for reclaim on **CLAIMED**. On reblock, claim is cleared; next `claim` mints **new lease** with `fence = 1` for the **new** generation. Cross-generation safety is **`materialization_revision` + `pause_generation`**, not fence carry-over.

#### `reclaim()` remains distinct

`reclaim()` — expired **CLAIMED** → **CLAIMED** with new owner/fence on **same** generation — **must not** implement authority pause or reblock.

#### Stale host after reblock

Host A after successful reblock **must not** `mark_consumed`, `abandon`, `authority_reblock_from_claimed`, or enter backend/effect path using old `ClaimAuthority`.

#### Backend entry gate

Coordinator must hold valid **current-generation** `ClaimAuthority` before backend / effect idempotency path.

### R2.8 Reblock matrix (mandatory)

| Situation | Reblock allowed? |
|-----------|-----------------:|
| Correct owner + revision + fence + generation | YES |
| Wrong owner | NO |
| Wrong fence | NO |
| Stale revision | NO |
| Stale generation | NO |
| Expired claim reclaimed by new host | Only **new** owner on same generation via `reclaim` — not reblock |
| Terminal descriptor (CONSUMED / ABANDONED) | NO |

### R2.9 Sequential authority & cross-authority isolation

Agent grant **APPLIED** for Agent generation **does not** satisfy Declarative or MSE requirements. Each authority pause uses R1 reblock + its own pending/grant canonical fields on Task governance (`declarative_hitl_*`, MSE grants unchanged).

Re-entry into Agent Governance on each physical attempt: **fresh evaluation**. Same exact Agent requirement + valid recovery state → verified grant may satisfy without new human approval; sibling invocation → reject.

### R2.10 Permanent invariants (add to proof obligations)

```text
RAW_STRING_IS_NOT_AUTHORITY = true
ONE_CANONICAL_AGENT_GRANT_STORE = true
ACTIVE_LOGICAL_WORK_COUNT <= 1
ONLY_CURRENT_FENCED_OWNER_CAN_REBLOCK = true
GRANT_ONE_SHOT_ACROSS_LOGICAL_INVOCATIONS = true
SAME_LOGICAL_INVOCATION_RECOVERY_ALLOWED = true
CROSS_AUTHORITY_GRANT_REUSE = false
```

Store count freeze:

```text
Agent Governance grant authority stores = 1
Suspended operation stores = 1
Continuation lifecycle mechanisms = 1
```

### R2.11 Revised R6-R5 implementation roadmap (post-R2)

| Gate | Deliverable |
|------|-------------|
| **R6-R5.1** | Agent Governance contracts + lifecycle states (`AVAILABLE`…`TERMINAL`) on canonical Task fields |
| **R6-R5.2** | Canonical durable grant/pending via `AgentGovernanceGrantLifecyclePort` + checkpoint CAS adapter |
| **R6-R5.3** | `AgentGovernanceGrantVerifier` + typed `AgentRuntimePolicyEvaluationContext` |
| **R6-R5.4** | `authority_reblock_from_claimed` + `pause_generation` / `ClaimAuthority` invariant tests |
| **R6-R5.5** | Agent Governance EE L3 pause bridge |
| **R6-R5.6** | Recovery + grant reservation/application + crash matrix behaviors |
| **R6-R5.7** | Sequential Declarative/MSE reblock (no second `prepare`) |
| **R6-R5.8** | Worker E2E |
| **R6-R5.9** | Restart + multi-host + adversarial |

### R2.12 R2 enterprise audit checklist

| # | Check | R2 |
|---|-------|-----|
| 1 | One owner per concern | PASS |
| 2 | One source-of-truth | PASS — Task governance + checkpoint CAS |
| 3 | Typed contract | PASS |
| 4 | CAS/fencing | PASS — grant lifecycle + `ClaimAuthority` |
| 5 | Restart-safe | PASS — crash matrices |
| 6 | Multi-host-safe | PASS — reserve/reclaim + suspended claim |
| 7 | Same logical invocation recoverable | PASS |
| 8 | Sibling blocked | PASS |
| 9 | Cross-authority isolation | PASS |
| 10 | No new lifecycle | PASS — additive fields/APIs only |
| 11 | No raw string authority | PASS |
| 12 | No effect-idempotency abuse | PASS — separation frozen |

---

*End of Amendment R2.*

