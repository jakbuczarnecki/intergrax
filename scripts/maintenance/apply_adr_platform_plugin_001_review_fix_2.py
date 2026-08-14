"""Apply REVIEW-FIX-2 to ADR-PLATFORM-PLUGIN-001 (docs only)."""

from __future__ import annotations

from pathlib import Path

ADR = Path(__file__).resolve().parents[2] / (
    "docs/project/technical/adr/entries/2026-08-14/ADR-PLATFORM-PLUGIN-001.md"
)


def main() -> None:
    text = ADR.read_text(encoding="utf-8")

    text = text.replace(
        "| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 (+ REVIEW-FIX-1) |",
        "| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 (+ REVIEW-FIX-1, REVIEW-FIX-2) |",
    )

    text = text.replace(
        "**REVIEW-FIX-1** closes three decisions required before IMPL-1:\n\n"
        "1. authoritative pre-approval invocation scope persistence (`DeclarativeHitlPendingApproval`),\n"
        "2. mandatory `invocation_scope_id` when `ToolExecutionRequest.idempotency_key` is `None`,\n"
        "3. typed transport of an approved grant into policy re-evaluation.",
        "**REVIEW-FIX-1** closes three decisions required before IMPL-1:\n\n"
        "1. authoritative pre-approval invocation scope persistence (`DeclarativeHitlPendingApproval`),\n"
        "2. mandatory `invocation_scope_id` when `ToolExecutionRequest.idempotency_key` is `None`,\n"
        "3. typed transport of an approved grant into policy re-evaluation.\n\n"
        "**REVIEW-FIX-2** closes two remaining contradictions before IMPL-1:\n\n"
        "1. grant consumption owned by orchestration resume boundary (`GraphExecutor.execute_fn`) — "
        "`RuntimeToolInvoker` must not mutate `Task` or call `DeclarativeHitlGrantCoordinator`,\n"
        "2. independent current invocation identity (`declarative_hitl_invocation_scope_id` / "
        "`PolicyEvaluationContext.invocation_scope_id`) distinct from `approval_grant.invocation_scope_id`.",
    )

    text = text.replace(
        "| 10 | `GraphExecutor.execute_fn` (resume) | reads `task.runtime.governance.declarative_hitl_grant` → `RuntimeRequest.declarative_hitl_grant` |",
        "| 10 | `GraphExecutor.execute_fn` (resume) | reads persisted grant → copies to `RuntimeRequest` → sets current invocation scope → clears persisted grant → `sync_metadata()` → executes resumed request (see §Grant consumption) |",
    )

    text = text.replace(
        "| 10 | **Grant owner:** `TaskGovernanceState.declarative_hitl_grant` — single-use approval artifact; created on canonical APPROVE; consumed per §Grant consumption. |",
        "| 10 | **Grant owner:** `TaskGovernanceState.declarative_hitl_grant` — single-use approval artifact; created on canonical APPROVE; persisted copy consumed at orchestration resume boundary per §Grant consumption. |",
    )

    text = text.replace(
        "| 15 | **Grant transport:** typed field chain only (see §Typed grant transport) — **no** `request.metadata[\"declarative_hitl_grant\"]`. |",
        "| 15 | **Grant transport:** typed field chain only (see §Typed grant transport) — **no** `request.metadata[\"declarative_hitl_grant\"]`. |\n"
        "| 18 | **Current invocation scope transport:** orchestration sets `RuntimeRequest.declarative_hitl_invocation_scope_id` from approved grant on resume; mirrored to `RuntimeState` and `PolicyEvaluationContext.invocation_scope_id`. **Must not** be inferred solely from `approval_grant.invocation_scope_id`. |",
    )

    text = text.replace(
        "- `runtime/policy` and `runtime/tools` **must not** import `Task`, `NexusHitlRunner`, or `HumanPauseCoordinator`.",
        "- `runtime/policy` and `runtime/tools` **must not** import `Task`, `NexusHitlRunner`, or `HumanPauseCoordinator`.\n"
        "- `RuntimeToolInvoker` and `DeclarativePolicyEnforcer` **must not** mutate `Task` / `TaskGovernanceState` or call `DeclarativeHitlGrantCoordinator`.",
    )

    old_invocation = """## Invocation identity

| Question | Frozen answer |
|----------|---------------|
| Can `idempotency_key` be sole identity? | **No** — optional on `ToolExecutionRequest`. |
| Mandatory correlation field | `invocation_scope_id: str` on pending approval and grant. |
| When generated | Once when bridge converts `DeclarativePolicyHitlRequiredError` → `DeclarativePolicyHitlSignal` (first REQUIRE_HITL raise for this invocation). |
| Format | `dhr_{uuid4().hex}` — new identifier; no existing step/invocation ID guarantees per-invocation uniqueness when `idempotency_key` is absent. |
| Stability | Copied unchanged: signal → pending approval → grant → resumed evaluation context. Survives checkpoint via `TaskGovernanceState` persistence. |
| `idempotency_key` role | Additional match dimension when non-null on both request and grant; never assumed present. |"""

    new_invocation = """## Invocation identity

Grant identity and **current invocation identity** are distinct fields. Matching compares them; neither side may be inferred from the other alone.

| Question | Frozen answer |
|----------|---------------|
| Can `idempotency_key` be sole identity? | **No** — optional on `ToolExecutionRequest`. |
| Grant / pending correlation field | `invocation_scope_id: str` on `DeclarativeHitlPendingApproval` and `DeclarativeHitlApprovalGrant`. |
| Current invocation field | `declarative_hitl_invocation_scope_id` on resume transport (`RuntimeRequest` / `RuntimeState`) and `PolicyEvaluationContext.invocation_scope_id` at evaluation. |
| When grant scope is generated | Once when bridge converts `DeclarativePolicyHitlRequiredError` → `DeclarativePolicyHitlSignal` (first REQUIRE_HITL raise for this logical invocation). |
| Format | `dhr_{uuid4().hex}` — new identifier; no existing step/invocation ID guarantees per-invocation uniqueness when `idempotency_key` is absent. |
| Stability | Grant scope copied unchanged: signal → pending approval → grant. Survives checkpoint via `TaskGovernanceState` persistence. |
| `idempotency_key` role | Additional match dimension when non-null on both request and grant; never assumed present. |

### First invocation vs resume

| Phase | `PolicyEvaluationContext.invocation_scope_id` | `approval_grant` | Frozen behavior |
|-------|-----------------------------------------------|------------------|-----------------|
| **First invocation** | `None` — no current invocation scope yet | `None` | Policy yields `REQUIRE_HITL`; bridge creates new `invocation_scope_id`; persists it in `declarative_hitl_pending`. |
| **Resume (approved)** | Set by orchestration to `grant.invocation_scope_id` via typed `declarative_hitl_invocation_scope_id` transport | Immutable copy attached by orchestration resume boundary | Enforcer compares `context.invocation_scope_id == grant.invocation_scope_id` plus remaining dimensions. |
| **Different logical invocation** | Must be a **different** current scope value | Prior grant must not match | New `REQUIRE_HITL` cycle; no grant reuse. |"""

    if old_invocation not in text:
        raise SystemExit("Invocation identity block not found")
    text = text.replace(old_invocation, new_invocation)

    old_transport = """## Typed grant transport

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

`human_approved` metadata flag remains for existing UAEP step-skip semantics only; it **does not** satisfy declarative policy and **does not** replace the grant."""

    new_transport = """## Typed grant transport

**Forbidden:** `request.metadata["declarative_hitl_grant"]`, untyped dict passthrough, or string-key task metadata as canonical cross-layer mechanism.

**Frozen typed path (grant copy + current invocation scope):**

```text
TaskGovernanceState.declarative_hitl_grant
    ↓ GraphExecutor.execute_fn (resume path only): read → copy → clear persisted grant → sync_metadata
RuntimeRequest.declarative_hitl_grant                    # immutable one-use grant copy
RuntimeRequest.declarative_hitl_invocation_scope_id      # current invocation identity for this resume
    ↓ AgentEngine.run_agent_with_result → RuntimeState construction
RuntimeState.declarative_hitl_grant
RuntimeState.declarative_hitl_invocation_scope_id
    ↓ RuntimeToolInvoker.invoke(state, request)
PolicyEvaluationContext.approval_grant                   # immutable grant copy (read-only)
PolicyEvaluationContext.invocation_scope_id              # current invocation identity (independent field)
    ↓ DeclarativePolicyEnforcer.evaluate_tool_invocation
```

| Hop | Owner module | Field |
|-----|--------------|-------|
| Persisted grant | `task_contract.py` | `TaskGovernanceState.declarative_hitl_grant` |
| Resume transfer + consume | `graph_executor.py` | reads grant; sets `RuntimeRequest.declarative_hitl_grant` + `declarative_hitl_invocation_scope_id`; clears persisted grant; `sync_metadata()` |
| Runtime mirror | `runtime_state.py` | `RuntimeState.declarative_hitl_grant`, `RuntimeState.declarative_hitl_invocation_scope_id` ← from request at engine init |
| Evaluation input | `evaluation.py` | `PolicyEvaluationContext.approval_grant`, `PolicyEvaluationContext.invocation_scope_id` |
| Consumer | `declarative_enforcer.py` | reads grant + current scope; **must not** mutate `Task` |

`PolicyEvaluationContext.invocation_scope_id` **must not** be populated by reading `approval_grant.invocation_scope_id` inside the enforcer as a substitute for an independently transported current scope. Orchestration owns the assignment on resume.

`human_approved` metadata flag remains for existing UAEP step-skip semantics only; it **does not** satisfy declarative policy and **does not** replace the grant."""

    if old_transport not in text:
        raise SystemExit("Typed grant transport block not found")
    text = text.replace(old_transport, new_transport)

    old_matching = """## Grant matching

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

Mismatch → `DeclarativePolicyHitlRequiredError` (re-pause) or orchestration failure per existing HITL path; grant not consumed."""

    new_matching = """## Grant matching

Matching compares **two independent sides**. No self-comparison within grant alone or within current context alone.

**CURRENT invocation context** (from `PolicyEvaluationContext`, `ToolExecutionRequest`, and current policy decision):

| Field | Source |
|-------|--------|
| `invocation_scope_id` | `PolicyEvaluationContext.invocation_scope_id` |
| `task_id` | task at evaluation boundary |
| `run_id` | `ToolExecutionRequest.run_id` |
| `step_id` | `ToolExecutionRequest.step_id` |
| `tool_id` | `ToolExecutionRequest.tool_id` |
| `idempotency_key` | `ToolExecutionRequest.idempotency_key` |
| matched rule IDs | current `PolicyEnforcementDecision` |
| provenance digest | current `PolicyEnforcementDecision.provenance_digest` |

**APPROVAL GRANT** (immutable `DeclarativeHitlApprovalGrant` copy on `PolicyEvaluationContext.approval_grant`):

| Field | Source |
|-------|--------|
| `invocation_scope_id` | `grant.invocation_scope_id` |
| `task_id` | `grant.task_id` |
| `run_id` | `grant.run_id` |
| `step_id` | `grant.step_id` |
| `tool_id` | `grant.tool_id` |
| `idempotency_key` | `grant.idempotency_key` |
| matched rule IDs | `grant.matched_rule_ids` |
| provenance digest | `grant.policy_provenance_digest` |

**Matching predicate** — grant satisfies `REQUIRE_HITL` only when **all** cross-field equalities pass:

| Dimension | Predicate |
|-----------|-----------|
| `invocation_scope_id` | `context.invocation_scope_id == grant.invocation_scope_id` (both sides required; first invocation has `context.invocation_scope_id is None` → no match) |
| `task_id` | current task id == `grant.task_id` |
| `run_id` | `ToolExecutionRequest.run_id` == `grant.run_id` |
| `step_id` | `ToolExecutionRequest.step_id` == `grant.step_id` |
| `tool_id` | `ToolExecutionRequest.tool_id` == `grant.tool_id` |
| matched rule IDs | set equality (order-independent) between current decision and `grant.matched_rule_ids` |
| `human_request_id` | originating approval id equals `grant.human_request_id` (audit correlation) |
| `idempotency_key` | if non-null on request, must equal `grant.idempotency_key`; if null on request, grant field ignored |
| provenance digest | if non-null on current decision, must equal `grant.policy_provenance_digest`; mismatch → no satisfaction (stale approval) |

| Policy action | Grant behavior |
|---------------|----------------|
| `DENY` | **Never** grant-satisfiable — evaluated first; in-memory grant copy ignored for authorization |
| `REQUIRE_HITL` | Grant may satisfy when predicate passes |
| `ALLOW` | Unchanged — grant irrelevant |

Mismatch after resume → `DeclarativePolicyHitlRequiredError` (re-pause) or orchestration failure; persisted grant already consumed at resume boundary and is **not** restored."""

    if old_matching not in text:
        raise SystemExit("Grant matching block not found")
    text = text.replace(old_matching, new_matching)

    old_consumption = """## Grant consumption

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

**Clearing owner (IMPL-1):** `DeclarativeHitlGrantCoordinator.consume_grant(task)` called from `RuntimeToolInvoker` immediately after enforcer accepts grant, before `ToolExecutor` call."""

    new_consumption = """## Grant consumption

### Orchestration-owned consumption point (frozen)

**Owner:** `GraphExecutor.execute_fn` resume boundary only (orchestration tier). Evidence: resume path already attaches governance artifacts to `RuntimeRequest`; consuming persisted grant here preserves frozen dependency direction (`runtime/tools` and `runtime/policy` do not mutate `Task`).

**Resume sequence (persisted grant single-use before low-level execution):**

1. Read `TaskGovernanceState.declarative_hitl_grant`
2. Copy immutable grant into `RuntimeRequest.declarative_hitl_grant`
3. Set `RuntimeRequest.declarative_hitl_invocation_scope_id = grant.invocation_scope_id`
4. Clear `TaskGovernanceState.declarative_hitl_grant = None`
5. `task.sync_metadata()`
6. Execute resumed `RuntimeRequest`

**Low-level boundaries (`RuntimeToolInvoker`, `DeclarativePolicyEnforcer`):**

- **MAY** read immutable grant copy on `PolicyEvaluationContext.approval_grant`
- **MAY** read `PolicyEvaluationContext.invocation_scope_id`
- **MUST NOT** mutate `Task` / `TaskGovernanceState`
- **MUST NOT** call `DeclarativeHitlGrantCoordinator`

IMPL-1 may implement steps 1–5 inline in `graph_executor.py` or via an orchestration helper on `DeclarativeHitlGrantCoordinator` (e.g. `transfer_persisted_grant_for_resume(task) → RuntimeRequestGrantBundle`); the coordinator remains human/orchestration-owned and is **never** imported from `RuntimeToolInvoker`.

### Failure and retry semantics

| Event | Grant behavior |
|-------|----------------|
| Resume boundary executes | Persisted grant cleared; in-memory copy on `RuntimeRequest` / `RuntimeState` only |
| Policy eval: **DENY** after resume | Invocation blocked; consumed persisted grant is **not** restored |
| Policy eval: **REQUIRE_HITL** + mismatch | Re-pause; consumed persisted grant is **not** restored |
| Policy eval: **REQUIRE_HITL** + match | Handler may run once using in-memory grant copy; no Task mutation at enforcer |
| Handler throws / `ToolExecutionResult.success=False` | Grant remains consumed (persisted already cleared); new approval required |
| Retry within **same** logical `RuntimeToolInvoker` execution | May reuse same in-memory request/grant copy only when existing retry semantics treat it as one invocation |
| Graph/step retry after returning to orchestration | No persisted grant → `REQUIRE_HITL` blocks again |
| APPROVE | Pending → grant; pending cleared |
| REJECT | Pending cleared; no grant |
| TIMEOUT | Pending cleared; no grant |
| ESCALATE (terminal) | Pending cleared; no grant |

**Security balance:** persisted consumption at orchestration resume prevents reusable approval across checkpoints. Handler failure after resume requires re-approval — aligns with handler call-count invariant and existing idempotency ledger when `idempotency_key` is set.

## Grant coordinator ownership

`DeclarativeHitlGrantCoordinator` (`intergrax/runtime/human/declarative_hitl_grant.py`) remains orchestration/human-owned.

| Allowed | Forbidden |
|---------|-----------|
| Create grant from pending on canonical APPROVE | `RuntimeToolInvoker` importing the coordinator |
| Clear pending/grant on REJECT / ESCALATE / TIMEOUT | `DeclarativePolicyEnforcer` mutating `Task` state |
| Orchestration-side transfer/consume of persisted grant at resume boundary | Low-level policy/tool layers calling coordinator or clearing `TaskGovernanceState` |"""

    if old_consumption not in text:
        raise SystemExit("Grant consumption block not found")
    text = text.replace(old_consumption, new_consumption)

    old_approve_flow = """GraphExecutor.execute_fn (resume)
  → RuntimeRequest.declarative_hitl_grant = task.runtime.governance.declarative_hitl_grant
  → request.metadata["human_approved"] = True  (UAEP skip only; not policy grant)

RuntimeToolInvoker.invoke (resume)
  → PolicyEvaluationContext(approval_grant=state.declarative_hitl_grant, ...)
  → DeclarativePolicyEnforcer: DENY still DENY
  → REQUIRE_HITL + matching grant → satisfied
  → DeclarativeHitlGrantCoordinator.consume_grant(task)
  → handler executes once"""

    new_approve_flow = """GraphExecutor.execute_fn (resume)
  → read task.runtime.governance.declarative_hitl_grant
  → RuntimeRequest.declarative_hitl_grant = copy(grant)
  → RuntimeRequest.declarative_hitl_invocation_scope_id = grant.invocation_scope_id
  → task.runtime.governance.declarative_hitl_grant = None
  → task.sync_metadata()
  → request.metadata["human_approved"] = True  (UAEP skip only; not policy grant)

RuntimeToolInvoker.invoke (resume)
  → PolicyEvaluationContext(
       approval_grant=state.declarative_hitl_grant,
       invocation_scope_id=state.declarative_hitl_invocation_scope_id,
       ...)
  → DeclarativePolicyEnforcer: DENY still DENY
  → REQUIRE_HITL + matching grant (context.invocation_scope_id == grant.invocation_scope_id, ...)
  → satisfied — no Task mutation; no coordinator call
  → handler executes once"""

    if old_approve_flow not in text:
        raise SystemExit("Human APPROVE flow block not found")
    text = text.replace(old_approve_flow, new_approve_flow)

    text = text.replace(
        "`declarative_enforcer` and `invoker` have no Task/HITL imports. Bridge imports contracts + error only. Grant coordinator lives in `runtime/human/` (orchestration tier). UAEP/graph runners consume existing `GovernanceResolution` path. No edge from `runtime/policy` or `runtime/tools` to `HumanPauseCoordinator`.",
        "`declarative_enforcer` and `invoker` have no Task/HITL imports and do not mutate `TaskGovernanceState`. Bridge imports contracts + error only. Grant coordinator lives in `runtime/human/` (orchestration tier) and is invoked only from intake/graph orchestration — **not** from `RuntimeToolInvoker`. Persisted grant consumption happens in `GraphExecutor.execute_fn` resume boundary before `RuntimeToolInvoker.invoke`. UAEP/graph runners consume existing `GovernanceResolution` path. No edge from `runtime/policy` or `runtime/tools` to `HumanPauseCoordinator` or `DeclarativeHitlGrantCoordinator`.",
    )

    text = text.replace(
        "Policy eval (`declarative_policy_evaluation`) → HITL signal (`declarative_policy_hitl_required`) → pause (`HUMAN_APPROVAL_REQUESTED`) → approval → grant created (`declarative_hitl_grant_created`) → grant satisfied (`declarative_hitl_grant_consumed`) → tool invocation end.",
        "Policy eval (`declarative_policy_evaluation`) → HITL signal (`declarative_policy_hitl_required`) → pause (`HUMAN_APPROVAL_REQUESTED`) → approval → grant created (`declarative_hitl_grant_created`) → orchestration resume transfer (`declarative_hitl_grant_consumed` at `GraphExecutor.execute_fn`) → grant satisfied at enforcer → tool invocation end.",
    )

    old_e2e = """## E2E acceptance contract

**ENFORCE** + `REQUIRE_HITL` on test tool.

1. First invoke: handler calls = 0; `WAITING_FOR_HUMAN`; `declarative_hitl_pending` persisted; checkpoint.
2. Approve via existing API; `declarative_hitl_grant` created from pending; resume; grant satisfies HITL; handler calls = 1; grant consumed; task completes.
3. REJECT: pending cleared; no grant; handler calls = 0.
4. ESCALATE: canonical escalation path; terminal escalation clears pending without grant.
5. AUDIT_ONLY: no pause for declarative `REQUIRE_HITL` alone.
6. Policy digest change after pause: grant does not satisfy without re-approval."""

    new_e2e = """## E2E acceptance contract

**ENFORCE** + `REQUIRE_HITL` on test tool.

1. First invoke: handler calls = 0; `WAITING_FOR_HUMAN`; `declarative_hitl_pending` persisted; checkpoint.
2. Approve via existing API; `declarative_hitl_grant` created from pending; resume; grant satisfies HITL; handler calls = 1; persisted grant consumed at resume boundary; task completes.
3. REJECT: pending cleared; no grant; handler calls = 0.
4. ESCALATE: canonical escalation path; terminal escalation clears pending without grant.
5. AUDIT_ONLY: no pause for declarative `REQUIRE_HITL` alone.
6. Policy digest change after pause: grant does not satisfy without re-approval.

**REVIEW-FIX-2 proof obligations (IMPL-1 E2E):**

| ID | Proof |
|----|-------|
| **A** | Once resumed execution starts, `TaskGovernanceState.declarative_hitl_grant` is absent (persisted grant consumed at orchestration resume). |
| **B** | `RuntimeRequest` / `RuntimeState` still carry the immutable one-use grant copy for the resumed invocation. |
| **C** | `PolicyEvaluationContext.invocation_scope_id` equals `grant.invocation_scope_id` for an approved resume (independent transport, not enforcer inference). |
| **D** | A second logical invocation with a different current `invocation_scope_id` does **not** reuse the prior grant. |
| **E** | `DENY` after approve does not restore the consumed persisted grant. |
| **F** | Failed resumed invocation (handler failure / mismatch re-pause) requires fresh human approval. |"""

    if old_e2e not in text:
        raise SystemExit("E2E block not found")
    text = text.replace(old_e2e, new_e2e)

    text = text.replace(
        "**Accepted** — pending approval DTO, grant DTO, invocation identity, typed transport, matching predicate, and consumption lifecycle frozen for IMPL-1.",
        "**Accepted** — pending approval DTO, grant DTO, distinct grant vs current invocation identity, orchestration-owned persisted grant consumption, typed transport, cross-field matching predicate, and failure/retry semantics frozen for IMPL-1 (REVIEW-FIX-2).",
    )

    text = text.replace(
        "approval grant (declarative_hitl_grant set, declarative_hitl_pending cleared)\n    ↓ successful matching invocation (§Grant consumption)\nconsumed / cleared (declarative_hitl_grant = None)",
        "approval grant (declarative_hitl_grant set, declarative_hitl_pending cleared)\n    ↓ orchestration resume transfer (§Grant consumption)\nconsumed / cleared (declarative_hitl_grant = None before RuntimeToolInvoker)",
    )

    ADR.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated {ADR}")


if __name__ == "__main__":
    main()
