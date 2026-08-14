"""Apply REVIEW-FIX-3 to ADR-PLATFORM-PLUGIN-001 (docs only)."""

from __future__ import annotations

from pathlib import Path

ADR = Path(__file__).resolve().parents[2] / (
    "docs/project/technical/adr/entries/2026-08-14/ADR-PLATFORM-PLUGIN-001.md"
)


def main() -> None:
    text = ADR.read_text(encoding="utf-8")

    text = text.replace(
        "| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 (+ REVIEW-FIX-1, REVIEW-FIX-2) |",
        "| **Task** | PLATFORM-PLUGIN-ENTERPRISE-4-ADR-1 (+ REVIEW-FIX-1, REVIEW-FIX-2, REVIEW-FIX-3) |",
    )

    text = text.replace(
        "2. independent current invocation identity (`declarative_hitl_invocation_scope_id` / "
        "`PolicyEvaluationContext.invocation_scope_id`) distinct from `approval_grant.invocation_scope_id`.",
        "2. independent current invocation identity (`declarative_hitl_invocation_scope_id` / "
        "`PolicyEvaluationContext.invocation_scope_id`) distinct from `approval_grant.invocation_scope_id`.\n\n"
        "**REVIEW-FIX-3** closes the final scope-ownership contradiction before IMPL-1:\n\n"
        "1. current invocation identity owned by `ToolExecutionRequest` — not `RuntimeRequest` / `RuntimeState` "
        "(one runtime request may execute multiple tool invocations, including parallel read-only calls),\n"
        "2. grant transport (`declarative_hitl_grant`) separated from per-invocation scope assignment,\n"
        "3. one-shot scope assignment at exact resumed tool-request reconstruction with dimensional verification,\n"
        "4. in-memory grant single-use within the same `RuntimeRequest` execution.",
    )

    text = text.replace(
        "| 10 | `GraphExecutor.execute_fn` (resume) | reads persisted grant → copies to `RuntimeRequest` → sets current invocation scope → clears persisted grant → `sync_metadata()` → executes resumed request (see §Grant consumption) |",
        "| 10 | `GraphExecutor.execute_fn` (resume) | reads persisted grant → copies to `RuntimeRequest.declarative_hitl_grant` → clears persisted grant → `sync_metadata()` → executes resumed request; invocation scope assigned per §Invocation scope assignment (see §Grant consumption) |",
    )

    text = text.replace(
        "| 18 | **Current invocation scope transport:** orchestration sets `RuntimeRequest.declarative_hitl_invocation_scope_id` from approved grant on resume; mirrored to `RuntimeState` and `PolicyEvaluationContext.invocation_scope_id`. **Must not** be inferred solely from `approval_grant.invocation_scope_id`. |",
        "| 18 | **Current invocation scope owner:** `ToolExecutionRequest.declarative_hitl_invocation_scope_id` — assigned once at exact resumed tool-request reconstruction (§Invocation scope assignment). Grant transport uses `RuntimeRequest` / `RuntimeState.declarative_hitl_grant` only. **Must not** broadcast scope to all tool calls in one runtime request. |",
    )

    old_invocation = """## Invocation identity

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

    new_invocation = """## Invocation identity

Grant identity (**approval evidence**) and **current invocation identity** are distinct concepts. Matching compares them; neither side may be inferred from the other alone.

| Question | Frozen answer |
|----------|---------------|
| Can `idempotency_key` be sole identity? | **No** — optional on `ToolExecutionRequest`. |
| Grant / pending correlation field | `invocation_scope_id: str` on `DeclarativeHitlPendingApproval` and `DeclarativeHitlApprovalGrant`. |
| Grant transport field | `declarative_hitl_grant` on `TaskGovernanceState` → `RuntimeRequest` → `RuntimeState` (approval evidence copy only). |
| Current invocation field | `declarative_hitl_invocation_scope_id: str \\| None` on **`ToolExecutionRequest`** — authoritative per-invocation identity. |
| Policy evaluation mirror | `PolicyEvaluationContext.invocation_scope_id` populated from `ToolExecutionRequest.declarative_hitl_invocation_scope_id` at invoke time — **not** from shared `RuntimeState`. |
| When grant scope is generated | Once when bridge converts `DeclarativePolicyHitlRequiredError` → `DeclarativePolicyHitlSignal` (first REQUIRE_HITL raise for this logical invocation). |
| Format | `dhr_{uuid4().hex}` — new identifier; no existing step/invocation ID guarantees per-invocation uniqueness when `idempotency_key` is absent. |
| Stability | Grant scope copied unchanged: signal → pending approval → grant. Survives checkpoint via `TaskGovernanceState` persistence. |
| `idempotency_key` role | Additional match dimension when non-null on both request and grant; never assumed present. |

**Evidence:** `tool_loop.execute_planned_tool_calls(calls: Sequence[PlannedToolCall])` may invoke multiple tools within one `RuntimeRequest` / runtime state (including parallel read-only calls). Each `_invoke_planned_call()` constructs a distinct `ToolExecutionRequest(run_id, step_id, tool_id, idempotency_key, ...)`. `RuntimeRequest`-level invocation identity is therefore too broad.

### First invocation vs resume

| Phase | `ToolExecutionRequest.declarative_hitl_invocation_scope_id` | `approval_grant` | Frozen behavior |
|-------|--------------------------------------------------------------|------------------|-----------------|
| **First invocation** | `None` — no current invocation scope yet | `None` | Policy yields `REQUIRE_HITL`; bridge creates new `invocation_scope_id`; persists it in `declarative_hitl_pending`. |
| **Resume (approved)** | Set on **exactly one** reconstructed `ToolExecutionRequest` by the resumed tool-request builder (§Invocation scope assignment) | Immutable copy on `RuntimeState.declarative_hitl_grant` from orchestration resume boundary | Enforcer compares `request.declarative_hitl_invocation_scope_id == grant.invocation_scope_id` plus remaining dimensions. |
| **Other tool calls same runtime request** | `None` | Same in-memory grant copy may be visible on `RuntimeState` | Grant **must not** satisfy `REQUIRE_HITL` without matching current scope on that specific request. |
| **Different logical invocation** | Must be a **different** current scope value (or `None`) | Prior grant must not match | New `REQUIRE_HITL` cycle; no grant reuse. |"""

    if old_invocation not in text:
        raise SystemExit("Invocation identity block not found")
    text = text.replace(old_invocation, new_invocation)

    old_transport = """## Typed grant transport

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

    new_transport = """## Typed grant transport

**Forbidden:** `request.metadata["declarative_hitl_grant"]`, untyped dict passthrough, string-key task metadata as canonical cross-layer mechanism, and **`RuntimeRequest` / `RuntimeState` as authoritative current invocation identity**.

**Two separate transports:**

| Concept | Typed path | Role |
|---------|------------|------|
| **A. Approval evidence** | `TaskGovernanceState.declarative_hitl_grant` → `RuntimeRequest.declarative_hitl_grant` → `RuntimeState.declarative_hitl_grant` | Makes approved grant available to execution within one resumed runtime request |
| **B. Current invocation identity** | `ToolExecutionRequest.declarative_hitl_invocation_scope_id` → `PolicyEvaluationContext.invocation_scope_id` | Identifies the one governed tool invocation currently attempting to consume that grant |

**Frozen typed path:**

```text
TaskGovernanceState.declarative_hitl_grant
    ↓ GraphExecutor.execute_fn (resume path only): read → copy → clear persisted grant → sync_metadata
RuntimeRequest.declarative_hitl_grant                    # immutable one-use grant copy (transport only)
    ↓ AgentEngine.run_agent_with_result → RuntimeState construction
RuntimeState.declarative_hitl_grant                      # grant transport mirror

Exact resumed tool request builder (§Invocation scope assignment)
    ↓ one-shot assignment on matched ToolExecutionRequest only
ToolExecutionRequest.declarative_hitl_invocation_scope_id = grant.invocation_scope_id

RuntimeToolInvoker.invoke(state, request)
    ↓
PolicyEvaluationContext(
    approval_grant=state.declarative_hitl_grant,
    invocation_scope_id=request.declarative_hitl_invocation_scope_id,
)
    ↓ DeclarativePolicyEnforcer.evaluate_tool_invocation
```

| Hop | Owner module | Field |
|-----|--------------|-------|
| Persisted grant | `task_contract.py` | `TaskGovernanceState.declarative_hitl_grant` |
| Resume transfer + consume | `graph_executor.py` | reads grant; sets `RuntimeRequest.declarative_hitl_grant`; clears persisted grant; `sync_metadata()` — **does not** set invocation scope |
| Runtime grant mirror | `runtime_state.py` | `RuntimeState.declarative_hitl_grant` ← from request at engine init |
| Invocation scope assignment | resumed tool-request builder (see §Invocation scope assignment) | `ToolExecutionRequest.declarative_hitl_invocation_scope_id` on exactly one matched request |
| Evaluation input | `evaluation.py` / `invoker.py` | `PolicyEvaluationContext.approval_grant`, `PolicyEvaluationContext.invocation_scope_id` |
| Consumer | `declarative_enforcer.py` | compares request-scoped current scope vs grant; **must not** mutate `Task` |

`PolicyEvaluationContext.invocation_scope_id` **must** come from `ToolExecutionRequest.declarative_hitl_invocation_scope_id`. It **must not** be populated from shared `RuntimeState`, and **must not** be inferred inside the enforcer from `approval_grant.invocation_scope_id` alone.

`human_approved` metadata flag remains for existing UAEP step-skip semantics only; it **does not** satisfy declarative policy and **does not** replace the grant.

## Invocation scope assignment

**Owner:** the module that reconstructs the exact resumed `ToolExecutionRequest` for the governed invocation (IMPL-1: allowlisted tool-request builders on the resume path — e.g. `tool_loop._invoke_planned_call`, `catalog_dispatch`, `execute_declarative_actions` / `DeclarativeToolInvoker` — whichever rebuilds the paused call).

**When:** after `RuntimeState.declarative_hitl_grant` is available and **before** `RuntimeToolInvoker.invoke` for that specific request.

**Algorithm (frozen):**

1. Candidate request must match pending/grant dimensions: run/task context, `step_id`, `tool_id`, and `idempotency_key` when present on the grant.
2. If exactly one candidate matches → set `candidate.declarative_hitl_invocation_scope_id = grant.invocation_scope_id`.
3. All other `ToolExecutionRequest` instances constructed in the same `RuntimeRequest` execution → `declarative_hitl_invocation_scope_id = None`.
4. If zero or multiple candidates match → **fail closed** (orchestration error / re-HITL); do not guess.

**Must not:** copy `grant.invocation_scope_id` to every `ToolExecutionRequest` in the same runtime request; assign scope at `GraphExecutor` without dimensional verification; mutate `Task` from tool layers.

IMPL-1 may use a local one-shot assignment helper owned by the request builder; persisted grant consumption remains orchestration-owned per §Grant consumption.

## Multi-tool resume semantics

One `RuntimeRequest` may plan calls `A`, `B`, `A2` (same `tool_id` allowed). Approval targets governed invocation **A** only.

| Call | `declarative_hitl_invocation_scope_id` | Grant satisfaction |
|------|----------------------------------------|------------------|
| **A** (approved target) | `grant.invocation_scope_id` | May satisfy `REQUIRE_HITL` when remaining dimensions match |
| **B** | `None` (or different scope) | **Cannot** satisfy using the same in-memory grant |
| **A2** (repeat same tool) | `None` unless provably the exact resumed governed invocation | **Cannot** reuse grant merely because `tool_id` matches |

## Parallel tool dispatch

The tool loop may execute read-only calls concurrently within one runtime request.

| Rule | Frozen behavior |
|------|-----------------|
| Grant attachment | A single approval grant may attach to **at most one** `ToolExecutionRequest` |
| Scope broadcast | **Never** copy grant/current scope to all parallel calls |
| Ambiguity before dispatch | If the intended governed invocation cannot be uniquely identified → **fail closed / re-HITL** rather than guessing |
| Policy evaluation | Each parallel `RuntimeToolInvoker.invoke` evaluates with its own `request.declarative_hitl_invocation_scope_id`; only the scoped request may consume the grant |"""

    if old_transport not in text:
        raise SystemExit("Typed grant transport block not found")
    text = text.replace(old_transport, new_transport)

    text = text.replace(
        "| `invocation_scope_id` | `PolicyEvaluationContext.invocation_scope_id` |",
        "| `invocation_scope_id` | `ToolExecutionRequest.declarative_hitl_invocation_scope_id` (mirrored to `PolicyEvaluationContext.invocation_scope_id` at invoke) |",
    )

    text = text.replace(
        "| `invocation_scope_id` | `context.invocation_scope_id == grant.invocation_scope_id` (both sides required; first invocation has `context.invocation_scope_id is None` → no match) |",
        "| `invocation_scope_id` | `ToolExecutionRequest.declarative_hitl_invocation_scope_id == grant.invocation_scope_id` (both sides required; `None` on request → no match) |",
    )

    old_consumption_resume = """**Resume sequence (persisted grant single-use before low-level execution):**

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
- **MUST NOT** call `DeclarativeHitlGrantCoordinator`"""

    new_consumption_resume = """**Resume sequence (persisted grant single-use before low-level execution):**

1. Read `TaskGovernanceState.declarative_hitl_grant`
2. Copy immutable grant into `RuntimeRequest.declarative_hitl_grant`
3. Clear `TaskGovernanceState.declarative_hitl_grant = None`
4. `task.sync_metadata()`
5. Execute resumed `RuntimeRequest`
6. Resumed tool-request builder performs one-shot `ToolExecutionRequest.declarative_hitl_invocation_scope_id` assignment (§Invocation scope assignment)

**In-memory grant single-use within one `RuntimeRequest`:**

- Persisted grant is consumed at step 3; an immutable copy remains on `RuntimeState.declarative_hitl_grant` for that runtime execution only.
- The grant is eligible **only** for the `ToolExecutionRequest` carrying matching `declarative_hitl_invocation_scope_id`.
- Other `ToolExecutionRequest` instances in the same runtime request **cannot** satisfy `REQUIRE_HITL` using that in-memory grant.
- After the matching invocation is selected and executed, no subsequent invocation in the same runtime request may obtain that scope id again (one-shot assignment; request builder owns local guard if needed).
- **Do not** mutate `Task` from `RuntimeToolInvoker`.

**Low-level boundaries (`RuntimeToolInvoker`, `DeclarativePolicyEnforcer`):**

- **MAY** read immutable grant copy on `PolicyEvaluationContext.approval_grant` (from `RuntimeState`)
- **MAY** read `PolicyEvaluationContext.invocation_scope_id` (from `ToolExecutionRequest`)
- **MUST NOT** mutate `Task` / `TaskGovernanceState`
- **MUST NOT** call `DeclarativeHitlGrantCoordinator`"""

    if old_consumption_resume not in text:
        raise SystemExit("Grant consumption resume block not found")
    text = text.replace(old_consumption_resume, new_consumption_resume)

    text = text.replace(
        "| Resume boundary executes | Persisted grant cleared; in-memory copy on `RuntimeRequest` / `RuntimeState` only |",
        "| Resume boundary executes | Persisted grant cleared; in-memory grant copy on `RuntimeRequest` / `RuntimeState`; scope assigned per-request only |",
    )

    text = text.replace(
        "| Retry within **same** logical `RuntimeToolInvoker` execution | May reuse same in-memory request/grant copy only when existing retry semantics treat it as one invocation |",
        "| Retry within **same** logical `RuntimeToolInvoker` execution | May reuse same in-memory request/grant copy only when existing retry semantics treat it as one invocation **and** the same `ToolExecutionRequest` retains the assigned scope |",
    )

    old_approve_flow = """GraphExecutor.execute_fn (resume)
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

    new_approve_flow = """GraphExecutor.execute_fn (resume)
  → read task.runtime.governance.declarative_hitl_grant
  → RuntimeRequest.declarative_hitl_grant = copy(grant)
  → task.runtime.governance.declarative_hitl_grant = None
  → task.sync_metadata()
  → request.metadata["human_approved"] = True  (UAEP skip only; not policy grant)

Resumed tool request builder (exact governed invocation)
  → verify run/task, step_id, tool_id, idempotency_key vs grant
  → ToolExecutionRequest.declarative_hitl_invocation_scope_id = grant.invocation_scope_id  (one call only)
  → all other ToolExecutionRequest instances in same RuntimeRequest → scope None

RuntimeToolInvoker.invoke (resume, scoped request only)
  → PolicyEvaluationContext(
       approval_grant=state.declarative_hitl_grant,
       invocation_scope_id=request.declarative_hitl_invocation_scope_id,
       ...)
  → DeclarativePolicyEnforcer: DENY still DENY
  → REQUIRE_HITL + matching grant (request.declarative_hitl_invocation_scope_id == grant.invocation_scope_id, ...)
  → satisfied — no Task mutation; no coordinator call
  → handler executes once"""

    if old_approve_flow not in text:
        raise SystemExit("Human APPROVE flow block not found")
    text = text.replace(old_approve_flow, new_approve_flow)

    text = text.replace(
        "- New typed fields on `RuntimeRequest`, `RuntimeState`, `PolicyEvaluationContext`, `TaskGovernanceState` — no metadata key addition.",
        "- New typed fields on `RuntimeRequest`, `RuntimeState`, `PolicyEvaluationContext`, `TaskGovernanceState`, `ToolExecutionRequest` — no metadata key addition.\n"
        "- `declarative_hitl_invocation_scope_id` on `ToolExecutionRequest` only — **not** on `RuntimeRequest` / `RuntimeState`.",
    )

    text = text.replace(
        "- `intergrax/runtime/nexus/tools/tool_loop.py`",
        "- `intergrax/tools/execution_models.py` (`ToolExecutionRequest.declarative_hitl_invocation_scope_id`)\n"
        "- `intergrax/runtime/nexus/tools/tool_loop.py`",
    )

    old_e2e_fix2 = """**REVIEW-FIX-2 proof obligations (IMPL-1 E2E):**

| ID | Proof |
|----|-------|
| **A** | Once resumed execution starts, `TaskGovernanceState.declarative_hitl_grant` is absent (persisted grant consumed at orchestration resume). |
| **B** | `RuntimeRequest` / `RuntimeState` still carry the immutable one-use grant copy for the resumed invocation. |
| **C** | `PolicyEvaluationContext.invocation_scope_id` equals `grant.invocation_scope_id` for an approved resume (independent transport, not enforcer inference). |
| **D** | A second logical invocation with a different current `invocation_scope_id` does **not** reuse the prior grant. |
| **E** | `DENY` after approve does not restore the consumed persisted grant. |
| **F** | Failed resumed invocation (handler failure / mismatch re-pause) requires fresh human approval. |"""

    new_e2e_fix2 = """**REVIEW-FIX-2 proof obligations (IMPL-1 E2E):**

| ID | Proof |
|----|-------|
| **A** | Once resumed execution starts, `TaskGovernanceState.declarative_hitl_grant` is absent (persisted grant consumed at orchestration resume). |
| **B** | `RuntimeRequest` / `RuntimeState` still carry the immutable one-use grant copy for the resumed runtime request. |
| **D** | A second logical invocation with a different current `invocation_scope_id` does **not** reuse the prior grant. |
| **E** | `DENY` after approve does not restore the consumed persisted grant. |
| **F** | Failed resumed invocation (handler failure / mismatch re-pause) requires fresh human approval. |

**REVIEW-FIX-3 proof obligations (IMPL-1 E2E):**

| ID | Proof |
|----|-------|
| **G** | One `RuntimeRequest` contains at least two tool calls (including multi-tool or parallel read-only path). |
| **H** | Human approval targets exactly one governed call. |
| **I** | Only the targeted `ToolExecutionRequest` receives `declarative_hitl_invocation_scope_id`; siblings receive `None`. |
| **J** | Approved call executes once (handler call count = 1). |
| **K** | Second tool call in the same runtime request cannot reuse the in-memory grant. |
| **L** | Repeated same `tool_id` within the same runtime request cannot reuse the grant without exact resumed scope match. |
| **M** | Parallel read-only invocation does not receive another call's grant or scope. |"""

    if old_e2e_fix2 not in text:
        raise SystemExit("E2E FIX-2 block not found")
    text = text.replace(old_e2e_fix2, new_e2e_fix2)

    text = text.replace(
        "**Accepted** — pending approval DTO, grant DTO, distinct grant vs current invocation identity, orchestration-owned persisted grant consumption, typed transport, cross-field matching predicate, and failure/retry semantics frozen for IMPL-1 (REVIEW-FIX-2).",
        "**Accepted** — pending approval DTO, grant DTO, grant transport vs per-request invocation identity, orchestration-owned persisted grant consumption, one-shot scope assignment at tool-request reconstruction, multi-tool and parallel semantics, cross-field matching predicate, in-memory grant single-use, and failure/retry semantics frozen for IMPL-1 (REVIEW-FIX-3). IMPL-1 READY. CAND-007 remains PARTIAL until E2E implementation.",
    )

    ADR.write_text(text, encoding="utf-8", newline="\n")
    print(f"Updated {ADR}")


if __name__ == "__main__":
    main()
