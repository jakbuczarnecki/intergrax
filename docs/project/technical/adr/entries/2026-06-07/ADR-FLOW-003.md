# ADR-FLOW-003: `MODIFY_PLAN` decision semantics

| Field | Value |
|-------|-------|
| **Status** | Accepted (FLOW-16) |
| **Date** | 2026-06-07 |
| **Deciders** | Intergrax platform architecture |
| **Related** | [`architecture/NEXUS_EXECUTION_FLOW.md`](../../architecture/NEXUS_EXECUTION_FLOW.md) §10 · plan `FLOW-16` / `FLOW-GAP-15` |

## Context

`AgentDecision.MODIFY_PLAN` appears in the UAEP decision enum and flow reference §10 decision matrix as **reserved / policy-dependent replan**. No Nexus runner today interprets this decision to mutate `NexusPlan` or `ExecutionGraph` mid-flight.

Product teams need a clear contract: either implement bounded replan, or document that replan is **out of scope** for harness v1 (dynamic changes go through `HANDOFF` or declarative `graph_spec` only).

## Decision

Adopt **Option B - reserved with explicit non-support in v1**:

1. **`MODIFY_PLAN` remains in the enum** for forward compatibility and external adapter parity.
2. **Nexus runtime v1** treats `MODIFY_PLAN` like `FAIL` with error code `MODIFY_PLAN_NOT_SUPPORTED` unless a future `OrchestrationProfile.allow_dynamic_replan=True` is set (default **False**).
3. **Authors MUST use** `AgentDecision.HANDOFF` for runtime graph extension and declarative `ApplicationGraphSpec` for static topology.
4. **Policy hook** at planning boundary (FLOW-11) is the extension point for pre-graph plan mutation - not mid-UAEP `MODIFY_PLAN`.

**Not chosen (v1):**

| Option | Why deferred |
|--------|--------------|
| Full dynamic replan mid-graph | Requires plan versioning, checkpoint invalidation, eval baseline re-bind - Tier-1 scope creep |

## Consequences

- Flow reference §10 and canon §42 decision table stay honest.
- No silent no-op when an agent returns `MODIFY_PLAN`.
- Future v2 can flip `allow_dynamic_replan` without enum break.

## Compliance (FLOW-16 acceptance)

- ADR status → **Accepted**
- `ExecutionInterruptHandler` or UAEP path emits typed failure/trace for unsupported `MODIFY_PLAN`
- Gate test: agent returns `MODIFY_PLAN` → task fails with documented error code
- Flow reference §23 `FLOW-GAP-15` paydown
