# MP-5F-B3A-C0 — Canonical workspace authority propagation

**Status:** design certified · C2 production propagation certified (see `test_b3a_c2_workspace_authority_propagation.py`).

Workspace authority propagates through typed runtime/step/assembly contracts (`RuntimeExecutionContext.workspace_id`, `AgentStepContext.workspace_id`, `ContextAssemblyRequest.workspace_id`). Metadata is not an authority source.

## Canonical semantic owner

| Semantic | Canonical owner | Public contract | Evidence |
| --- | --- | --- | --- |
| Workspace existence, membership, collaboration policy | Collaborative Work | `intergrax/contracts/collaborative_work.py` | CW-INV-02; domain arch `COLLABORATIVE_WORK.md` |
| Trusted execution admission workspace binding | Governed execution / runtime intake | `RootExecutionLaunchRequest`, `CanonicalExecutionIntakeRequest`, `RuntimeExecutionAdmissionRequest` | Non-empty `workspace_id` after admission |
| Task / Nexus turn intake transport | Task envelope + runtime request | `TaskEnvelope.workspace_id`, `RuntimeRequest.workspace_id` | `Task.to_envelope` / `to_runtime_request` |
| ACP session transport (optional) | Agent run entry | `AgentRunRequest.workspace_id` → `AgentStepContext.workspace_id` | Host sets after admission; not metadata-derived |
| Context assembly consumer transport | Context Engineering | `ContextAssemblyRequest.workspace_id` (optional) | CE-1.2 assembly input |

Transport does not transfer ownership: CE and UCL consume `workspace_id`; Collaborative Work remains semantic authority.

## Propagation path

```text
Collaborative Work (workspace truth)
  → admission / launch (RootExecutionLaunchRequest, CanonicalExecutionIntakeRequest)
  → TaskEnvelope.workspace_id / RuntimeRequest.workspace_id / AgentRunRequest.workspace_id
  → RuntimeExecutionContext.workspace_id / AgentStepContext.workspace_id
  → ContextAssemblyRequest.workspace_id
  → ContextEngine.assemble
  → (C1) UclArtifactOwnershipScope from request fields only
```

`workspace_id` MUST NOT be derived from `context_scope_id`, Nexus private state, metadata dictionaries, or `context_scope` session bindings.

## Optionality

- Platform-wide: `ContextAssemblyRequest.workspace_id` is **optional**.
- Workspace-bound / UCL artifact optimization (C1): **required**; fail-closed when missing (`resolve_ucl_artifact_ownership_scope` returns `None`).

## Tenant invariant

When `workspace_id` is present on assembly request, it must belong to the same `tenant_id` on that request. Cross-tenant membership validation remains owned by Collaborative Work / admission; CE does not mint workspace identity.

## C1 handoff

C1 may build:

```python
UclArtifactOwnershipScope(
    tenant_id=request.tenant_id,
    workspace_id=request.workspace_id.strip(),
)
```

only when `request.workspace_id` is set and artifact optimization is required.

## Consumer boundaries

- UCL repository redesign: out of scope (C0).
- No Collaborative Work internal types in UCL contracts.
- No Nexus-private workspace API as public contract.
