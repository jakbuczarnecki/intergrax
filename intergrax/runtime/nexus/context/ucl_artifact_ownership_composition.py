# © Artur Czarnecki. All rights reserved.

"""Composition boundary: assembly request scope → UCL artifact ownership (MP-5F-B3A-C1)."""

from __future__ import annotations

from intergrax.context.contracts import ContextAssemblyRequest
from intergrax.context.planning import ContextPlan
from intergrax.runtime.context_lifecycle.contracts import UclArtifactOwnershipScope


def resolve_ucl_artifact_ownership_scope(
    request: ContextAssemblyRequest,
    *,
    context_plan: ContextPlan,
) -> UclArtifactOwnershipScope | None:
    """Map canonical assembly request workspace to UCL ownership when optimization runs."""
    if not context_plan.optimization_required:
        return None
    workspace_id = request.workspace_id
    if workspace_id is None or not workspace_id.strip():
        return None
    return UclArtifactOwnershipScope(
        tenant_id=request.tenant_id,
        workspace_id=workspace_id.strip(),
    )
