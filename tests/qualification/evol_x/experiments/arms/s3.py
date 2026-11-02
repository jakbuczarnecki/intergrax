# © Artur Czarnecki. All rights reserved.

"""EXP-S3: in-flight workflow definition change (WM oracle + real pinning/checkpoint primitives)."""

from __future__ import annotations

from intergrax.applications._shared.profile_resolution import (
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    inherit_child_execution_pinned_revision,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    resolve_profile,
    resolve_revision_for_execution,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    EffectiveProfileRevisionScope,
)
from intergrax.applications.contracts.profile_resolution.errors import (
    EffectiveProfileRevisionConflictError,
)
from intergrax.contracts.execution_identity import mint_execution_id
from testing_support.change_safety.assess.workflow_evolution import (
    WorkflowChange,
    classify_workflow_change,
    unsafe_inflight_adoptions,
)
from tests.qualification.evol_x.experiments._harness import ArmResult, load_oracle

SCENARIO = "S3"
_SCOPE = EffectiveProfileRevisionScope(application_id="evolx.s3", tenant_id="tenant-s3")
_TENANT = "tenant-s3"


def _two_revisions():
    store = InMemoryEffectiveProfileRevisionStore()
    app = ApplicationEnvironmentProfile.lab_defaults(profile_id="evolx.s3")
    r1 = materialize_effective_profile_revision(resolve_profile(app, layers=()), scope=_SCOPE, store=store)
    r2 = materialize_effective_profile_revision(
        resolve_profile(app, layers=()), scope=_SCOPE, store=store, predecessor_revision_id=r1.revision_id
    )
    return store, r1, r2


def run() -> tuple[ArmResult, ArmResult]:
    rows = load_oracle("s3_workflow_inflight")["rows"]
    matrix_ok = all(
        (v := classify_workflow_change(WorkflowChange(**row["change"]))).matrix_id == row["expected_matrix_id"]
        and v.decision.value == row["expected_decision"]
        for row in rows
    )
    # Positive arm: a running execution stays pinned to r1 after r2 exists; its child inherits r1.
    store, r1, r2 = _two_revisions()
    pins = InMemoryEffectiveProfileExecutionPinningStore()
    execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=r1, tenant_id=_TENANT, execution_id=execution_id, pinning_store=pins, revision_store=store
    )
    resolved = resolve_revision_for_execution(
        tenant_id=_TENANT,
        execution_id=execution_id,
        pinning_store=pins,
        revision_store=store,
        scope_application_id=_SCOPE.application_id,
        scope_tenant_id=_SCOPE.tenant_id,
    )
    child = inherit_child_execution_pinned_revision(
        tenant_id=_TENANT, parent_execution_id=execution_id, child_execution_id=mint_execution_id(), pinning_store=pins
    )
    pin_preserved = resolved.revision_id == r1.revision_id != r2.revision_id and child.revision_id == r1.revision_id
    pos_ok = matrix_ok and pin_preserved
    # Negative arm: unsafe in-flight adoption (re-pin running execution to r2) is rejected by production,
    # and the assessor marks every in-place in-flight adoption as non-permissive (unsafe == 0).
    rejected = False
    try:
        pin_effective_profile_revision_for_execution(
            revision=r2, tenant_id=_TENANT, execution_id=execution_id, pinning_store=pins, revision_store=store
        )
    except EffectiveProfileRevisionConflictError:
        rejected = True
    unsafe = len(unsafe_inflight_adoptions())
    neg_ok = rejected and unsafe == 0
    return (
        ArmResult(SCENARIO, "positive", pos_ok, f"matrix_rows={len(rows)} pin_preserved={pin_preserved}"),
        ArmResult(SCENARIO, "negative", neg_ok, f"production_rejects_repin={rejected} unsafe_adoptions={unsafe}"),
    )
