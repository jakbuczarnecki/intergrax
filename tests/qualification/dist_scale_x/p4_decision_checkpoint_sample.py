# © Artur Czarnecki. All rights reserved.

"""Minimal decision checkpoint for P4 qualification (SX-F03)."""

from __future__ import annotations


def build_p4_qualification_decision_checkpoint(
    tenant_id: str = "tenant-a",
    *,
    subject: str = "wl5",
    decision_id: str | None = None,
):
    from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
    from intergrax.contracts.decision_finalization import (
        decision_finalization_key,
        initial_decision_finalize_guard,
    )
    from intergrax.contracts.decision_identity import (
        DecisionExecutionLineage,
        DecisionIdentity,
        DecisionScope,
        initial_decision_version,
        mint_decision_id,
        validate_decision_id,
    )
    from intergrax.contracts.decision_lifecycle import initial_decision_lifecycle_state
    from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
    from intergrax.contracts.decision_revision import decision_revision_checkpoint_state
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    resolved_decision_id = (
        validate_decision_id(decision_id) if decision_id is not None else mint_decision_id()
    )
    identity = DecisionIdentity(
        decision_id=resolved_decision_id,
        version=initial_decision_version(),
        scope=DecisionScope(namespace="dist-scale-x-p4", subject=subject),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    revision = decision_revision_checkpoint_state(
        proposal_ref=DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        revision_count=1,
        max_revisions=3,
    )
    lifecycle = initial_decision_lifecycle_state(identity)
    return decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
