# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime validation for Decision governance material on side-effect boundaries (GR-6)."""

from __future__ import annotations

from intergrax.contracts.canonical_inner_governance import CanonicalInnerGovernanceViolation
from intergrax.contracts.decision_governance_material import (
    DecisionGovernanceMaterialMismatchError,
    DecisionGovernanceMaterialRef,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest


def assert_decision_governance_material_bound(
    request: MeaningfulSideEffectRequest,
) -> None:
    """Fail closed when typed decision material diverges from side-effect identity."""
    material = request.decision_governance_material
    if material is None:
        return
    if request.tenant_id is not None and request.tenant_id != material.tenant_id:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material tenant_id does not match side effect",
        )
    if request.task_id != material.task_id:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material task_id does not match side effect",
        )
    if request.run_id != material.run_id:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material run_id does not match side effect",
        )
    if request.attempt_id != material.attempt_id:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material attempt_id does not match side effect",
        )
    if request.execution_id != material.execution_id:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material execution_id does not match side effect",
        )
    if request.action != material.bound_action_kind:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material action kind does not match side effect",
        )
    if request.resource is None:
        raise CanonicalInnerGovernanceViolation(
            reason="decision governance material requires side effect resource identity",
        )
    if request.resource != material.bound_action_subject:
        raise CanonicalInnerGovernanceViolation(
            reason=(
                "decision governance material bound_action_subject "
                "does not match side effect resource"
            ),
        )
    _assert_material_internally_consistent(material)


def _assert_material_internally_consistent(material: DecisionGovernanceMaterialRef) -> None:
    try:
        _ = material.bound_action
        _ = material.execution_lineage
    except (TypeError, ValueError) as exc:
        raise CanonicalInnerGovernanceViolation(
            reason=f"decision governance material invalid: {exc}",
        ) from exc


def revalidate_decision_governance_material_digest(
    *,
    material: DecisionGovernanceMaterialRef,
    expected_digest: str,
) -> None:
    """Reject tampered decision material digest at enforcement time."""
    if material.decision_material_digest != expected_digest:
        raise DecisionGovernanceMaterialMismatchError(
            "decision governance material digest mismatch",
        )


__all__ = [
    "assert_decision_governance_material_bound",
    "revalidate_decision_governance_material_digest",
]
