# © Artur Czarnecki. All rights reserved.

"""Explicit harness-host control-plane policy bundle composition (TASK-CPM-1C)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
    MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
)
from intergrax.applications._shared.task_control_governance import (
    MUTATION_TYPE_CANCEL_TASK_EXECUTION,
    MUTATION_TYPE_RESUME_TASK_EXECUTION,
    MUTATION_TYPE_SET_TASK_AUTONOMY,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.runtime_policy_bundle import (
    ImmutableRuntimePolicyBundle,
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    bundle_backed_control_plane_mutation_evaluator,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

_HARNESS_CONTROL_PLANE_BUNDLE_ID = "harness.control_plane"
_REFERENCE_PRODUCTION_CONTROL_PLANE_BUNDLE_ID = "reference_production.control_plane"
_POLICY_ISSUED_AT = datetime(2026, 8, 24, 0, 0, 0, tzinfo=timezone.utc)


def build_harness_host_control_plane_policy_bundle() -> ImmutableRuntimePolicyBundle:
    """Immutable pack for general harness-host control-plane mutations."""
    return build_immutable_runtime_policy_bundle(
        bundle_id=_HARNESS_CONTROL_PLANE_BUNDLE_ID,
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="harness.task_control.cancel_task_execution",
                description="Explicit allow for governed cooperative task cancel",
                match_action=MUTATION_TYPE_CANCEL_TASK_EXECUTION,
                effect="allow",
            ),
            PolicyBundleRule(
                rule_id="harness.task_control.set_task_autonomy",
                description="Explicit allow for governed active task autonomy mutation",
                match_action=MUTATION_TYPE_SET_TASK_AUTONOMY,
                effect="allow",
            ),
            PolicyBundleRule(
                rule_id="harness.task_control.resume_task_execution",
                description="Explicit allow for governed operator checkpoint resume",
                match_action=MUTATION_TYPE_RESUME_TASK_EXECUTION,
                effect="allow",
            ),
        ),
        issued_at=_POLICY_ISSUED_AT,
    )


def build_reference_production_lifecycle_policy_bundle() -> ImmutableRuntimePolicyBundle:
    """Immutable pack for reference production lifecycle service mutations only."""
    return build_immutable_runtime_policy_bundle(
        bundle_id=_REFERENCE_PRODUCTION_CONTROL_PLANE_BUNDLE_ID,
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="reference_production.lifecycle_admit",
                description="Reference production runtime revision admission",
                match_action=MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
                effect="allow",
            ),
            PolicyBundleRule(
                rule_id="reference_production.lifecycle_activate",
                description="Reference production runtime revision activation",
                match_action=MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
                effect="allow",
            ),
        ),
        issued_at=_POLICY_ISSUED_AT,
    )


def build_harness_control_plane_mutation_boundary(
    env: ApplicationEnvironmentProfile,
) -> ControlPlaneMutationAuthorizationBoundary:
    """Canonical explicit policy authority for PRODUCT harness host mutations."""
    del env
    bundle = build_harness_host_control_plane_policy_bundle()
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )


def build_reference_production_control_plane_mutation_boundary(
    env: ApplicationEnvironmentProfile,
) -> ControlPlaneMutationAuthorizationBoundary:
    """Narrow lifecycle-only policy authority for reference production service."""
    del env
    bundle = build_reference_production_lifecycle_policy_bundle()
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )


__all__ = [
    "build_harness_control_plane_mutation_boundary",
    "build_harness_host_control_plane_policy_bundle",
    "build_reference_production_control_plane_mutation_boundary",
    "build_reference_production_lifecycle_policy_bundle",
]
