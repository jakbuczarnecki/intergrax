# © Artur Czarnecki. All rights reserved.

"""Explicit production-capacity control-plane policy bundle composition (GR-12 ECP)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.runtime_policy_bundle import (
    ImmutableRuntimePolicyBundle,
    PolicyBundleRule,
    build_immutable_runtime_policy_bundle,
)
from intergrax.runtime.capacity.control_plane_governance import (
    MUTATION_TYPE_SCALE_CELERY_WORKERS,
    MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
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

_PRODUCTION_CAPACITY_CONTROL_PLANE_BUNDLE_ID = "production_capacity.control_plane"
_POLICY_ISSUED_AT = datetime(2026, 8, 24, 0, 0, 0, tzinfo=timezone.utc)


def build_production_capacity_control_plane_policy_bundle() -> ImmutableRuntimePolicyBundle:
    """Immutable pack for governed ECP scale mutations on product hosts."""
    return build_immutable_runtime_policy_bundle(
        bundle_id=_PRODUCTION_CAPACITY_CONTROL_PLANE_BUNDLE_ID,
        version="1.0.0",
        rules=(
            PolicyBundleRule(
                rule_id="production_capacity.ecp.scale_k8s_deployment",
                description="Explicit allow for governed K8s deployment scale",
                match_action=MUTATION_TYPE_SCALE_K8S_DEPLOYMENT,
                effect="allow",
            ),
            PolicyBundleRule(
                rule_id="production_capacity.ecp.scale_celery_workers",
                description="Explicit allow for governed Celery worker scale",
                match_action=MUTATION_TYPE_SCALE_CELERY_WORKERS,
                effect="allow",
            ),
        ),
        issued_at=_POLICY_ISSUED_AT,
    )


def build_production_capacity_mutation_boundary(
    env: ApplicationEnvironmentProfile,
) -> ControlPlaneMutationAuthorizationBoundary:
    """Canonical explicit policy authority for PRODUCT ECP mutations."""
    del env
    bundle = build_production_capacity_control_plane_policy_bundle()
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )


__all__ = [
    "build_production_capacity_control_plane_policy_bundle",
    "build_production_capacity_mutation_boundary",
]
