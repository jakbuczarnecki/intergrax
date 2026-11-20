#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.4 — Celery/K8s production-scale adapters (fail-closed posture, DP-29)."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from intergrax.applications._shared.control_plane_composition import ControlPlaneCompositionError
from intergrax.applications._shared.production_capacity_governance_wiring import (
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import (
    CapacityQualification,
    resolve_production_capacity_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.capacity.production_adapters import CapacityBackendPosture
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

_CLUSTER_URL_ENV = "INTERGRAX_KUBERNETES_URL"
_MISSING_BOUNDARY_CODE = "ECP_BLOCKED_MISSING_BOUNDARY"


class _AllowingEvaluator:
    """Inline stand-in policy evaluator: always allows (no test import, shape of the evaluator protocol)."""

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        del request
        return PolicyDecision(action=PolicyAction.ALLOW, reason="ok")


@contextmanager
def _cluster_url_removed() -> Iterator[None]:
    """Remove the cluster URL for the check and restore the caller's value afterwards."""
    previous = os.environ.pop(_CLUSTER_URL_ENV, None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ[_CLUSTER_URL_ENV] = previous


def _check_missing_boundary(env: ApplicationEnvironmentProfile) -> str | None:
    governance = build_production_capacity_governance(env)
    try:
        resolve_production_capacity_wiring(env, governance=governance)
    except ControlPlaneCompositionError as exc:
        if exc.blocker_code != _MISSING_BOUNDARY_CODE:
            return f"missing boundary refused with {exc.blocker_code!r}, expected {_MISSING_BOUNDARY_CODE!r}"
        return None
    return "production capacity wiring must refuse a missing mutation authorization boundary"


def _check_default_posture_without_cluster(env: ApplicationEnvironmentProfile) -> str | None:
    governance = build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowingEvaluator()),
    )
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    if not wiring.enabled:
        return "product host must enable production capacity adapters"
    if wiring.adapters is not None or wiring.probe_passed:
        return "production capacity wiring must build no adapters and pass no probe without a cluster"
    if wiring.qualification is not CapacityQualification.UNQUALIFIED_NO_CLUSTER:
        return f"production capacity wiring without a cluster must be unqualified, got {wiring.qualification!r}"
    return None


def _check_ci_deterministic_posture(env: ApplicationEnvironmentProfile) -> str | None:
    governance = build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowingEvaluator()),
    )
    wiring = resolve_production_capacity_wiring(
        env,
        governance=governance,
        backend_posture=CapacityBackendPosture.CI_DETERMINISTIC,
    )
    if not wiring.enabled or wiring.adapters is None:
        return "CI deterministic posture must build the in-memory adapters"
    if wiring.probe_passed:
        return "an in-memory capacity backend must never be reported as a passed probe"
    if wiring.qualification is not CapacityQualification.CI_DETERMINISTIC_ONLY:
        return f"CI deterministic posture must be qualified CI_DETERMINISTIC_ONLY, got {wiring.qualification!r}"
    return None


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    with _cluster_url_removed():
        for check in (
            _check_missing_boundary,
            _check_default_posture_without_cluster,
            _check_ci_deterministic_posture,
        ):
            failure = check(env)
            if failure is not None:
                print(failure, file=sys.stderr)
                return 1
    print(
        "OK: production Celery/K8s capacity adapters "
        "(refuse a missing boundary; unqualified without a cluster; in-memory never a passed probe)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
