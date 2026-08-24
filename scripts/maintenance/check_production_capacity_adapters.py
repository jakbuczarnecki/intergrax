#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-30.4 — Celery/K8s production-scale adapters."""

from __future__ import annotations

import sys

from intergrax.applications._shared.production_capacity_governance_wiring import (
    build_production_capacity_governance,
)
from intergrax.applications._shared.production_capacity_wiring import resolve_production_capacity_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


class _HarnessProductionCapacityPolicy:
    """Explicit gate-only allow policy for production adapter probe evidence."""

    def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
        del request
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="harness_production_capacity_probe",
            policy_rule_id="harness.production_capacity.scale_probe",
        )


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    governance = build_production_capacity_governance(
        env,
        mutation_authorization_boundary=ControlPlaneMutationAuthorizationBoundary(
            evaluator=_HarnessProductionCapacityPolicy(),
        ),
    )
    wiring = resolve_production_capacity_wiring(env, governance=governance)
    if not wiring.enabled:
        print("product host must enable production capacity adapters", file=sys.stderr)
        return 1
    if wiring.adapters is None or not wiring.probe_passed:
        print("production capacity adapter probe failed", file=sys.stderr)
        return 1
    print("OK: production Celery/K8s capacity adapters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
