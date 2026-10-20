# © Artur Czarnecki. All rights reserved.

"""Package B — sole in-repo owner for operator production control-plane governance (PB-S01)."""

from __future__ import annotations

from pathlib import Path

from intergrax.agent_distribution.control_plane_governance import (
    StaticApplicationEnvironmentTenantResolver,
)
from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_host_control_plane_policy_bundle,
    build_reference_production_lifecycle_policy_bundle,
)
from intergrax.applications._shared.production_control_plane_governance_wiring import (
    ProductionControlPlaneGovernance,
)
from intergrax.applications._shared.settings_loader import EnvReader
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.runtime_policy_bundle import ImmutableRuntimePolicyBundle
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_policy import (
    bundle_backed_control_plane_mutation_evaluator,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

_ENV_PREFIX = "INTERGRAX_OPERATOR_PRODUCTION_"
_FORBIDDEN_IN_REPO_BUNDLE_IDS = frozenset(
    {
        build_reference_production_lifecycle_policy_bundle().bundle_id,
        build_harness_host_control_plane_policy_bundle().bundle_id,
    },
)


class PackageBOperatorProductionGovernanceConfigurationError(ValueError):
    """Operator production governance configuration is missing or not authorized."""


def _load_operator_mutation_policy_bundle_from_env(
    *,
    env: EnvReader | None = None,
) -> ImmutableRuntimePolicyBundle:
    reader = env or EnvReader(_ENV_PREFIX)
    bundle_id = reader.optional_str("MUTATION_POLICY_BUNDLE_ID")
    if not bundle_id:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"{_ENV_PREFIX}MUTATION_POLICY_BUNDLE_ID is required for Package B strict serving",
        )
    path_raw = reader.optional_str("MUTATION_POLICY_BUNDLE_PATH")
    if not path_raw:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"{_ENV_PREFIX}MUTATION_POLICY_BUNDLE_PATH is required "
            "(operator-supplied immutable policy artifact; not in-repo)",
        )
    if bundle_id in _FORBIDDEN_IN_REPO_BUNDLE_IDS:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"mutation policy bundle_id {bundle_id!r} is lab/reference-only; "
            "operator-approved production bundle required",
        )
    path = Path(path_raw)
    if not path.is_file():
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"mutation policy bundle path does not exist: {path}",
        )
    try:
        bundle = ImmutableRuntimePolicyBundle.model_validate_json(
            path.read_text(encoding="utf-8"),
        ).with_canonical_digest()
    except (OSError, ValueError) as exc:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"mutation policy bundle at {path} is invalid: {exc}",
        ) from exc
    if bundle.bundle_id != bundle_id:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"MUTATION_POLICY_BUNDLE_ID {bundle_id!r} does not match "
            f"artifact bundle_id {bundle.bundle_id!r}",
        )
    return bundle


def load_package_b_operator_production_control_plane_governance_from_env(
    *,
    env: EnvReader | None = None,
) -> ProductionControlPlaneGovernance:
    """Build ``ProductionControlPlaneGovernance`` from operator-supplied env configuration."""
    reader = env or EnvReader(_ENV_PREFIX)
    bundle = _load_operator_mutation_policy_bundle_from_env(env=reader)
    tenant_id = reader.optional_str("GOVERNANCE_TENANT_ID")
    if not tenant_id:
        raise PackageBOperatorProductionGovernanceConfigurationError(
            f"{_ENV_PREFIX}GOVERNANCE_TENANT_ID is required",
        )
    service_id = reader.str("GOVERNANCE_SERVICE_ID", default="gca-strict-production")
    principal = RequestIdentity(
        tenant_id=tenant_id,
        user_id=service_id,
        principal_type=PrincipalType.SERVICE,
        auth_subject=service_id,
    )
    boundary = ControlPlaneMutationAuthorizationBoundary(
        evaluator=bundle_backed_control_plane_mutation_evaluator(
            RuntimePolicyBundleEvaluator(bundle),
        ),
    )
    return ProductionControlPlaneGovernance(
        principal=principal,
        mutation_authorization_boundary=boundary,
        environment_tenant_resolver=StaticApplicationEnvironmentTenantResolver(
            tenant_id=tenant_id,
        ),
    )


__all__ = [
    "PackageBOperatorProductionGovernanceConfigurationError",
    "load_package_b_operator_production_control_plane_governance_from_env",
]
