# © Artur Czarnecki. All rights reserved.

"""Tier-3 evaluation wiring (Phase EVAL-1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications._shared.evaluation_runtime_bridge import (
    EvaluationWiringOptions,
    resolve_evaluation_wiring_options,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    EvaluationProfile,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    FileOnlineEvaluationRegistry,
    OnlineEvaluationRegistry,
    TenantScopedOnlineEvaluationRegistry,
    default_online_evaluation_registry,
)
from intergrax.runtime.architecture.runtime_governance_bridge import (
    RuntimeArchitectureGovernanceBridge,
)


def _resolve_evaluation_registry(profile: EvaluationProfile) -> FileOnlineEvaluationRegistry | None:
    if not profile.online_registry_enabled:
        return None
    if profile.registry_path is not None:
        return FileOnlineEvaluationRegistry(profile.registry_path)
    return default_online_evaluation_registry()


@dataclass(frozen=True, slots=True)
class ApplicationEvaluationWiring:
    """Resolved evaluation artifacts for a Tier-3 host."""

    profile: EvaluationProfile
    options: EvaluationWiringOptions
    registry: OnlineEvaluationRegistry | None
    governance_bridge: RuntimeArchitectureGovernanceBridge | None
    domain_fragments: dict[str, Any]
    tenant_id: str | None = None


def wire_application_evaluation(
    env: ApplicationEnvironmentProfile,
    *,
    tenant_id: str | None = None,
) -> ApplicationEvaluationWiring:
    """Materialize evaluation registry and governance bridge from environment profile.

    With ``tenant_id`` the registry is bound to that tenant (PDC-D5-R015): the host can
    only write and read its own tenant's observations. Without it the registry is the
    legacy unscoped one, which must not be shared between tenants.
    """
    profile = env.evaluation_profile
    options = resolve_evaluation_wiring_options(profile)
    file_registry = _resolve_evaluation_registry(profile)
    registry: OnlineEvaluationRegistry | None = file_registry
    if file_registry is not None and tenant_id is not None:
        registry = TenantScopedOnlineEvaluationRegistry(file_registry, tenant_id)
    governance_bridge = (
        RuntimeArchitectureGovernanceBridge(evaluation_registry=registry)
        if profile.shadow_eval_enabled or profile.online_registry_enabled
        else None
    )
    return ApplicationEvaluationWiring(
        profile=profile,
        options=options,
        registry=registry,
        governance_bridge=governance_bridge,
        tenant_id=tenant_id,
        domain_fragments={
            "evaluation_governance": {
                "shadow_eval_enabled": profile.shadow_eval_enabled,
                "online_registry_enabled": profile.online_registry_enabled,
                "offline_eval_runner_enabled": profile.offline_eval_runner_enabled,
                "trend_comparison_enabled": profile.trend_comparison_enabled,
                "require_baseline_for_release": profile.require_baseline_for_release,
                "evaluation_assets_ref": profile.evaluation_assets_ref,
            },
        },
    )
