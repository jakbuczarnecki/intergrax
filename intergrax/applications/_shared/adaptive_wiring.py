# © Artur Czarnecki. All rights reserved.

"""Tier-3 adaptive profile wiring (Phase W-ADAPT-4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intergrax.applications.contracts.environment_profile import (
    AdaptiveProfile,
    ApplicationEnvironmentProfile,
)
from intergrax.runtime.adaptive.adaptation_executor import AdaptationExecutor
from intergrax.runtime.adaptive.policy_learning_approval import (
    InMemoryPolicyLearningApprovalStore,
    PolicyLearningApprovalStore,
)
from intergrax.runtime.adaptive.profile_lifecycle import ProfileVersionLifecycleManager
from intergrax.runtime.adaptive.profile_pointer_store import (
    ProfileActivePointerStore,
    SQLiteProfileActivePointerStore,
)
from intergrax.runtime.adaptive.profile_version_store import (
    ProfileVersionStore,
    SQLiteProfileVersionStore,
)
from intergrax.runtime.adaptive.signal_collector import SignalCollector
from intergrax.runtime.adaptive.signal_store import SQLiteSignalStore, default_signal_store_path
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge


@dataclass(frozen=True, slots=True)
class ApplicationAdaptiveWiring:
    """Resolved adaptive harness artifacts for a Tier-3 host."""

    profile: AdaptiveProfile
    profile_version_store: ProfileVersionStore | None
    pointer_store: ProfileActivePointerStore | None
    lifecycle_manager: ProfileVersionLifecycleManager | None
    adaptation_executor: AdaptationExecutor | None
    signal_collector: SignalCollector | None
    governance_bridge: RuntimeArchitectureGovernanceBridge | None
    approval_store: PolicyLearningApprovalStore | None
    domain_fragments: dict[str, Any]


def wire_adaptive_profile(
    env: ApplicationEnvironmentProfile,
    *,
    evaluation_governance_bridge: RuntimeArchitectureGovernanceBridge | None = None,
) -> ApplicationAdaptiveWiring:
    """Materialize adaptive stores, executor, and signal collector from environment profile."""
    profile = env.adaptive_profile
    if not profile.enabled:
        return ApplicationAdaptiveWiring(
            profile=profile,
            profile_version_store=None,
            pointer_store=None,
            lifecycle_manager=None,
            adaptation_executor=None,
            signal_collector=None,
            governance_bridge=evaluation_governance_bridge,
            approval_store=None,
            domain_fragments={"adaptive_enabled": False, "adaptive_mode": profile.mode},
        )

    version_store = SQLiteProfileVersionStore(db_path=profile.profile_versions_db_path)
    pointer_store = SQLiteProfileActivePointerStore(db_path=profile.profile_pointers_db_path)
    lifecycle_manager = ProfileVersionLifecycleManager(version_store)
    approval_store = InMemoryPolicyLearningApprovalStore()
    executor = AdaptationExecutor(
        profile_store=version_store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle_manager,
        approval_store=approval_store,
    )

    signal_db_path = profile.signal_store_path or default_signal_store_path()
    signal_store = SQLiteSignalStore(db_path=signal_db_path)
    signal_collector = SignalCollector(signal_store, application_id=env.profile_id)

    governance_bridge = evaluation_governance_bridge or RuntimeArchitectureGovernanceBridge()

    return ApplicationAdaptiveWiring(
        profile=profile,
        profile_version_store=version_store,
        pointer_store=pointer_store,
        lifecycle_manager=lifecycle_manager,
        adaptation_executor=executor,
        signal_collector=signal_collector,
        governance_bridge=governance_bridge,
        approval_store=approval_store,
        domain_fragments={
            "adaptive_enabled": True,
            "adaptive_mode": profile.mode,
            "enabled_loops": [loop.value for loop in profile.enabled_loops],
            "canary_traffic_percent": profile.canary_traffic_percent,
        },
    )
