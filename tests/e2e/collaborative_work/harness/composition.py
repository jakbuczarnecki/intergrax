# © Artur Czarnecki. All rights reserved.

"""Public-boundary composition from durable Collaborative Work repositories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.collaborative_work.persistence_provider import (
    resolve_collaborative_work_repositories,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)

from tests.e2e.collaborative_work.harness.runtime_policy import MutableRuntimePolicyEvaluator


@dataclass(frozen=True, slots=True)
class MultiplayerE2EContext:
    """Provider-neutral runtime assembled from public platform contracts."""

    profile: IntegrationProfile
    bundle: CollaborativeWorkRepositories
    boundary: MeaningfulSideEffectAuthorizationBoundary
    runtime_policy: MutableRuntimePolicyEvaluator


def build_authorization_boundary(
    bundle: CollaborativeWorkRepositories,
    runtime_policy: MutableRuntimePolicyEvaluator,
    *,
    clock: Callable[[], datetime],
) -> MeaningfulSideEffectAuthorizationBoundary:
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=bundle.operation_profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=bundle.membership,
            delegation_repository=bundle.delegation,
            principal_authority_repository=bundle.principal_authority,
            clock=clock,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(bundle.policy),
        runtime_policy_evaluator=runtime_policy,
    )
    return MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)


def open_multiplayer_e2e_context(
    profile: IntegrationProfile,
    runtime_policy: MutableRuntimePolicyEvaluator,
    *,
    clock: Callable[[], datetime],
) -> MultiplayerE2EContext:
    bundle = resolve_collaborative_work_repositories(profile)
    boundary = build_authorization_boundary(bundle, runtime_policy, clock=clock)
    return MultiplayerE2EContext(
        profile=profile,
        bundle=bundle,
        boundary=boundary,
        runtime_policy=runtime_policy,
    )


def reopen_multiplayer_e2e_context(
    profile: IntegrationProfile,
    runtime_policy: MutableRuntimePolicyEvaluator,
    *,
    clock: Callable[[], datetime],
) -> MultiplayerE2EContext:
    """Independent bundle/connection — durability proof, not same-object reopen."""
    return open_multiplayer_e2e_context(profile, runtime_policy, clock=clock)
