# © Artur Czarnecki. All rights reserved.

"""CRIT-V-6.2 critic assembly resolver tests."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.critic_assembly_resolver import (
    CriticAssemblyError,
    assert_critic_assembly_valid,
    validate_critic_wiring,
)
from intergrax.applications._shared.critic_wiring import wire_application_critic
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CriticProfile,
    CriticVerificationScopes,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_validate_critic_wiring_requires_rubric_for_semantic_judge() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.assembly")
    env.critic_profile = CriticProfile(
        semantic_judge_enabled=True,
        scopes=CriticVerificationScopes(node_partial=True),
    )
    wiring = wire_application_critic(env)
    result = validate_critic_wiring(wiring, env)
    assert not result.valid
    assert any("default_rubric_ref" in error for error in result.errors)


def test_validate_critic_wiring_requires_l1_client_for_semantic_judge() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.assembly.l1")
    env.critic_profile = CriticProfile(
        semantic_judge_enabled=True,
        default_rubric_ref="rubric.default",
        scopes=CriticVerificationScopes(node_partial=True),
    )
    wiring = wire_application_critic(env)
    result = validate_critic_wiring(wiring, env, l1_client=None)
    assert not result.valid
    assert any("L1 eval tool client" in error for error in result.errors)


def test_assert_critic_assembly_valid_passes_lab_defaults() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.assembly.ok")
    wiring = wire_application_critic(env)
    assert_critic_assembly_valid(wiring, env)


def test_assert_critic_assembly_raises_on_invalid_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="critic.assembly.fail")
    env.critic_profile = CriticProfile(
        require_critic_on_completion=True,
        scopes=CriticVerificationScopes(node_partial=False, graph_final=False),
    )
    wiring = wire_application_critic(env)
    with pytest.raises(CriticAssemblyError):
        assert_critic_assembly_valid(wiring, env)
