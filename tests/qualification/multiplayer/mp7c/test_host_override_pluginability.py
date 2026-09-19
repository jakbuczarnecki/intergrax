# © Artur Czarnecki. All rights reserved.

"""MP-7C — explicit override / pluginability at host composition layer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.runtime_policy import PolicyAction
from tests.qualification.multiplayer.mp7b.consumer import (
    Tier3MultiplayerAuthorizationDeniedError,
    Tier3MultiplayerConsumer,
    build_enforcement_request_from_lkw_scope_fixture,
)
from tests.qualification.multiplayer.mp7b.custom_ports import (
    AllowingAuthorizationPort,
    DenyingAuthorizationPort,
)
from tests.qualification.multiplayer.mp7c.host_composition import (
    empty_in_memory_repositories,
    resolve_host_wiring,
    strict_host_environment,
)

pytestmark = pytest.mark.unit


def test_explicit_custom_port_beats_platform_default() -> None:
    custom = AllowingAuthorizationPort()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        wiring = resolve_host_wiring(
            strict_host_environment(),
            explicit=custom,
            collaborative_work_repositories=empty_in_memory_repositories(),
        )
    resolve_mock.assert_not_called()
    assert wiring.authorization_port is custom
    assert wiring.owned_collaborative_work_persistence is None


def test_explicit_override_does_not_materialize_cw_persistence() -> None:
    custom = DenyingAuthorizationPort()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        with patch(
            "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring._build_port_from_materialized_repositories",
        ) as build_mock:
            wiring = resolve_host_wiring(
                strict_host_environment(),
                explicit=custom,
            )
    resolve_mock.assert_not_called()
    build_mock.assert_not_called()
    assert wiring.authorization_port is custom
    assert wiring.owned_collaborative_work_persistence is None


def test_same_consumer_works_with_host_default_and_explicit_override() -> None:
    request = build_enforcement_request_from_lkw_scope_fixture(
        tenant_id="mp7c-tenant",
        workspace_id="mp7c-workspace",
        acting_principal_id="mp7c-principal",
        operation_id="mp7c.plugin.proof",
    )
    # Same consumer type for host-injected custom ports (default path covered elsewhere).
    allow_wiring = resolve_host_wiring(
        strict_host_environment(),
        explicit=AllowingAuthorizationPort(),
    )
    deny_wiring = resolve_host_wiring(
        strict_host_environment(),
        explicit=DenyingAuthorizationPort(),
    )
    assert allow_wiring.authorization_port is not None
    assert deny_wiring.authorization_port is not None
    assert isinstance(allow_wiring.authorization_port, MeaningfulSideEffectAuthorizationPort)
    assert isinstance(deny_wiring.authorization_port, MeaningfulSideEffectAuthorizationPort)
    allow_consumer = Tier3MultiplayerConsumer(authorization=allow_wiring.authorization_port)
    deny_consumer = Tier3MultiplayerConsumer(authorization=deny_wiring.authorization_port)
    allowed = allow_consumer.request_authorized_side_effect(request)
    assert allowed.permitted is True
    assert allowed.decision.action is PolicyAction.ALLOW
    with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
        deny_consumer.request_authorized_side_effect(request)


def test_explicit_override_precedence_over_injected_repositories() -> None:
    custom = AllowingAuthorizationPort()
    bundle = empty_in_memory_repositories()
    wiring = resolve_host_wiring(
        strict_host_environment(),
        explicit=custom,
        collaborative_work_repositories=bundle,
    )
    assert wiring.authorization_port is custom
    assert wiring.owned_collaborative_work_persistence is None
