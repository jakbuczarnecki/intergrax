# © Artur Czarnecki. All rights reserved.

"""MP-7B — Tier-3 contract consumption, authorization, and pluginability proofs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.collaborative_work import PrincipalKind
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)
from intergrax.contracts.runtime_policy import PolicyAction
from tests.qualification.multiplayer.mp7b.composition import (
    compose_allow_platform_port,
    compose_deny_platform_port,
)
from tests.qualification.multiplayer.mp7b.consumer import (
    Tier3MultiplayerAuthorizationDeniedError,
    Tier3MultiplayerAuthorizationFailedError,
    Tier3MultiplayerConsumer,
    build_enforcement_request_from_lkw_scope_fixture,
    map_lkw_actor_fixture_to_principal,
)
from tests.qualification.multiplayer.mp7b.custom_ports import (
    AllowingAuthorizationPort,
    DenyingAuthorizationPort,
    ExplodingAuthorizationPort,
)
from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution

pytestmark = pytest.mark.unit

_CONSUMER_MODULE = Path(__file__).resolve().parent / "consumer.py"


def test_principal_contract_from_lkw_actor_fixture() -> None:
    principal = map_lkw_actor_fixture_to_principal(
        lkw_actor_id="lkw-user-42",
        tenant_id="tenant-lkw",
        principal_kind=PrincipalKind.HUMAN,
    )
    assert principal.principal_id == "lkw-user-42"
    assert principal.tenant_id == "tenant-lkw"
    assert principal.principal_kind is PrincipalKind.HUMAN


def test_workspace_scope_request_is_public_contract_not_authority() -> None:
    request = build_enforcement_request_from_lkw_scope_fixture(
        tenant_id="tenant-lkw",
        workspace_id="ws-product-1",
        acting_principal_id="lkw-user-42",
        operation_id="mp7b.scope.proof",
    )
    assert request.tenant_id == "tenant-lkw"
    assert request.workspace_id == "ws-product-1"
    # Scope IDs are request fields — consumer must not treat them as authorization proof.
    consumer = Tier3MultiplayerConsumer(authorization=DenyingAuthorizationPort())
    with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
        consumer.request_authorized_side_effect(request)


def test_custom_port_allow_and_deny_same_consumer() -> None:
    request = build_enforcement_request_from_lkw_scope_fixture(
        tenant_id="t1",
        workspace_id="w1",
        acting_principal_id="p1",
        operation_id="mp7b.plugin.proof",
    )
    allow_consumer = Tier3MultiplayerConsumer(authorization=AllowingAuthorizationPort())
    deny_consumer = Tier3MultiplayerConsumer(authorization=DenyingAuthorizationPort())
    allowed = allow_consumer.request_authorized_side_effect(request)
    assert allowed.permitted is True
    assert allowed.decision.action is PolicyAction.ALLOW
    with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
        deny_consumer.request_authorized_side_effect(request)


def test_pluginability_consumer_code_unchanged_between_implementations() -> None:
    request = build_enforcement_request_from_lkw_scope_fixture(
        tenant_id="t1",
        workspace_id="w1",
        acting_principal_id="p1",
        operation_id="mp7b.plugin.unchanged",
    )
    # Same consumer type; only injected port differs.
    for port in (AllowingAuthorizationPort(), DenyingAuthorizationPort()):
        consumer = Tier3MultiplayerConsumer(authorization=port)
        assert isinstance(consumer.authorization, MeaningfulSideEffectAuthorizationPort)
        if isinstance(port, AllowingAuthorizationPort):
            assert consumer.request_authorized_side_effect(request).permitted is True
        else:
            with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
                consumer.request_authorized_side_effect(request)


def test_authorization_exception_fails_closed() -> None:
    request = build_enforcement_request_from_lkw_scope_fixture(
        tenant_id="t1",
        workspace_id="w1",
        acting_principal_id="p1",
        operation_id="mp7b.fail.closed",
    )
    consumer = Tier3MultiplayerConsumer(authorization=ExplodingAuthorizationPort())
    with pytest.raises(Tier3MultiplayerAuthorizationFailedError):
        consumer.request_authorized_side_effect(request)


def test_real_platform_implementation_allow() -> None:
    port, request = compose_allow_platform_port()
    consumer = Tier3MultiplayerConsumer(authorization=port)
    mse = request.meaningful_side_effect_request
    assert mse is not None
    with bound_gr3_active_execution(
        run_id=mse.run_id,
        attempt_id=mse.attempt_id,
        execution_id=mse.execution_id,
    ):
        result = consumer.request_authorized_side_effect(request)
    assert result.permitted is True
    assert result.decision.action is PolicyAction.ALLOW


def test_real_platform_implementation_deny() -> None:
    port, request = compose_deny_platform_port()
    consumer = Tier3MultiplayerConsumer(authorization=port)
    mse = request.meaningful_side_effect_request
    assert mse is not None
    with bound_gr3_active_execution(
        run_id=mse.run_id,
        attempt_id=mse.attempt_id,
        execution_id=mse.execution_id,
    ):
        with pytest.raises(Tier3MultiplayerAuthorizationDeniedError):
            consumer.request_authorized_side_effect(request)


def test_consumer_module_imports_only_public_contracts() -> None:
    tree = ast.parse(
        _CONSUMER_MODULE.read_text(encoding="utf-8"), filename=str(_CONSUMER_MODULE)
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for mod in imported:
        assert not mod.startswith("intergrax.collaborative_work"), mod
        assert "CollaborativeWorkEnforcementGate" not in mod
        assert "PostgreSQLCollaborativeWorkStore" not in mod
        assert "SQLiteCollaborativeWorkStore" not in mod
    assert any(m.startswith("intergrax.contracts.") for m in imported)
    source = _CONSUMER_MODULE.read_text(encoding="utf-8")
    assert "isinstance(" not in source
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "# type: ignore" not in source
    assert "authorization: Any" not in source
    assert "authorization: object" not in source
