# © Artur Czarnecki. All rights reserved.

"""Harness-host meaningful side-effect authorization composition (P2D-R1 / P2D-R1A)."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    close_harness_host_runtime,
)
from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    build_harness_host_meaningful_side_effect_authorization_port,
    resolve_harness_host_meaningful_side_effect_authorization_port,
    resolve_harness_host_meaningful_side_effect_authorization_wiring,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)

pytestmark = pytest.mark.unit

_WIRING_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "applications"
    / "_shared"
    / "harness_meaningful_side_effect_authorization_wiring.py"
)


class _RecordingMsePort:
    def authorize(self, request: object, **_: object) -> object:
        del request
        raise AssertionError("custom port must not authorize in this proof")


class _ObservableCollaborativeWorkStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _in_memory_core_bundle(
    store: _ObservableCollaborativeWorkStore | None = None,
) -> CollaborativeWorkRepositories:
    resolved_store = store if store is not None else _ObservableCollaborativeWorkStore()
    return CollaborativeWorkRepositories(
        membership=InMemoryWorkspaceMembershipRepository(),
        delegation=InMemoryAuthorityDelegationRepository(),
        principal_authority=InMemoryPrincipalAuthorityRepository(),
        policy=InMemoryCollaborativePolicyRepository(),
        operation_profile=InMemoryCollaborativeOperationPolicyProfileRepository(),
        store=resolved_store,
    )


def _strict_product_env() -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    return manifest.environment or build_governed_contractor_environment_profile(settings)


def test_shared_host_wiring_module_does_not_import_sqlite_persistence() -> None:
    tree = ast.parse(_WIRING_MODULE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    assert "intergrax.collaborative_work.persistence" in imported
    source = _WIRING_MODULE.read_text(encoding="utf-8")
    assert "open_sqlite_collaborative_work_repositories" not in source
    assert "sqlite" not in source.lower()


def test_resolve_returns_none_for_non_strict_host() -> None:
    env = _strict_product_env().model_copy(
        update={
            "meta": _strict_product_env().meta.model_copy(
                update={"execution_mode": ExecutionMode.BALANCED},
            ),
        },
    )
    wiring = resolve_harness_host_meaningful_side_effect_authorization_wiring(env)
    assert wiring.authorization_port is None
    assert wiring.owned_collaborative_work_persistence is None
    assert (
        resolve_harness_host_meaningful_side_effect_authorization_port(env) is None
    )


def test_strict_default_uses_provider_resolver_not_sqlite(tmp_path: Path) -> None:
    del tmp_path
    env = _strict_product_env()
    store = _ObservableCollaborativeWorkStore()
    bundle = _in_memory_core_bundle(store)
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
        return_value=bundle,
    ) as resolve_mock:
        wiring = resolve_harness_host_meaningful_side_effect_authorization_wiring(env)
    resolve_mock.assert_called_once_with(env.integration_profile)
    assert wiring.authorization_port is not None
    assert isinstance(wiring.authorization_port, MeaningfulSideEffectAuthorizationPort)
    assert wiring.owned_collaborative_work_persistence is bundle


def test_build_port_with_injected_repositories_does_not_resolve_provider() -> None:
    env = _strict_product_env()
    bundle = _in_memory_core_bundle()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        port = build_harness_host_meaningful_side_effect_authorization_port(
            env,
            collaborative_work_repositories=bundle,
        )
    resolve_mock.assert_not_called()
    assert isinstance(port, MeaningfulSideEffectAuthorizationPort)


def test_explicit_custom_port_is_used_as_is_without_cw_allocation() -> None:
    env = _strict_product_env()
    custom = _RecordingMsePort()
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        wiring = resolve_harness_host_meaningful_side_effect_authorization_wiring(
            env,
            explicit=custom,
            collaborative_work_repositories=_in_memory_core_bundle(),
        )
    resolve_mock.assert_not_called()
    assert wiring.authorization_port is custom
    assert wiring.owned_collaborative_work_persistence is None


def test_borrowed_collaborative_work_repositories_are_not_host_owned() -> None:
    env = _strict_product_env()
    bundle = _in_memory_core_bundle()
    wiring = resolve_harness_host_meaningful_side_effect_authorization_wiring(
        env,
        collaborative_work_repositories=bundle,
    )
    assert wiring.authorization_port is not None
    assert wiring.owned_collaborative_work_persistence is None


def test_close_harness_host_runtime_closes_owned_collaborative_work_persistence() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = _in_memory_core_bundle(store)
    runtime = MagicMock(spec=HarnessHostRuntime)
    runtime.env_wiring = MagicMock()
    runtime.env_wiring.event_delivery = MagicMock()
    runtime._owned_collaborative_work_persistence = bundle
    with patch(
        "intergrax.applications._shared.harness_host_composition.resolve_harness_host_event_bus",
        return_value=MagicMock(),
    ):
        close_harness_host_runtime(runtime)
    assert store.closed is True


def test_close_harness_host_runtime_does_not_close_borrowed_bundle_field() -> None:
    store = _ObservableCollaborativeWorkStore()
    bundle = _in_memory_core_bundle(store)
    runtime = MagicMock(spec=HarnessHostRuntime)
    runtime.env_wiring = MagicMock()
    runtime.env_wiring.event_delivery = MagicMock()
    runtime._owned_collaborative_work_persistence = None
    with patch(
        "intergrax.applications._shared.harness_host_composition.resolve_harness_host_event_bus",
        return_value=MagicMock(),
    ):
        close_harness_host_runtime(runtime)
    assert store.closed is False
    bundle.close()
    assert store.closed is True
