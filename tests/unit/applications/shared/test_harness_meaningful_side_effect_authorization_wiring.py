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
from echo.echo_agent import EchoAgent
from intergrax.applications._shared.harness_host_runtime import (
    HarnessHostRuntime,
    build_harness_host_runtime,
    close_harness_host_runtime,
)
from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    HarnessMeaningfulSideEffectAuthorizationWiring,
)
from intergrax.applications._shared.harness_registry_authority import RegistryAssemblyMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.registry.agent_registry import AgentRegistry
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
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)

pytestmark = pytest.mark.unit

_SHARED_APP_ROOT = Path(__file__).resolve().parents[4] / "intergrax" / "applications" / "_shared"
_WIRING_MODULE = _SHARED_APP_ROOT / "harness_meaningful_side_effect_authorization_wiring.py"
_HOST_RUNTIME_MODULE = _SHARED_APP_ROOT / "harness_host_runtime.py"


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
        wiring = resolve_harness_host_meaningful_side_effect_authorization_wiring(
            env,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        )
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
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
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
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
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


def _build_harness_host_runtime_function_source() -> str:
    text = _HOST_RUNTIME_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_harness_host_runtime":
            lines = text.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError("build_harness_host_runtime not found")


def test_build_harness_host_runtime_has_no_vendor_collaborative_work_literals() -> None:
    segment = _build_harness_host_runtime_function_source()
    for forbidden in (
        "sqlite_options",
        "relational_db",
        "collaborative_work.db",
        '"sqlite"',
    ):
        assert forbidden not in segment


def _minimal_diag_style_manifest() -> ApplicationManifest:
    env = _strict_product_env().model_copy(
        update={
            "meta": _strict_product_env().meta.model_copy(
                update={"execution_mode": ExecutionMode.BALANCED},
            ),
        },
    )
    return ApplicationManifest.lab(
        app_id="mse_host_neutrality",
        name="MSE Host Neutrality",
        route_prefix="/v1/mse_host_neutrality",
        env_prefix="MSE_HOST_NEUTRALITY_",
        agents=[AgentBinding.mount(EchoAgent, contract_id="echo", capabilities=["echo.basic"])],
        environment=env,
    )


def _echo_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(EchoAgent())
    return registry


def _capture_host_mse_wiring(
    manifest: ApplicationManifest,
    *,
    trace_db_path: Path | None = None,
    document_store: InMemoryDocumentStore | None = None,
    collaborative_work_integration_profile: IntegrationProfile | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    def _record_wiring(
        environment: object,
        **kwargs: object,
    ) -> HarnessMeaningfulSideEffectAuthorizationWiring:
        captured["environment"] = environment
        captured.update(kwargs)
        return HarnessMeaningfulSideEffectAuthorizationWiring(authorization_port=None)

    doc_store = document_store if document_store is not None else InMemoryDocumentStore()
    with (
        patch(
            "intergrax.applications._shared.harness_host_runtime.resolve_harness_host_meaningful_side_effect_authorization_wiring",
            side_effect=_record_wiring,
        ),
        patch(
            "intergrax.applications._shared.diagnostic_runtime_wiring.wire_terminal_execution_diagnostics",
            return_value=MagicMock(),
        ),
    ):
        build_harness_host_runtime(
            manifest,
            manifest.environment,
            registry=_echo_registry(),
            registry_assembly_mode=RegistryAssemblyMode.MANIFEST_DEVELOPMENT,
            trace_db_path=trace_db_path,
            runtime_events_db_path=trace_db_path.parent / "events.db" if trace_db_path else None,
            document_store=doc_store,
            collaborative_work_integration_profile=collaborative_work_integration_profile,
            use_in_memory_trace=trace_db_path is None,
        )
    assert captured["collaborative_work_integration_profile"] == (
        collaborative_work_integration_profile
    )
    return captured


def test_host_passes_explicit_collaborative_work_integration_profile_unmodified() -> None:
    marker = {"mse_host_neutrality_marker": "explicit-profile-proof"}
    explicit_profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: marker},
    )
    captured = _capture_host_mse_wiring(
        _minimal_diag_style_manifest(),
        collaborative_work_integration_profile=explicit_profile,
    )
    profile = captured["collaborative_work_integration_profile"]
    assert profile is explicit_profile
    assert profile.options[SQLITE.slug] == marker


def test_host_default_collaborative_work_integration_profile_is_not_locally_mutated(
    tmp_path: Path,
) -> None:
    captured = _capture_host_mse_wiring(
        _minimal_diag_style_manifest(),
        trace_db_path=tmp_path / "trace.db",
        document_store=InMemoryDocumentStore(),
    )
    assert captured["collaborative_work_integration_profile"] is None


def test_trace_db_path_does_not_mutate_collaborative_work_integration_profile(
    tmp_path: Path,
) -> None:
    without_trace = _capture_host_mse_wiring(_minimal_diag_style_manifest())
    with_trace = _capture_host_mse_wiring(
        _minimal_diag_style_manifest(),
        trace_db_path=tmp_path / "trace.db",
        document_store=InMemoryDocumentStore(),
    )
    assert without_trace["collaborative_work_integration_profile"] == (
        with_trace["collaborative_work_integration_profile"]
    )


def test_document_store_does_not_mutate_collaborative_work_integration_profile() -> None:
    without_store = _capture_host_mse_wiring(_minimal_diag_style_manifest())
    with_store = _capture_host_mse_wiring(
        _minimal_diag_style_manifest(),
        document_store=InMemoryDocumentStore(),
    )
    assert without_store["collaborative_work_integration_profile"] == (
        with_store["collaborative_work_integration_profile"]
    )


def test_strict_default_profile_reaches_resolver_without_host_mutation(tmp_path: Path) -> None:
    del tmp_path
    env = _strict_product_env()
    explicit_profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"proof": "as-is"}},
    )
    with patch(
        "intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring.resolve_collaborative_work_repositories",
    ) as resolve_mock:
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            env,
            collaborative_work_integration_profile=explicit_profile,
            decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
        )
    resolve_mock.assert_called_once_with(explicit_profile)


def test_strict_harness_missing_decision_policy_fails_closed() -> None:
    from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
        OrchestrationDecisionBoundCompositionError,
    )

    env = _strict_product_env()
    bundle = _in_memory_core_bundle()
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        resolve_harness_host_meaningful_side_effect_authorization_wiring(
            env,
            collaborative_work_repositories=bundle,
        )
