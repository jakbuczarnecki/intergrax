# © Artur Czarnecki. All rights reserved.

"""EBH-2E-AR1-A-R1-R1-R1 — canonical install-to-rollback contract authority proof."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    RollbackRuntimeRevisionRequest,
)
from intergrax.agent_distribution.agent_contract_authority import (
    contract_metadata_content_digest,
)
from intergrax.applications._shared.active_registry_projection import (
    resolve_active_registry_projection,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_registry_projection_input_bundle import (
    build_production_registry_projection_for_revision,
    build_production_registry_projection_input_bundle_for_revision,
)
from intergrax.applications._shared.registry_projection import (
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
    RegistryProjectionInputBundle,
)
from intergrax.applications._shared.wiring import (
    _index_manifest_bindings,
    binding_from_roster_entry,
)
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.contracts.agent_contract_meta import AgentContract
from tests.unit.applications.test_historical_projection_isolation_phase4e import (
    _APP,
    _BINDING_ID,
    _DIGEST_A,
    _DIGEST_B,
    _ENV,
    _LOGICAL_AGENT_ID,
    _SCENARIO_A,
    _SCENARIO_B,
    _SLOT_ID,
    _contract_for_scenario,
    _authority_resolver,
    _bind_agent,
    _build_phase4e_stack,
    _build_revision,
    _enabled_entry,
    _manifest,
    _mutate_to_state_b,
    _setup_state_a,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    admin_test_principal,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CONTRACT_ID = _LOGICAL_AGENT_ID
_VERSION_A = "1.0.0"
_VERSION_B = "2.0.0"
_REV_A = "rev-ar1-a"
_REV_B = "rev-ar1-b"


def _wire_projection_coordinator(
    stack: AdminStack,
) -> tuple[
    ApplicationRegistryProjectionCoordinator,
    InMemoryRegistryProjectionInputStore,
    InMemoryRuntimeRegistryProjectionStore,
]:
    input_store = InMemoryRegistryProjectionInputStore()
    projection_store = InMemoryRuntimeRegistryProjectionStore()
    coordinator = ApplicationRegistryProjectionCoordinator(
        revision_store=stack.service._revision_store,
        input_store=input_store,
        projection_store=projection_store,
    )
    stack.service._activation_service._projection_coordinator = coordinator
    return coordinator, input_store, projection_store


def _production_bundle(
    stack: AdminStack,
    runtime_revision_id: str,
) -> RegistryProjectionInputBundle:
    manifest = _manifest()
    return build_production_registry_projection_input_bundle_for_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        runtime_revision_id=runtime_revision_id,
        manifest=manifest,
        build_context=ApplicationBuildContext.for_manifest(manifest),
        authority=_authority_resolver(stack),
    )


def _contract_from_bundle(bundle: RegistryProjectionInputBundle) -> AgentContract:
    authority = bundle.roster_contract_authority
    assert authority is not None
    entry = _enabled_entry(bundle)
    binding = binding_from_roster_entry(
        entry,
        _index_manifest_bindings(bundle.manifest),
    )
    return authority.contract_for_binding(binding)


def _register_and_activate(
    stack: AdminStack,
    input_store: InMemoryRegistryProjectionInputStore,
    *,
    runtime_revision_id: str,
    pointer_revision: int,
    prior_traffic_revision_id: str | None,
    artifact_locator: str,
    artifact_digest: str,
    mutation_id: str,
) -> None:
    bundle = _production_bundle(stack, runtime_revision_id)
    input_store.register(bundle)
    stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        principal=admin_test_principal(),
        request=ActivateRuntimeRevisionRequest(
            mutation_id=mutation_id,
            runtime_revision_id=runtime_revision_id,
            artifact_locator=artifact_locator,
            expected_artifact_digest=artifact_digest,
            expected_serving_pointer_revision=pointer_revision,
            expected_prior_traffic_revision_id=prior_traffic_revision_id,
        ),
    )


def _resolve_calls_forbidden_resolver(module_path: Path) -> list[ast.Call]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "resolve_agent_contract_from_binding":
            calls.append(node)
        if isinstance(func, ast.Attribute) and func.attr == "resolve_agent_contract_from_binding":
            calls.append(node)
    return calls


_PRODUCTION_AUTHORITY_SCAN_ROOT = Path("intergrax/applications/_shared")
_PRODUCTION_AUTHORITY_SCAN_GLOBS = (
    "production_*.py",
    "registry_projection*.py",
    "durable_agent_platform_runtime.py",
)


def _production_authority_scan_paths() -> list[Path]:
    paths: list[Path] = []
    for pattern in _PRODUCTION_AUTHORITY_SCAN_GLOBS:
        paths.extend(sorted(_PRODUCTION_AUTHORITY_SCAN_ROOT.glob(pattern)))
    return paths


def test_historical_phase4e_install_has_no_direct_authority_store_persist() -> None:
    source = Path(
        "tests/unit/applications/test_historical_projection_isolation_phase4e.py"
    ).read_text(encoding="utf-8")
    assert "persist_package_contract_authority" not in source


@pytest.mark.parametrize("module_path", _production_authority_scan_paths(), ids=lambda p: p.name)
def test_production_authority_modules_forbid_dynamic_contract_resolver(
    module_path: Path,
) -> None:
    if module_path.name == "roster_agent_contract_authority.py":
        pytest.skip("roster bridge owns lab compatibility resolver")
    source = module_path.read_text(encoding="utf-8")
    assert "materialize_manifest_contract_authority_lab_compat" not in source
    assert _resolve_calls_forbidden_resolver(module_path) == []


def test_canonical_install_to_rollback_contract_authority(tmp_path: Path) -> None:
    contract_a = _contract_for_scenario(_SCENARIO_A)
    contract_b = _contract_for_scenario(_SCENARIO_B)
    assert contract_a.id == contract_b.id == _CONTRACT_ID
    assert contract_a.version != contract_b.version

    stack = _build_phase4e_stack(tmp_path)
    _, input_store, projection_store = _wire_projection_coordinator(stack)

    _setup_state_a(stack)
    built_a = _build_revision(stack, _REV_A, mutation_id="mut-ar1-build-a")
    _mutate_to_state_b(stack)
    built_b = _build_revision(stack, _REV_B, mutation_id="mut-ar1-build-b")

    revision_a = stack.service._revision_store.get_revision(_REV_A)
    revision_b = stack.service._revision_store.get_revision(_REV_B)
    assert revision_a is not None and revision_b is not None
    assert _DIGEST_A in revision_a.installed_agent_package_digests
    assert _DIGEST_B in revision_b.installed_agent_package_digests
    assert revision_a.effective_roster_revision_id != revision_b.effective_roster_revision_id

    record_a = stack.service._artifact_metadata_store.get_package_contract_authority(
        _DIGEST_A,
        _CONTRACT_ID,
    )
    record_b = stack.service._artifact_metadata_store.get_package_contract_authority(
        _DIGEST_B,
        _CONTRACT_ID,
    )
    assert record_a is not None and record_b is not None
    assert record_a.metadata_digest != record_b.metadata_digest
    assert record_a.contract_version == _VERSION_A
    assert record_b.contract_version == _VERSION_B

    digest_a_before_b = record_a.metadata_digest

    _register_and_activate(
        stack,
        input_store,
        runtime_revision_id=_REV_A,
        pointer_revision=0,
        prior_traffic_revision_id=None,
        artifact_locator=built_a.artifact_locator or "test://artifact",
        artifact_digest=built_a.materialization_artifact_digest or "",
        mutation_id="mut-ar1-activate-a",
    )

    serving_a = stack.service.inspect_serving(application_id=_APP, application_environment_id=_ENV)
    assert serving_a.traffic_serving_revision_id == _REV_A

    with patch(
        "intergrax.applications._shared.agent_resolution.resolve_agent_contract_from_binding",
        side_effect=AssertionError("dynamic contract resolver forbidden"),
    ):
        bundle_serving_a = _production_bundle(stack, _REV_A)
        resolved_a = _contract_from_bundle(bundle_serving_a)
        assert resolved_a.version == _VERSION_A
        assert resolved_a.description == contract_a.description

        historical_a = _production_bundle(stack, _REV_A)
        assert _contract_from_bundle(historical_a).version == _VERSION_A

        _register_and_activate(
            stack,
            input_store,
            runtime_revision_id=_REV_B,
            pointer_revision=1,
            prior_traffic_revision_id=_REV_A,
            artifact_locator=built_b.artifact_locator or "test://artifact",
            artifact_digest=built_b.materialization_artifact_digest or "",
            mutation_id="mut-ar1-activate-b",
        )

        serving_b = stack.service.inspect_serving(
            application_id=_APP,
            application_environment_id=_ENV,
        )
        assert serving_b.traffic_serving_revision_id == _REV_B

        bundle_serving_b = _production_bundle(stack, _REV_B)
        resolved_b = _contract_from_bundle(bundle_serving_b)
        assert resolved_b.version == _VERSION_B
        assert resolved_b.description == contract_b.description

        historical_while_b = _production_bundle(stack, _REV_A)
        assert _contract_from_bundle(historical_while_b).version == _VERSION_A

        stack.service.rollback_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            principal=admin_test_principal(),
            request=RollbackRuntimeRevisionRequest(
                mutation_id="mut-ar1-rollback",
                expected_current_traffic_revision_id=_REV_B,
                expected_serving_pointer_revision=2,
                target_runtime_revision_id=_REV_A,
            ),
        )

        serving_after = stack.service.inspect_serving(
            application_id=_APP,
            application_environment_id=_ENV,
        )
        assert serving_after.traffic_serving_revision_id == _REV_A

        bootstrap_production_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=stack.service._serving_store,
            projection_store=projection_store,
        )
        active = resolve_active_registry_projection(
            application_id=_APP,
            application_environment_id=_ENV,
            serving_store=stack.service._serving_store,
            projection_store=projection_store,
        )
        assert active.evidence.runtime_revision_id == _REV_A

        bundle_after_rollback = _production_bundle(stack, _REV_A)
        restored = _contract_from_bundle(bundle_after_rollback)
        assert restored.version == _VERSION_A
        assert restored.version != resolved_b.version
        assert restored.description == contract_a.description

        build_production_registry_projection_for_revision(
            application_id=_APP,
            application_environment_id=_ENV,
            runtime_revision_id=_REV_A,
            manifest=_manifest(),
            build_context=ApplicationBuildContext.for_manifest(_manifest()),
            authority=_authority_resolver(stack),
        )

    record_a_after = stack.service._artifact_metadata_store.get_package_contract_authority(
        _DIGEST_A,
        _CONTRACT_ID,
    )
    assert record_a_after is not None
    assert record_a_after.metadata_digest == digest_a_before_b
    assert record_a_after.metadata_digest == contract_metadata_content_digest(contract_a)

    entry_a = next(
        entry
        for entry in bundle_after_rollback.effective_roster.entries
        if entry.effective_enablement
    )
    assert entry_a.package_digest == _DIGEST_A
    assert entry_a.active_installation_id == _SCENARIO_A.installation_id

    active_install = stack.service._installation_service.resolve_active_for_slot(
        _ENV,
        _SLOT_ID,
    )
    assert active_install is not None
    assert active_install.package_identity.package_digest == _DIGEST_B

    sys.path[:] = [p for p in sys.path if not p.endswith("agents/echo")]
    roster_authority = bundle_after_rollback.roster_contract_authority
    assert roster_authority is not None
    binding = binding_from_roster_entry(
        entry_a,
        _index_manifest_bindings(bundle_after_rollback.manifest),
    )
    assert roster_authority.contract_for_binding(binding).version == _VERSION_A

