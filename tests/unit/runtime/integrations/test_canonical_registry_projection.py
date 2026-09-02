# © Artur Czarnecki. All rights reserved.

"""Canonical catalog authority and contract projection guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import IntegrationContractSpec
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability
from intergrax.runtime.integrations.registry_v2 import (
    IntegrationContractProjectionError,
    build_contract_registry_snapshot,
    build_integration_registration,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


class _ExternalRelationalIntegration(RelationalStoreIntegrationContract):
    """Synthetic external provider outside built-in package layout."""


def _external_contract_factory(*, enabled: bool = False) -> _ExternalRelationalIntegration:
    return _ExternalRelationalIntegration.for_provider(
        provider_id="external_sqlite_like",
        display_name="External SQL",
        config=CategoryIntegrationConfig(enabled=enabled),
    )


class ExternalRelationalPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return IntegrationManifest(
            slug="external_sqlite_like",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_EXTERNAL_SQL",
            description="Synthetic external relational provider for canonical projection proof",
        )

    @classmethod
    def create_integration(cls, **kwargs: object) -> _ExternalRelationalIntegration:
        _ = kwargs
        return _external_contract_factory(enabled=True)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _external_contract_spec() -> IntegrationContractSpec:
    sample = _external_contract_factory(enabled=False)
    return IntegrationContractSpec(
        category="relational_store",
        provider_id=sample.provider_id,
        integration_kind=sample.integration_kind,
        contract_class=RelationalStoreIntegrationContract,
        integration_class=_ExternalRelationalIntegration,
        contract_factory=_external_contract_factory,
        config_class=type(sample.config),
        display_name=sample.display_name,
        capabilities=tuple(capability.value for capability in sample.capabilities),
        security_posture=sample.security_posture,
        supports_runtime_binding=True,
        supports_health_check=PlatformIntegrationCapability.HEALTH_CHECK.value
        in tuple(capability.value for capability in sample.capabilities),
        metadata={"source": "external_plugin_test"},
    )


def test_external_plugin_registers_once_and_projects() -> None:
    register_integration_plugin(
        ExternalRelationalPlugin,
        contract_specs=(_external_contract_spec(),),
    )

    entry = get_entry("external_sqlite_like")
    assert entry.contract_specs
    profile = IntegrationProfile(relational_store=ExternalRelationalPlugin)
    integration = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(integration, _ExternalRelationalIntegration)

    snapshot = build_contract_registry_snapshot(slugs=("external_sqlite_like",))
    registration = snapshot.get(provider_id="external_sqlite_like", category="relational_store")
    assert registration.slug == "external_sqlite_like"
    assert registration.category == "relational_store"
    assert registration.integration_class is _ExternalRelationalIntegration


def test_projection_snapshot_rebuilds_after_catalog_clear() -> None:
    register_integration_plugin(
        ExternalRelationalPlugin,
        contract_specs=(_external_contract_spec(),),
    )
    assert len(build_contract_registry_snapshot(slugs=("external_sqlite_like",))) == 1

    clear_catalog()
    assert len(build_contract_registry_snapshot(slugs=("external_sqlite_like",))) == 0


def test_identity_mismatch_fails_projection() -> None:
    register_integration_plugin(
        ExternalRelationalPlugin,
        contract_specs=(
            IntegrationContractSpec(
                category="relational_store",
                provider_id="wrong_provider_id",
                integration_kind="relational_store",
                contract_class=RelationalStoreIntegrationContract,
                integration_class=_ExternalRelationalIntegration,
                contract_factory=_external_contract_factory,
            ),
        ),
    )

    with pytest.raises(IntegrationContractProjectionError, match="identity mismatch"):
        build_integration_registration("external_sqlite_like")


def test_builtin_sqlite_qdrant_projection_from_canonical_catalog() -> None:
    register_default_integrations(preset="core")

    sqlite = build_integration_registration("sqlite")
    qdrant = build_integration_registration("qdrant")

    assert sqlite.provider_id == "sqlite"
    assert sqlite.category == "relational_store"
    assert qdrant.provider_id == "qdrant"
    assert qdrant.category == "vector_store"
    assert sqlite.factory is not None
    assert sqlite.integration_class.__name__.endswith("Integration")


def test_qualification_core_does_not_import_registry_v2_for_execution() -> None:
    qualification_dir = REPO_ROOT / "intergrax" / "core" / "qualification"
    offenders: list[str] = []
    for path in qualification_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "registry_v2" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


def test_registry_v2_authoritative_projection_has_no_reflection_helpers() -> None:
    source = (REPO_ROOT / "intergrax" / "runtime" / "integrations" / "registry_v2.py").read_text(
        encoding="utf-8",
    )
    tree = ast.parse(source)
    forbidden_names = {"vars", "getattr", "hasattr", "__dict__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_names:
                pytest.fail(f"registry_v2 authoritative projection must not call {node.func.id}()")
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            pytest.fail("registry_v2 authoritative projection must not access __dict__")


def test_contract_capture_isolated_as_registration_time_compatibility() -> None:
    capture_path = REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_capture.py"
    assert capture_path.is_file()
    source = capture_path.read_text(encoding="utf-8")
    assert "P2" not in source  # documented in ADR instead
    assert "_find_integration_class" in source
