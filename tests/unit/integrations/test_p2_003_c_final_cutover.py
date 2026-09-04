# © Artur Czarnecki. All rights reserved.

"""P2-003-C — final explicit integration discovery cutover gates."""

from __future__ import annotations

import ast
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.layout import SLUG_CATEGORY, categories_for_provider, provider_package_path
from intergrax.integrations.providers.llm_guardrail.register_all import GUARD_SLUGS, register_llm_guardrail_integrations
from intergrax.integrations.providers.managed_retrieval.openai.register import register_openai_managed_retrieval_integration
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import (
    IntegrationContractSpec,
    declare_integration_contract,
    typed_contract_categories,
)
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.categories.ai import LlmGuardrailIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability, PlatformIntegrationSecurityPosture
from intergrax.runtime.integrations.registry_v2 import build_contract_registry, build_integration_registration

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def canonical_provider_category_keys() -> frozenset[tuple[str, str]]:
    return frozenset((slug, category) for slug in sorted(SLUG_CATEGORY) for category in categories_for_provider(slug))


def explicit_provider_category_keys() -> frozenset[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
            if contract_path.is_file():
                keys.add((slug, category))
    return frozenset(keys)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


def test_contract_capture_module_deleted() -> None:
    capture_path = REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_capture.py"
    assert not capture_path.is_file()


def test_plugin_register_has_no_contract_capture_fallback() -> None:
    source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py").read_text(
        encoding="utf-8",
    )
    assert "contract_capture" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "contract_capture" in node.module:
            pytest.fail("plugin_register must not import contract_capture")


def test_typed_categories_derive_from_provider_category_contract_registry() -> None:
    assert typed_contract_categories() == frozenset(PROVIDER_CATEGORY_CONTRACT_REGISTRY.keys())


def test_final_inventory_all_canonical_keys_explicit() -> None:
    canonical = canonical_provider_category_keys()
    explicit = explicit_provider_category_keys()
    missing = canonical - explicit
    assert missing == set()
    assert len(canonical) == len(explicit) == 200


def test_staged_and_deferred_keys_empty() -> None:
    plugin_source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py").read_text(
        encoding="utf-8",
    )
    assert "EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS" not in plugin_source
    contract_source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_spec.py").read_text(
        encoding="utf-8",
    )
    assert "EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS" not in contract_source


def test_managed_retrieval_category_gated_without_staged_keys() -> None:
    manifest = IntegrationManifest(
        slug="external_managed",
        categories=(IntegrationCategory.MANAGED_RETRIEVAL,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_MANAGED",
        description="external managed retrieval",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_openai_managed_retrieval_remains_explicit() -> None:
    register_openai_managed_retrieval_integration()
    entry = get_entry("openai")
    assert any(spec.category == "managed_retrieval" for spec in entry.contract_specs)
    registration = build_integration_registration("openai", category="managed_retrieval")
    assert registration.category == "managed_retrieval"


@pytest.mark.parametrize("slug", GUARD_SLUGS)
def test_guardrail_provider_owned_manifest(slug: str) -> None:
    manifest_path = REPO_ROOT / provider_package_path(slug, "llm_guardrail") / "manifest.py"
    assert manifest_path.is_file()


@pytest.mark.parametrize("slug", GUARD_SLUGS)
def test_guardrail_provider_owned_contract_spec(slug: str) -> None:
    contract_path = REPO_ROOT / provider_package_path(slug, "llm_guardrail") / "contract_spec.py"
    register_path = REPO_ROOT / provider_package_path(slug, "llm_guardrail") / "register.py"
    assert contract_path.is_file()
    assert "contract_specs=CONTRACT_SPECS" in register_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("slug", GUARD_SLUGS)
def test_guardrail_registration_side_effect_free(slug: str, monkeypatch: pytest.MonkeyPatch) -> None:
    register_module = import_module(f"intergrax.integrations.providers.llm_guardrail.{slug}.register")
    register_fn = getattr(register_module, f"register_{slug}_integration")
    bundle_module = import_module(f"intergrax.integrations.providers.llm_guardrail.{slug}.bundle")
    catalog_factory_name = f"create_{slug}_llm_guardrail"
    integration_factory_name = f"create_{slug}_llm_guardrail_integration"

    def _must_not_run(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"{slug}: catalog factory must not run during registration")

    if hasattr(bundle_module, catalog_factory_name):
        monkeypatch.setattr(bundle_module, catalog_factory_name, _must_not_run)
    if hasattr(bundle_module, integration_factory_name):
        monkeypatch.setattr(bundle_module, integration_factory_name, _must_not_run)
    register_fn()
    entry = get_entry(slug)
    assert entry.contract_specs


def test_all_guardrails_project_to_registry_v2() -> None:
    register_llm_guardrail_integrations()
    registry = build_contract_registry()
    for slug in GUARD_SLUGS:
        registration = registry.get(provider_id=slug, category="llm_guardrail")
        assert registration.contract_class is LlmGuardrailIntegrationContract
        assert registration.integration_class.__module__.endswith(f".{slug}.integration")


class _ExternalGuardrailIntegration(LlmGuardrailIntegrationContract):
    pass


def _external_guardrail_factory(*, enabled: bool = False) -> _ExternalGuardrailIntegration:
    return _ExternalGuardrailIntegration.for_provider(
        provider_id="external_guard",
        display_name="External Guard",
        config=CategoryIntegrationConfig(enabled=enabled),
    )


def test_external_llm_guardrail_without_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="external_guard",
        categories=(IntegrationCategory.LLM_GUARDRAIL,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_GUARD",
        description="external guardrail",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_external_llm_guardrail_with_specs_succeeds() -> None:
    spec = declare_integration_contract(
        category="llm_guardrail",
        provider_id="external_guard",
        integration_class=_ExternalGuardrailIntegration,
        contract_class=LlmGuardrailIntegrationContract,
        contract_factory=_external_guardrail_factory,
        display_name="External Guard",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "external_plugin_test"},
    )
    manifest = IntegrationManifest(
        slug="external_guard",
        categories=(IntegrationCategory.LLM_GUARDRAIL,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_GUARD",
        description="external guardrail",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_guard")
    assert registration.category == "llm_guardrail"


def test_duplicate_category_specs_fail_closed() -> None:
    manifest = IntegrationManifest(
        slug="dup_spec",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_DUP",
        description="dup",
    )

    class _DupIntegration(RelationalStoreIntegrationContract):
        pass

    def _factory(*, enabled: bool = False) -> _DupIntegration:
        return _DupIntegration.for_provider(
            provider_id="dup_spec",
            display_name="Dup",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="relational_store",
        provider_id="dup_spec",
        integration_class=_DupIntegration,
        contract_class=RelationalStoreIntegrationContract,
        contract_factory=_factory,
        display_name="Dup",
        config_class=CategoryIntegrationConfig,
        capabilities=(PlatformIntegrationCapability.CONNECT,),
        security_posture=PlatformIntegrationSecurityPosture(),
    )
    with pytest.raises(ValueError, match="duplicate contract spec category"):
        register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec, spec))


@pytest.mark.parametrize(
    ("slug", "patch_target"),
    [
        ("nemo_guardrails", "intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails.create_nemo_guardrails_backend"),
        ("bedrock_guardrails", "intergrax.integrations.providers.llm_guardrail.bundles.bedrock_guardrails.create_bedrock_guardrails_backend"),
        ("azure_content_safety", "intergrax.integrations.providers.llm_guardrail.bundles.http_guardrail.create_azure_content_safety_backend"),
        ("lakera", "intergrax.integrations.providers.llm_guardrail.bundles.http_guardrail.create_lakera_backend"),
        ("llama_guard", "intergrax.integrations.providers.llm_guardrail.bundles.llama_guard.create_llama_guard_backend"),
    ],
)
def test_high_risk_guardrail_registration_does_not_materialize_backend(
    slug: str,
    patch_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name, attr = patch_target.rsplit(".", 1)
    module = import_module(module_name)

    def _must_not_create(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"{slug}: backend must not be created during registration")

    monkeypatch.setattr(module, attr, _must_not_create)
    register_module = import_module(f"intergrax.integrations.providers.llm_guardrail.{slug}.register")
    getattr(register_module, f"register_{slug}_integration")()
    assert get_entry(slug).contract_specs
