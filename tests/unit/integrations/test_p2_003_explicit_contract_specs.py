# © Artur Czarnecki. All rights reserved.

"""P2-003 — explicit integration contract declaration architecture gates."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.conversation_channel.slack.contract_spec import (
    CONTRACT_SPEC as SLACK_CONVERSATION_CONTRACT_SPEC,
)
from intergrax.integrations.providers.layout import (
    SLUG_CATEGORY,
    categories_for_provider,
    provider_import_path,
    provider_package_path,
)
from intergrax.integrations.providers.managed_retrieval.openai.contract_spec import (
    CONTRACT_SPEC as OPENAI_MANAGED_RETRIEVAL_CONTRACT_SPEC,
)
from intergrax.integrations.providers.managed_retrieval.openai.register import (
    register_openai_managed_retrieval_integration,
)
from intergrax.integrations.providers.observability_backend.langfuse.contract_spec import (
    CONTRACT_SPEC as LANGFUSE_CONTRACT_SPEC,
)
from intergrax.integrations.providers.relational_store.postgresql.contract_spec import (
    CONTRACT_SPEC as POSTGRESQL_CONTRACT_SPEC,
)
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.registry import contract_capture
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import (
    B1_TYPED_CONTRACT_CATEGORIES,
    B2_TYPED_CONTRACT_CATEGORIES,
    B3_TYPED_CONTRACT_CATEGORIES,
    B4_TYPED_CONTRACT_CATEGORIES,
    B5_TYPED_CONTRACT_CATEGORIES,
    EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS,
    IntegrationContractSpec,
    declare_integration_contract,
)
from intergrax.integrations.registry.plugin_register import register_from_manifest, register_integration_plugin
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories.collaboration import IssueTrackerIntegrationContract
from intergrax.runtime.integrations.categories.messaging import MessageBusIntegrationContract
from intergrax.runtime.integrations.categories.automation import BrowserAutomationIntegrationContract
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories.security import FeatureFlagIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability, PlatformIntegrationSecurityPosture
from intergrax.runtime.integrations.registry_v2 import build_integration_registration

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

_FORBIDDEN_CONTRACT_SPEC_NAMES = frozenset(
    {"import_module", "vars", "getattr", "hasattr", "__dict__"},
)


def b1_provider_category_keys() -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in B1_TYPED_CONTRACT_CATEGORIES:
                keys.append((slug, category))
    return tuple(keys)


def b2_provider_category_keys() -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in B2_TYPED_CONTRACT_CATEGORIES:
                keys.append((slug, category))
    return tuple(keys)


def b3_provider_category_keys() -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in B3_TYPED_CONTRACT_CATEGORIES:
                keys.append((slug, category))
    return tuple(keys)


def b4_provider_category_keys() -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in B4_TYPED_CONTRACT_CATEGORIES:
                keys.append((slug, category))
    return tuple(keys)


def b5_provider_category_keys() -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in B5_TYPED_CONTRACT_CATEGORIES:
                keys.append((slug, category))
    return tuple(keys)


def typed_register_function(slug: str, category: str) -> Callable[..., Any]:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    for name, obj in inspect.getmembers(register_module, inspect.isfunction):
        if name.startswith("register_") and name.endswith("_integration"):
            return obj
    msg = f"Missing register_*_integration in {slug}/{category}"
    raise AssertionError(msg)


def b1_register_function(slug: str, category: str) -> Callable[..., Any]:
    return typed_register_function(slug, category)


def b2_register_function(slug: str, category: str) -> Callable[..., Any]:
    return typed_register_function(slug, category)


def b3_register_function(slug: str, category: str) -> Callable[..., Any]:
    return typed_register_function(slug, category)


def b4_register_function(slug: str, category: str) -> Callable[..., Any]:
    return typed_register_function(slug, category)


def b5_register_function(slug: str, category: str) -> Callable[..., Any]:
    return typed_register_function(slug, category)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    yield
    clear_catalog()


class _SampleRelationalIntegration(RelationalStoreIntegrationContract):
    pass


def _sample_factory(*, enabled: bool = False) -> _SampleRelationalIntegration:
    return _SampleRelationalIntegration.for_provider(
        provider_id="sample_sql",
        display_name="Sample SQL",
        config=CategoryIntegrationConfig(enabled=enabled),
    )


def _sample_explicit_spec() -> IntegrationContractSpec:
    return declare_integration_contract(
        category="relational_store",
        provider_id="sample_sql",
        integration_class=_SampleRelationalIntegration,
        contract_class=RelationalStoreIntegrationContract,
        contract_factory=_sample_factory,
        display_name="Sample SQL",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "test"},
    )


def test_b1_inventory_gate_all_typed_keys_explicit() -> None:
    b1_keys = b1_provider_category_keys()
    explicit_keys: list[tuple[str, str]] = []
    for slug, category in b1_keys:
        contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        assert contract_path.is_file(), f"missing contract_spec for {(slug, category)}"
        register_path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        register_source = register_path.read_text(encoding="utf-8")
        assert "contract_specs=CONTRACT_SPECS" in register_source
        explicit_keys.append((slug, category))
    assert len(explicit_keys) == len(b1_keys)


def test_b2_inventory_gate_all_typed_keys_explicit() -> None:
    b2_keys = b2_provider_category_keys()
    explicit_keys: list[tuple[str, str]] = []
    for slug, category in b2_keys:
        contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        assert contract_path.is_file(), f"missing contract_spec for {(slug, category)}"
        register_path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        register_source = register_path.read_text(encoding="utf-8")
        assert "contract_specs=CONTRACT_SPECS" in register_source
        explicit_keys.append((slug, category))
    assert len(explicit_keys) == len(b2_keys)


def test_b3_category_set_matches_expected_infrastructure_operations_categories() -> None:
    assert B3_TYPED_CONTRACT_CATEGORIES == frozenset(
        {
            "observability_backend",
            "cloud_platform",
            "ci_cd",
            "workflow_orchestrator",
            "feature_flag",
            "secrets_store",
            "billing_meter",
        }
    )


def test_b3_inventory_gate_all_typed_keys_explicit() -> None:
    b3_keys = b3_provider_category_keys()
    explicit_keys: list[tuple[str, str]] = []
    for slug, category in b3_keys:
        contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        assert contract_path.is_file(), f"missing contract_spec for {(slug, category)}"
        register_path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        register_source = register_path.read_text(encoding="utf-8")
        assert "contract_specs=CONTRACT_SPECS" in register_source
        explicit_keys.append((slug, category))
    assert len(explicit_keys) == len(b3_keys)


def test_explicit_spec_populates_catalog_entry() -> None:
    manifest = IntegrationManifest(
        slug="sample_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_SAMPLE_SQL",
        description="sample",
    )
    register_from_manifest(
        manifest,
        lambda **_: {"slug": "sample_sql"},
        contract_specs=(_sample_explicit_spec(),),
    )
    entry = get_entry("sample_sql")
    assert len(entry.contract_specs) == 1
    assert entry.contract_specs[0].provider_id == "sample_sql"


def test_registry_v2_derives_from_explicit_specs() -> None:
    register_postgresql_integration()
    registration = build_integration_registration("postgresql")
    assert registration.provider_id == "postgresql"
    assert registration.category == "relational_store"
    assert registration.integration_class is POSTGRESQL_CONTRACT_SPEC.integration_class


def test_migrated_builtin_uses_explicit_declaration_not_capture_metadata() -> None:
    register_postgresql_integration()
    entry = get_entry("postgresql")
    assert entry.contract_specs[0].metadata.get("source") == "explicit_provider_declaration"


def test_explicit_registration_does_not_execute_catalog_factory() -> None:
    factory = MagicMock(return_value={"slug": "sample_sql"})
    manifest = IntegrationManifest(
        slug="sample_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_SAMPLE_SQL",
        description="sample",
    )
    register_from_manifest(
        manifest,
        factory,
        contract_specs=(_sample_explicit_spec(),),
    )
    factory.assert_not_called()


def test_category_mismatch_fails_explicit_registration() -> None:
    manifest = IntegrationManifest(
        slug="sample_sql",
        categories=(IntegrationCategory.KEY_VALUE_CACHE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_SAMPLE_SQL",
        description="sample",
    )
    with pytest.raises(ValueError, match="not declared on manifest categories"):
        register_from_manifest(
            manifest,
            lambda **_: {},
            contract_specs=(_sample_explicit_spec(),),
        )


def test_integration_class_mismatch_fails_declaration() -> None:
    with pytest.raises(TypeError, match="must subclass"):
        declare_integration_contract(
            category="relational_store",
            provider_id="bad",
            integration_class=object,
            contract_class=RelationalStoreIntegrationContract,
            contract_factory=_sample_factory,
            display_name="bad",
            config_class=CategoryIntegrationConfig,
            capabilities=(PlatformIntegrationCapability.CONNECT,),
            security_posture=PlatformIntegrationSecurityPosture(),
        )


def test_b1_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="sqlite",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_SQLITE",
        description="sqlite",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_managed_retrieval_explicit_registration() -> None:
    register_openai_managed_retrieval_integration()
    entry = get_entry("openai")
    assert entry.contract_specs == (OPENAI_MANAGED_RETRIEVAL_CONTRACT_SPEC,)
    registration = build_integration_registration("openai")
    assert registration.category == "managed_retrieval"


def test_slack_conversation_metadata_is_provider_owned() -> None:
    metadata = SLACK_CONVERSATION_CONTRACT_SPEC.metadata
    assert metadata.get("conversation_features") == ("text", "single_choice")
    assert metadata.get("runtime_implemented") is True
    register_source = (
        REPO_ROOT / "intergrax" / "integrations" / "providers" / "conversation_channel" / "slack" / "register.py"
    ).read_text(encoding="utf-8")
    assert "contract_specs=" in register_source
    assert "contract_capture" not in register_source


def test_langfuse_observability_uses_explicit_factory_not_name_guess() -> None:
    assert LANGFUSE_CONTRACT_SPEC.contract_factory.__name__ == "create_langfuse_observability_integration"
    assert LANGFUSE_CONTRACT_SPEC.integration_kind == "observability_vendor"


def test_staged_non_b1b2b3_explicit_keys_remain_in_migration_set() -> None:
    assert EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS == frozenset({("openai", "managed_retrieval")})
    assert ("langfuse", "observability_backend") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("slack", "notification_channel") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("slack", "conversation_channel") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("postgresql", "relational_store") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("github", "issue_tracker") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("prometheus", "observability_backend") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS


def test_plugin_register_has_no_provider_module_scanning() -> None:
    source = (
        REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"vars", "import_module"}:
                pytest.fail("plugin_register must not scan provider modules reflectively")


def test_plugin_register_has_no_builtin_layout_dependency() -> None:
    source = (
        REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py"
    ).read_text(encoding="utf-8")
    assert "intergrax.integrations.providers.layout" not in source
    assert "SLUG_CATEGORY" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "providers.layout" in node.module:
                pytest.fail("plugin_register must not import built-in provider layout")


def test_b1_contract_spec_modules_have_no_reflection() -> None:
    for slug, category in b1_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CONTRACT_SPEC_NAMES:
                    pytest.fail(f"{path}: forbidden reflective call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                pytest.fail(f"{path}: forbidden __dict__ access")


def test_b2_contract_spec_modules_have_no_reflection() -> None:
    for slug, category in b2_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CONTRACT_SPEC_NAMES:
                    pytest.fail(f"{path}: forbidden reflective call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                pytest.fail(f"{path}: forbidden __dict__ access")


def test_b1_register_modules_do_not_import_contract_capture() -> None:
    for slug, category in b1_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        source = path.read_text(encoding="utf-8")
        assert "contract_capture" not in source, path.as_posix()
        assert "contract_specs=" in source, path.as_posix()


def test_b2_register_modules_do_not_import_contract_capture() -> None:
    for slug, category in b2_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        source = path.read_text(encoding="utf-8")
        assert "contract_capture" not in source, path.as_posix()
        assert "contract_specs=" in source, path.as_posix()


@pytest.mark.parametrize("slug,category", b1_provider_category_keys())
def test_b1_registration_bypasses_contract_capture(slug: str, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError(f"capture_builtin_contract_specs must not run for B1 {(slug, category)}")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    b1_register_function(slug, category)()
    entry = get_entry(slug)
    assert entry.contract_specs
    assert entry.contract_specs[0].category == category
    assert entry.contract_specs[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug,category", b1_provider_category_keys())
def test_b1_registration_does_not_execute_catalog_factory(
    slug: str,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    register_fn = b1_register_function(slug, category)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: IntegrationManifest,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> IntegrationManifest:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


@pytest.mark.parametrize("slug,category", b2_provider_category_keys())
def test_b2_registration_bypasses_contract_capture(slug: str, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError(f"capture_builtin_contract_specs must not run for B2 {(slug, category)}")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    b2_register_function(slug, category)()
    entry = get_entry(slug)
    assert entry.contract_specs
    matching = [spec for spec in entry.contract_specs if spec.category == category]
    assert matching
    assert matching[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug,category", b2_provider_category_keys())
def test_b2_registration_does_not_execute_catalog_factory(
    slug: str,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    register_fn = b2_register_function(slug, category)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: IntegrationManifest,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> IntegrationManifest:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


def test_b2_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="kafka",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_KAFKA",
        description="kafka",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_external_fake_b1_provider_explicit_registration() -> None:
    class _ExternalB1Integration(RelationalStoreIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB1Integration:
        return _ExternalB1Integration.for_provider(
            provider_id="external_b1_sql",
            display_name="External B1 SQL",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="relational_store",
        provider_id="external_b1_sql",
        integration_class=_ExternalB1Integration,
        contract_class=RelationalStoreIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B1 SQL",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "external_plugin_test"},
    )
    manifest = IntegrationManifest(
        slug="external_b1_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B1_SQL",
        description="external fake B1 provider",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_b1_sql")
    assert registration.category == "relational_store"
    assert registration.integration_class is _ExternalB1Integration


def test_external_fake_b1_provider_without_explicit_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B1 typed provider")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    manifest = IntegrationManifest(
        slug="external_b1_sql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B1_SQL",
        description="external fake B1 provider",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def _external_b1_plugin_spec() -> IntegrationContractSpec:
    class _ExternalB1Integration(RelationalStoreIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB1Integration:
        return _ExternalB1Integration.for_provider(
            provider_id="external_b1_sql",
            display_name="External B1 SQL",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    return declare_integration_contract(
        category="relational_store",
        provider_id="external_b1_sql",
        integration_class=_ExternalB1Integration,
        contract_class=RelationalStoreIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B1 SQL",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "external_plugin_test"},
    )


class _ExternalB1SqlPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return IntegrationManifest(
            slug="external_b1_sql",
            categories=(IntegrationCategory.RELATIONAL_STORE,),
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_EXTERNAL_B1_SQL",
            description="external fake B1 provider",
        )

    @classmethod
    def create_integration(cls, **kwargs: Any) -> Any:
        _ = kwargs
        return _external_b1_plugin_spec().contract_factory()


def test_register_integration_plugin_external_b1_without_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B1 plugin")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_integration_plugin(_ExternalB1SqlPlugin)


def test_register_integration_plugin_external_b1_with_explicit_spec_succeeds() -> None:
    register_integration_plugin(_ExternalB1SqlPlugin, contract_specs=(_external_b1_plugin_spec(),))
    registration = build_integration_registration("external_b1_sql")
    assert registration.category == "relational_store"


def test_multi_category_manifest_with_b1_category_without_specs_fails() -> None:
    manifest = IntegrationManifest(
        slug="multi_category_b1",
        categories=(IntegrationCategory.RELATIONAL_STORE, IntegrationCategory.ISSUE_TRACKER),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B1",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_partial_explicit_specs_missing_required_b1_category_fails() -> None:
    class _IssueTrackerIntegration(IssueTrackerIntegrationContract):
        pass

    def _issue_factory(*, enabled: bool = False) -> _IssueTrackerIntegration:
        return _IssueTrackerIntegration.for_provider(
            provider_id="multi_category_b1",
            display_name="Issue Tracker",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    issue_spec = declare_integration_contract(
        category="issue_tracker",
        provider_id="multi_category_b1",
        integration_class=_IssueTrackerIntegration,
        contract_class=IssueTrackerIntegrationContract,
        contract_factory=_issue_factory,
        display_name="Issue Tracker",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
    )
    manifest = IntegrationManifest(
        slug="multi_category_b1",
        categories=(IntegrationCategory.RELATIONAL_STORE, IntegrationCategory.ISSUE_TRACKER),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B1",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="is missing explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {}, contract_specs=(issue_spec,))


def test_explicit_spec_covering_required_b1_category_succeeds() -> None:
    class _MultiCategoryRelationalIntegration(RelationalStoreIntegrationContract):
        pass

    def _relational_factory(*, enabled: bool = False) -> _MultiCategoryRelationalIntegration:
        return _MultiCategoryRelationalIntegration.for_provider(
            provider_id="multi_category_b1",
            display_name="Multi Category B1",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    relational_spec = declare_integration_contract(
        category="relational_store",
        provider_id="multi_category_b1",
        integration_class=_MultiCategoryRelationalIntegration,
        contract_class=RelationalStoreIntegrationContract,
        contract_factory=_relational_factory,
        display_name="Multi Category B1",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "test"},
    )

    class _MultiCategorySearchIntegration(SearchProviderIntegrationContract):
        pass

    def _search_factory(*, enabled: bool = False) -> _MultiCategorySearchIntegration:
        return _MultiCategorySearchIntegration.for_provider(
            provider_id="multi_category_b1",
            display_name="Multi Category B1",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    search_spec = declare_integration_contract(
        category="search_provider",
        provider_id="multi_category_b1",
        integration_class=_MultiCategorySearchIntegration,
        contract_class=SearchProviderIntegrationContract,
        contract_factory=_search_factory,
        display_name="Multi Category B1",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.READ,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "test"},
    )
    manifest = IntegrationManifest(
        slug="multi_category_b1",
        categories=(IntegrationCategory.RELATIONAL_STORE, IntegrationCategory.SEARCH_PROVIDER),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B1",
        description="multi-category manifest",
    )
    register_from_manifest(
        manifest,
        lambda **_: {},
        contract_specs=(relational_spec, search_spec),
    )
    entry = get_entry("multi_category_b1")
    assert any(spec.category == "relational_store" for spec in entry.contract_specs)


def test_external_fake_b2_provider_explicit_registration() -> None:
    class _ExternalB2Integration(MessageBusIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB2Integration:
        return _ExternalB2Integration.for_provider(
            provider_id="external_b2_bus",
            display_name="External B2 Bus",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="message_bus",
        provider_id="external_b2_bus",
        integration_class=_ExternalB2Integration,
        contract_class=MessageBusIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B2 Bus",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "external_plugin_test"},
    )
    manifest = IntegrationManifest(
        slug="external_b2_bus",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B2_BUS",
        description="external fake B2 provider",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_b2_bus")
    assert registration.category == "message_bus"
    assert registration.integration_class is _ExternalB2Integration


def test_external_fake_b2_provider_without_explicit_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B2 typed provider")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    manifest = IntegrationManifest(
        slug="external_b2_bus",
        categories=(IntegrationCategory.MESSAGE_BUS,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B2_BUS",
        description="external fake B2 provider",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_contract_capture_has_no_conversation_channel_vendor_switch() -> None:
    source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_capture.py").read_text(
        encoding="utf-8",
    )
    assert 'slug == "slack"' not in source
    assert "conversation_channel" not in source


def test_b2_factory_backward_compatibility_representatives() -> None:
    from intergrax.integrations.providers.issue_tracker.github.bundle import create_github_issue_tracker
    from intergrax.integrations.providers.message_bus.kafka.bundle import create_kafka_message_bus
    from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_notification_channel
    from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge

    assert callable(create_github_issue_tracker)
    assert callable(create_kafka_message_bus)
    assert callable(create_slack_notification_channel)
    assert callable(create_confluence_wiki_knowledge)


def test_b2_registry_v2_derives_from_explicit_specs() -> None:
    from intergrax.integrations.providers.issue_tracker.github.register import register_github_integration

    register_github_integration()
    registration = build_integration_registration("github")
    assert registration.provider_id == "github"
    assert registration.category == "issue_tracker"


def test_non_b1b2b3b4b5_ordinary_reflective_keys_are_empty() -> None:
    from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
    from intergrax.runtime.integrations.registry_v2 import DEFERRED_LLM_GUARDRAIL_SLUGS
    from intergrax.integrations.registry.contract_spec import required_explicit_contract_categories

    explicit_category_gated = required_explicit_contract_categories()
    normal_typed_keys: set[tuple[str, str]] = set()
    for slug in sorted(SLUG_CATEGORY):
        for category in categories_for_provider(slug):
            if category in PROVIDER_CATEGORY_CONTRACT_REGISTRY:
                normal_typed_keys.add((slug, category))

    deferred_typed_keys = {(slug, "llm_guardrail") for slug in DEFERRED_LLM_GUARDRAIL_SLUGS}
    explicit_normal_keys = {
        key
        for key in normal_typed_keys
        if key[1] in explicit_category_gated or key in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    }
    reflective_normal_keys = normal_typed_keys - explicit_normal_keys - deferred_typed_keys
    assert reflective_normal_keys == set()


_B2_BOOTSTRAP_SOURCE_FILES: tuple[str, ...] = (
    "intergrax/integrations/registry/bootstrap_core.py",
    "intergrax/integrations/registry/bootstrap_extended.py",
    "intergrax/integrations/registry/bootstrap_m6_p4.py",
    "intergrax/integrations/registry/bootstrap_m6_p5.py",
    "intergrax/integrations/registry/bootstrap_m6_p6.py",
    "intergrax/integrations/registry/bootstrap_m7_p7.py",
)


def _b2_keys_from_bootstrap_sources() -> frozenset[tuple[str, str]]:
    import re

    pattern = re.compile(
        r"intergrax\.integrations\.providers\.(message_bus|notification_channel|conversation_channel|issue_tracker|wiki_knowledge|collaboration_suite)\.([a-z0-9_]+)\.register",
    )
    keys: set[tuple[str, str]] = set()
    for relative_path in _B2_BOOTSTRAP_SOURCE_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        keys.update(pattern.findall(source))
    return frozenset((slug, category) for category, slug in keys)


def test_b2_bootstrap_provider_sets_preserved() -> None:
    expected = _b2_keys_from_bootstrap_sources()
    assert expected
    for slug, category in sorted(expected):
        b2_register_function(slug, category)()
        entry = get_entry(slug)
        assert entry is not None, (slug, category)
        assert any(spec.category == category for spec in entry.contract_specs), (slug, category)
        clear_catalog()


def test_b3_contract_spec_modules_have_no_reflection() -> None:
    for slug, category in b3_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CONTRACT_SPEC_NAMES:
                    pytest.fail(f"{path}: forbidden reflective call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                pytest.fail(f"{path}: forbidden __dict__ access")


def test_b3_register_modules_do_not_import_contract_capture() -> None:
    for slug, category in b3_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        source = path.read_text(encoding="utf-8")
        assert "contract_capture" not in source, path.as_posix()
        assert "contract_specs=" in source, path.as_posix()


@pytest.mark.parametrize("slug,category", b3_provider_category_keys())
def test_b3_registration_bypasses_contract_capture(slug: str, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError(f"capture_builtin_contract_specs must not run for B3 {(slug, category)}")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    b3_register_function(slug, category)()
    entry = get_entry(slug)
    assert entry.contract_specs
    matching = [spec for spec in entry.contract_specs if spec.category == category]
    assert matching
    assert matching[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug,category", b3_provider_category_keys())
def test_b3_registration_does_not_execute_catalog_factory(
    slug: str,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    register_fn = b3_register_function(slug, category)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: IntegrationManifest,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> IntegrationManifest:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


def test_b3_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="prometheus",
        categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_PROMETHEUS",
        description="prometheus",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_contract_capture_has_no_observability_backend_factory_name_special_case() -> None:
    source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_capture.py").read_text(
        encoding="utf-8",
    )
    assert "OBSERVABILITY_BACKEND_CATEGORY" not in source
    assert "observability_integration" not in source


def test_external_fake_b3_provider_explicit_registration() -> None:
    class _ExternalB3Integration(FeatureFlagIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB3Integration:
        return _ExternalB3Integration.for_provider(
            provider_id="external_b3_flag",
            display_name="External B3 Flag",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="feature_flag",
        provider_id="external_b3_flag",
        integration_class=_ExternalB3Integration,
        contract_class=FeatureFlagIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B3 Flag",
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
        slug="external_b3_flag",
        categories=(IntegrationCategory.FEATURE_FLAG,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B3_FLAG",
        description="external fake B3 provider",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_b3_flag")
    assert registration.category == "feature_flag"
    assert registration.integration_class is _ExternalB3Integration


def test_external_fake_b3_provider_without_explicit_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B3 typed provider")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    manifest = IntegrationManifest(
        slug="external_b3_flag",
        categories=(IntegrationCategory.FEATURE_FLAG,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B3_FLAG",
        description="external fake B3 provider",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_secrets_store_registration_does_not_materialize_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.secrets_store import vault as vault_pkg

    def _must_not_create_client(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("secrets_store client must not be materialized during registration")

    monkeypatch.setattr(vault_pkg.bundle, "create_vault_secrets_store_integration", _must_not_create_client)
    monkeypatch.setattr(vault_pkg.bundle, "create_vault_secrets_store", _must_not_create_client)
    vault_pkg.register.register_vault_integration()
    entry = get_entry("vault")
    assert any(spec.category == "secrets_store" for spec in entry.contract_specs)


def test_b3_registry_v2_derives_from_explicit_specs() -> None:
    from intergrax.integrations.providers.observability_backend.prometheus.register import register_prometheus_integration

    register_prometheus_integration()
    registration = build_integration_registration("prometheus")
    assert registration.provider_id == "prometheus"
    assert registration.category == "observability_backend"


_B3_BOOTSTRAP_SOURCE_FILES: tuple[str, ...] = _B2_BOOTSTRAP_SOURCE_FILES


def _b3_keys_from_bootstrap_sources() -> frozenset[tuple[str, str]]:
    import re

    pattern = re.compile(
        r"intergrax\.integrations\.providers\.(observability_backend|cloud_platform|ci_cd|workflow_orchestrator|feature_flag|secrets_store|billing_meter)\.([a-z0-9_]+)\.register",
    )
    keys: set[tuple[str, str]] = set()
    for relative_path in _B3_BOOTSTRAP_SOURCE_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        keys.update((slug, category) for category, slug in pattern.findall(source))
    return frozenset(keys)


def test_b3_bootstrap_provider_sets_preserved() -> None:
    expected = _b3_keys_from_bootstrap_sources()
    assert expected
    for slug, category in sorted(expected):
        b3_register_function(slug, category)()
        entry = get_entry(slug)
        assert entry is not None, (slug, category)
        assert any(spec.category == category for spec in entry.contract_specs), (slug, category)
        clear_catalog()


def test_b4_category_set_matches_expected_security_runtime_media_categories() -> None:
    assert B4_TYPED_CONTRACT_CATEGORIES == frozenset(
        {
            "browser_automation",
            "security_scanner",
            "sandbox_host",
            "identity_provider",
            "model_serving_runtime",
            "speech_provider",
            "vision_serving",
            "ml_inference_host",
        }
    )


def test_b4_inventory_gate_all_typed_keys_explicit() -> None:
    b4_keys = b4_provider_category_keys()
    explicit_keys: list[tuple[str, str]] = []
    for slug, category in b4_keys:
        contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        assert contract_path.is_file(), f"missing contract_spec for {(slug, category)}"
        register_path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        register_source = register_path.read_text(encoding="utf-8")
        assert "contract_specs=CONTRACT_SPECS" in register_source
        explicit_keys.append((slug, category))
    assert len(explicit_keys) == len(b4_keys) == 21


def test_b4_typed_keys_exact_inventory() -> None:
    b4_keys = b4_provider_category_keys()
    assert b4_keys == (
        ("apify", "browser_automation"),
        ("auth0", "identity_provider"),
        ("browserbase", "browser_automation"),
        ("clerk", "identity_provider"),
        ("daytona", "sandbox_host"),
        ("deepgram", "speech_provider"),
        ("e2b", "sandbox_host"),
        ("elevenlabs", "speech_provider"),
        ("firecrawl", "browser_automation"),
        ("keycloak", "identity_provider"),
        ("modal", "sandbox_host"),
        ("okta", "identity_provider"),
        ("ollama", "model_serving_runtime"),
        ("playwright", "browser_automation"),
        ("replicate", "ml_inference_host"),
        ("selenium", "browser_automation"),
        ("semgrep", "security_scanner"),
        ("snyk", "security_scanner"),
        ("triton", "vision_serving"),
        ("trivy", "security_scanner"),
        ("workos", "identity_provider"),
    )


def test_b4_contract_spec_modules_have_no_reflection() -> None:
    for slug, category in b4_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CONTRACT_SPEC_NAMES:
                    pytest.fail(f"{path}: forbidden reflective call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                pytest.fail(f"{path}: forbidden __dict__ access")


def test_b4_register_modules_do_not_import_contract_capture() -> None:
    for slug, category in b4_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        source = path.read_text(encoding="utf-8")
        assert "contract_capture" not in source, path.as_posix()
        assert "contract_specs=" in source, path.as_posix()


@pytest.mark.parametrize("slug,category", b4_provider_category_keys())
def test_b4_registration_bypasses_contract_capture(slug: str, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError(f"capture_builtin_contract_specs must not run for B4 {(slug, category)}")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    b4_register_function(slug, category)()
    entry = get_entry(slug)
    assert entry.contract_specs
    matching = [spec for spec in entry.contract_specs if spec.category == category]
    assert matching
    assert matching[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug,category", b4_provider_category_keys())
def test_b4_registration_does_not_execute_catalog_factory(
    slug: str,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    register_fn = b4_register_function(slug, category)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: IntegrationManifest,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> IntegrationManifest:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


def test_b4_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="playwright",
        categories=(IntegrationCategory.BROWSER_AUTOMATION,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_PLAYWRIGHT",
        description="playwright",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_external_fake_b4_provider_explicit_registration() -> None:
    class _ExternalB4Integration(BrowserAutomationIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB4Integration:
        return _ExternalB4Integration.for_provider(
            provider_id="external_b4_browser",
            display_name="External B4 Browser",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="browser_automation",
        provider_id="external_b4_browser",
        integration_class=_ExternalB4Integration,
        contract_class=BrowserAutomationIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B4 Browser",
        config_class=CategoryIntegrationConfig,
        capabilities=(
            PlatformIntegrationCapability.CONNECT,
            PlatformIntegrationCapability.WRITE,
            PlatformIntegrationCapability.HEALTH_CHECK,
        ),
        security_posture=PlatformIntegrationSecurityPosture(),
        supports_runtime_binding=True,
        supports_health_check=True,
        metadata={"source": "external_plugin_test"},
    )
    manifest = IntegrationManifest(
        slug="external_b4_browser",
        categories=(IntegrationCategory.BROWSER_AUTOMATION,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B4_BROWSER",
        description="external fake B4 provider",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_b4_browser")
    assert registration.category == "browser_automation"
    assert registration.integration_class is _ExternalB4Integration


def test_external_fake_b4_provider_without_explicit_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B4 typed provider")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    manifest = IntegrationManifest(
        slug="external_b4_browser",
        categories=(IntegrationCategory.BROWSER_AUTOMATION,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B4_BROWSER",
        description="external fake B4 provider",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_multi_category_manifest_with_b4_category_without_specs_fails() -> None:
    manifest = IntegrationManifest(
        slug="multi_category_b4",
        categories=(IntegrationCategory.BROWSER_AUTOMATION, IntegrationCategory.SEARCH_PROVIDER),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B4",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_partial_explicit_specs_missing_required_b4_category_fails() -> None:
    class _SearchIntegration:
        pass

    manifest = IntegrationManifest(
        slug="multi_category_b4",
        categories=(IntegrationCategory.BROWSER_AUTOMATION, IntegrationCategory.SEARCH_PROVIDER),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B4",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="is missing explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {}, contract_specs=())


def test_llm_guardrail_remains_deferred_outside_b5_gate() -> None:
    from intergrax.integrations.providers.llm_guardrail.register_all import register_llm_guardrail_integrations
    from intergrax.runtime.integrations.registry_v2 import DEFERRED_LLM_GUARDRAIL_SLUGS

    assert "llm_guardrail" not in B5_TYPED_CONTRACT_CATEGORIES
    assert len(DEFERRED_LLM_GUARDRAIL_SLUGS) == 9
    register_llm_guardrail_integrations()
    for slug in DEFERRED_LLM_GUARDRAIL_SLUGS:
        entry = get_entry(slug)
        assert entry is not None
        assert entry.contract_specs == ()


def test_browser_automation_registration_does_not_launch_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.browser_automation import playwright as playwright_pkg

    def _must_not_launch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("browser must not launch during registration")

    monkeypatch.setattr(playwright_pkg.bundle, "create_playwright_browser_automation", _must_not_launch)
    monkeypatch.setattr(playwright_pkg.bundle, "create_playwright_browser_automation_integration", _must_not_launch)
    playwright_pkg.register.register_playwright_integration()
    entry = get_entry("playwright")
    assert any(spec.category == "browser_automation" for spec in entry.contract_specs)


def test_sandbox_registration_does_not_allocate_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.sandbox_host import e2b as e2b_pkg

    def _must_not_allocate(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sandbox must not allocate during registration")

    monkeypatch.setattr(e2b_pkg.bundle, "create_e2b_sandbox_host", _must_not_allocate)
    monkeypatch.setattr(e2b_pkg.bundle, "create_e2b_sandbox_host_integration", _must_not_allocate)
    e2b_pkg.register.register_e2b_integration()
    entry = get_entry("e2b")
    assert any(spec.category == "sandbox_host" for spec in entry.contract_specs)


def test_identity_registration_does_not_create_remote_auth_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.identity_provider import auth0 as auth0_pkg

    def _must_not_create_client(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("identity client must not be created during registration")

    monkeypatch.setattr(auth0_pkg.bundle, "create_auth0_identity_provider", _must_not_create_client)
    monkeypatch.setattr(auth0_pkg.bundle, "create_auth0_identity_provider_integration", _must_not_create_client)
    auth0_pkg.register.register_auth0_integration()
    entry = get_entry("auth0")
    assert any(spec.category == "identity_provider" for spec in entry.contract_specs)


def test_model_media_registration_does_not_start_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.model_serving_runtime import ollama as ollama_pkg
    from intergrax.integrations.providers.vision_serving import triton as triton_pkg

    def _must_not_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("model runtime must not start during registration")

    monkeypatch.setattr(ollama_pkg.bundle, "create_ollama_model_serving_runtime", _must_not_start)
    monkeypatch.setattr(ollama_pkg.bundle, "create_ollama_model_serving_runtime_integration", _must_not_start)
    ollama_pkg.register.register_ollama_integration()
    assert any(spec.category == "model_serving_runtime" for spec in get_entry("ollama").contract_specs)
    clear_catalog()

    monkeypatch.setattr(triton_pkg.bundle, "create_triton_vision_serving", _must_not_start)
    monkeypatch.setattr(triton_pkg.bundle, "create_triton_vision_serving_integration", _must_not_start)
    triton_pkg.register.register_triton_integration()
    assert any(spec.category == "vision_serving" for spec in get_entry("triton").contract_specs)


def test_b4_registry_v2_derives_from_explicit_specs() -> None:
    from intergrax.integrations.providers.browser_automation.playwright.register import register_playwright_integration

    register_playwright_integration()
    registration = build_integration_registration("playwright")
    assert registration.provider_id == "playwright"
    assert registration.category == "browser_automation"


_B4_BOOTSTRAP_SOURCE_FILES: tuple[str, ...] = _B3_BOOTSTRAP_SOURCE_FILES


def _b4_keys_from_bootstrap_sources() -> frozenset[tuple[str, str]]:
    import re

    pattern = re.compile(
        r"intergrax\.integrations\.providers\.(browser_automation|security_scanner|sandbox_host|identity_provider|model_serving_runtime|speech_provider|vision_serving|ml_inference_host)\.([a-z0-9_]+)\.register",
    )
    keys: set[tuple[str, str]] = set()
    for relative_path in _B4_BOOTSTRAP_SOURCE_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        keys.update((slug, category) for category, slug in pattern.findall(source))
    return frozenset(keys)


def test_b4_bootstrap_provider_sets_preserved() -> None:
    expected = _b4_keys_from_bootstrap_sources()
    assert expected
    for slug, category in sorted(expected):
        b4_register_function(slug, category)()
        entry = get_entry(slug)
        assert entry is not None, (slug, category)
        assert any(spec.category == category for spec in entry.contract_specs), (slug, category)
        clear_catalog()


def test_staged_non_b1b2b3b4b5_explicit_keys_remain_in_migration_set() -> None:
    assert EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS == frozenset({("openai", "managed_retrieval")})
    assert ("playwright", "browser_automation") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("trivy", "security_scanner") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("tavily", "search_provider") not in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS


def test_b5_category_set_matches_expected_search_parser_rerank_crm_categories() -> None:
    assert B5_TYPED_CONTRACT_CATEGORIES == frozenset(
        {
            "search_provider",
            "document_parser",
            "rerank_provider",
            "crm",
        }
    )


def test_b5_inventory_gate_all_typed_keys_explicit() -> None:
    b5_keys = b5_provider_category_keys()
    explicit_keys: list[tuple[str, str]] = []
    for slug, category in b5_keys:
        contract_path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        assert contract_path.is_file(), f"missing contract_spec for {(slug, category)}"
        register_path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        register_source = register_path.read_text(encoding="utf-8")
        assert "contract_specs=CONTRACT_SPECS" in register_source
        explicit_keys.append((slug, category))
    assert len(explicit_keys) == len(b5_keys) == 24


def test_b5_typed_keys_exact_inventory() -> None:
    b5_keys = b5_provider_category_keys()
    assert b5_keys == (
        ("algolia", "search_provider"),
        ("arxiv", "search_provider"),
        ("bing", "search_provider"),
        ("brave", "search_provider"),
        ("cohere_rerank", "rerank_provider"),
        ("docling", "document_parser"),
        ("exa", "search_provider"),
        ("google_cse", "search_provider"),
        ("google_places", "search_provider"),
        ("hubspot", "crm"),
        ("jina_rerank", "rerank_provider"),
        ("llamaparse", "document_parser"),
        ("openpyxl", "document_parser"),
        ("perplexity", "search_provider"),
        ("pymupdf", "document_parser"),
        ("python_docx", "document_parser"),
        ("reddit", "search_provider"),
        ("salesforce", "crm"),
        ("semantic_scholar", "search_provider"),
        ("serpapi", "search_provider"),
        ("tavily", "search_provider"),
        ("unstructured", "document_parser"),
        ("whisper", "document_parser"),
        ("yt_dlp", "document_parser"),
    )


def test_b5_contract_spec_modules_have_no_reflection() -> None:
    for slug, category in b5_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "contract_spec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CONTRACT_SPEC_NAMES:
                    pytest.fail(f"{path}: forbidden reflective call {node.func.id}()")
            if isinstance(node, ast.Attribute) and node.attr == "__dict__":
                pytest.fail(f"{path}: forbidden __dict__ access")


def test_b5_register_modules_do_not_import_contract_capture() -> None:
    for slug, category in b5_provider_category_keys():
        path = REPO_ROOT / provider_package_path(slug, category) / "register.py"
        source = path.read_text(encoding="utf-8")
        assert "contract_capture" not in source, path.as_posix()
        assert "contract_specs=" in source, path.as_posix()


@pytest.mark.parametrize("slug,category", b5_provider_category_keys())
def test_b5_registration_bypasses_contract_capture(slug: str, category: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError(f"capture_builtin_contract_specs must not run for B5 {(slug, category)}")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    b5_register_function(slug, category)()
    entry = get_entry(slug)
    assert entry.contract_specs
    matching = [spec for spec in entry.contract_specs if spec.category == category]
    assert matching
    assert matching[0].metadata.get("source") == "explicit_provider_declaration"


@pytest.mark.parametrize("slug,category", b5_provider_category_keys())
def test_b5_registration_does_not_execute_catalog_factory(
    slug: str,
    category: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_module = import_module(f"{provider_import_path(slug, category)}.register")
    register_fn = b5_register_function(slug, category)
    from intergrax.integrations.registry import plugin_register

    original_rfm = plugin_register.register_from_manifest

    def tracking_rfm(
        manifest: IntegrationManifest,
        factory: Callable[..., Any],
        **kwargs: Any,
    ) -> IntegrationManifest:
        factory_mock = MagicMock(wraps=factory)
        result = original_rfm(manifest, factory_mock, **kwargs)
        factory_mock.assert_not_called()
        return result

    monkeypatch.setattr(register_module, "register_from_manifest", tracking_rfm)
    register_fn()


def test_b5_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="tavily",
        categories=(IntegrationCategory.SEARCH_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_TAVILY",
        description="tavily",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs"):
        register_from_manifest(manifest, lambda **_: {})


def test_external_fake_b5_provider_explicit_registration() -> None:
    class _ExternalB5SearchIntegration(SearchProviderIntegrationContract):
        pass

    def _external_factory(*, enabled: bool = False) -> _ExternalB5SearchIntegration:
        return _ExternalB5SearchIntegration.for_provider(
            provider_id="external_b5_search",
            display_name="External B5 Search",
            config=CategoryIntegrationConfig(enabled=enabled),
        )

    spec = declare_integration_contract(
        category="search_provider",
        provider_id="external_b5_search",
        integration_class=_ExternalB5SearchIntegration,
        contract_class=SearchProviderIntegrationContract,
        contract_factory=_external_factory,
        display_name="External B5 Search",
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
        slug="external_b5_search",
        categories=(IntegrationCategory.SEARCH_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B5_SEARCH",
        description="external fake B5 provider",
    )
    register_from_manifest(manifest, lambda **_: {}, contract_specs=(spec,))
    registration = build_integration_registration("external_b5_search")
    assert registration.category == "search_provider"
    assert registration.integration_class is _ExternalB5SearchIntegration


def test_external_fake_b5_provider_without_explicit_specs_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _capture_must_not_run(*_args: object, **_kwargs: object) -> tuple[IntegrationContractSpec, ...]:
        raise AssertionError("capture_builtin_contract_specs must not run for external B5 typed provider")

    monkeypatch.setattr(contract_capture, "capture_builtin_contract_specs", _capture_must_not_run)
    manifest = IntegrationManifest(
        slug="external_b5_search",
        categories=(IntegrationCategory.SEARCH_PROVIDER,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_EXTERNAL_B5_SEARCH",
        description="external fake B5 provider",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_multi_category_manifest_with_b5_category_without_specs_fails() -> None:
    manifest = IntegrationManifest(
        slug="multi_category_b5",
        categories=(IntegrationCategory.SEARCH_PROVIDER, IntegrationCategory.BROWSER_AUTOMATION),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B5",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="requires explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {})


def test_partial_explicit_specs_missing_required_b5_category_fails() -> None:
    manifest = IntegrationManifest(
        slug="multi_category_b5",
        categories=(IntegrationCategory.SEARCH_PROVIDER, IntegrationCategory.BROWSER_AUTOMATION),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_MULTI_CATEGORY_B5",
        description="multi-category manifest",
    )
    with pytest.raises(ValueError, match="is missing explicit contract_specs for typed categories"):
        register_from_manifest(manifest, lambda **_: {}, contract_specs=())


def test_search_provider_registration_does_not_initialize_remote_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.search_provider import tavily as tavily_pkg

    def _must_not_search(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("search client must not initialize during registration")

    monkeypatch.setattr(tavily_pkg.bundle, "create_tavily_search_provider", _must_not_search)
    monkeypatch.setattr(tavily_pkg.bundle, "create_tavily_search_provider_integration", _must_not_search)
    tavily_pkg.register.register_tavily_integration()
    entry = get_entry("tavily")
    assert any(spec.category == "search_provider" for spec in entry.contract_specs)


def test_document_parser_registration_does_not_parse_or_load_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.document_parser import pymupdf as pymupdf_pkg

    def _must_not_parse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parser must not parse or load models during registration")

    monkeypatch.setattr(pymupdf_pkg.bundle, "create_pymupdf_document_parser", _must_not_parse)
    monkeypatch.setattr(pymupdf_pkg.bundle, "create_pymupdf_document_parser_integration", _must_not_parse)
    pymupdf_pkg.register.register_pymupdf_integration()
    entry = get_entry("pymupdf")
    assert any(spec.category == "document_parser" for spec in entry.contract_specs)


def test_rerank_provider_registration_does_not_score(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.rerank_provider import cohere_rerank as cohere_pkg

    def _must_not_score(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("rerank client must not score during registration")

    monkeypatch.setattr(cohere_pkg.bundle, "create_cohere_rerank_provider", _must_not_score)
    monkeypatch.setattr(cohere_pkg.bundle, "create_cohere_rerank_rerank_provider_integration", _must_not_score)
    cohere_pkg.register.register_cohere_rerank_integration()
    entry = get_entry("cohere_rerank")
    assert any(spec.category == "rerank_provider" for spec in entry.contract_specs)


def test_crm_registration_does_not_authenticate(monkeypatch: pytest.MonkeyPatch) -> None:
    from intergrax.integrations.providers.crm import salesforce as salesforce_pkg

    def _must_not_auth(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("CRM client must not authenticate during registration")

    monkeypatch.setattr(salesforce_pkg.bundle, "create_salesforce_crm", _must_not_auth)
    monkeypatch.setattr(salesforce_pkg.bundle, "create_salesforce_crm_integration", _must_not_auth)
    salesforce_pkg.register.register_salesforce_integration()
    entry = get_entry("salesforce")
    assert any(spec.category == "crm" for spec in entry.contract_specs)


def test_b5_registry_v2_derives_from_explicit_specs() -> None:
    from intergrax.integrations.providers.search_provider.tavily.register import register_tavily_integration

    register_tavily_integration()
    registration = build_integration_registration("tavily")
    assert registration.provider_id == "tavily"
    assert registration.category == "search_provider"


_B5_BOOTSTRAP_SOURCE_FILES: tuple[str, ...] = _B4_BOOTSTRAP_SOURCE_FILES


def _b5_keys_from_bootstrap_sources() -> frozenset[tuple[str, str]]:
    import re

    pattern = re.compile(
        r"intergrax\.integrations\.providers\.(search_provider|document_parser|rerank_provider|crm)\.([a-z0-9_]+)\.register",
    )
    keys: set[tuple[str, str]] = set()
    for relative_path in _B5_BOOTSTRAP_SOURCE_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        keys.update((slug, category) for category, slug in pattern.findall(source))
    return frozenset(keys)


def test_b5_bootstrap_provider_sets_preserved() -> None:
    expected = _b5_keys_from_bootstrap_sources()
    assert expected
    for slug, category in sorted(expected):
        b5_register_function(slug, category)()
        entry = get_entry(slug)
        assert entry is not None, (slug, category)
        assert any(spec.category == category for spec in entry.contract_specs), (slug, category)
        clear_catalog()


def test_contract_capture_remaining_surface_is_deferred_guardrails_only() -> None:
    source = (REPO_ROOT / "intergrax" / "integrations" / "registry" / "contract_capture.py").read_text(
        encoding="utf-8",
    )
    assert "DEFERRED_LLM_GUARDRAIL_SLUGS" in source
    assert "if slug in DEFERRED_LLM_GUARDRAIL_SLUGS" in source
