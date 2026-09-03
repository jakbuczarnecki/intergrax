# © Artur Czarnecki. All rights reserved.

"""P2-003 — explicit integration contract declaration architecture gates."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.conversation_channel.slack.contract_spec import (
    CONTRACT_SPEC as SLACK_CONVERSATION_CONTRACT_SPEC,
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
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.contract_spec import (
    EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS,
    IntegrationContractSpec,
    declare_integration_contract,
)
from intergrax.integrations.registry.plugin_register import register_from_manifest
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig
from intergrax.runtime.integrations.contracts import PlatformIntegrationCapability, PlatformIntegrationSecurityPosture
from intergrax.runtime.integrations.registry_v2 import build_integration_registration

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def test_migrated_builtin_without_explicit_specs_fails_closed() -> None:
    manifest = IntegrationManifest(
        slug="postgresql",
        categories=(IntegrationCategory.RELATIONAL_STORE,),
        status=IntegrationStatus.BETA,
        env_prefix="INTERGRAX_POSTGRESQL",
        description="postgresql",
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


def test_multi_category_slack_keys_require_explicit_per_category() -> None:
    assert ("slack", "notification_channel") in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS
    assert ("slack", "conversation_channel") in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS


def test_plugin_register_has_no_provider_module_scanning() -> None:
    source = (
        REPO_ROOT / "intergrax" / "integrations" / "registry" / "plugin_register.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"vars", "import_module"}:
                pytest.fail("plugin_register must not scan provider modules reflectively")


def test_contract_capture_not_imported_by_migrated_register_modules() -> None:
    migrated_registers = [
        "intergrax/integrations/providers/relational_store/postgresql/register.py",
        "intergrax/integrations/providers/managed_retrieval/openai/register.py",
        "intergrax/integrations/providers/conversation_channel/slack/register.py",
        "intergrax/integrations/providers/notification_channel/slack/register.py",
        "intergrax/integrations/providers/observability_backend/langfuse/register.py",
    ]
    for relative in migrated_registers:
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "contract_capture" not in source
        assert "contract_specs=" in source


def test_explicit_inventory_counts_gate() -> None:
    assert len(EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS) == 5
