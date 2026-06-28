# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY
from intergrax.runtime.integrations.contracts import PlatformIntegrationKind

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFERRED_WITH_REASON: dict[str, str] = {
    "llm_guard": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "guardrails_ai": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "nemo_guardrails": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "openguardrails": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "presidio": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "llama_guard": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "lakera": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "azure_content_safety": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
    "bedrock_guardrails": "Shared llm_guardrail/bundles/ layout — no per-slug provider package with bundle.py",
}

_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "newrelic": "NewRelic",
    "opentelemetry_collector": "OpenTelemetryCollector",
    "aws": "Aws",
    "gcp": "Gcp",
    "azure_sql": "AzureSql",
    "cloud_sql": "CloudSql",
    "mssql": "Mssql",
    "pgvector": "Pgvector",
    "yt_dlp": "YtDlp",
    "e2b": "E2b",
    "n8n": "N8n",
    "okta": "Okta",
    "auth0": "Auth0",
}

EXPECTED_NON_OBSERVABILITY_SLUGS = frozenset(
    slug for slug, category in SLUG_CATEGORY.items() if category != "observability_backend"
)

def _discover_migrated_slugs() -> frozenset[str]:
    migrated: set[str] = set()
    for slug, category in SLUG_CATEGORY.items():
        if category == "observability_backend" or slug in DEFERRED_WITH_REASON:
            continue
        integration_path = (
            _PROJECT_ROOT
            / "intergrax"
            / "integrations"
            / "providers"
            / category
            / slug
            / "integration.py"
        )
        if integration_path.is_file():
            migrated.add(slug)
    return frozenset(migrated)


MIGRATED_SLUGS = _discover_migrated_slugs()


def _slug_to_pascal(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def _category_to_pascal(category: str) -> str:
    return "".join(part.capitalize() for part in category.split("_"))


def _class_prefix(slug: str, category: str) -> str:
    return f"{_slug_to_pascal(slug)}{_category_to_pascal(category)}"


def _provider_pkg(slug: str, category: str) -> str:
    return f"intergrax.integrations.providers.{category}.{slug}"


def _integration_module(slug: str, category: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug, category)}.integration")


def _bundle_module(slug: str, category: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug, category)}.bundle")


def _register_module(slug: str, category: str) -> Any:
    return importlib.import_module(f"{_provider_pkg(slug, category)}.register")


def _integration_class(slug: str, category: str) -> type:
    prefix = _class_prefix(slug, category)
    return getattr(_integration_module(slug, category), f"{prefix}Integration")


def _contract_factory(slug: str, category: str) -> Any:
    return getattr(_bundle_module(slug, category), f"create_{slug}_{category}_integration")


def _legacy_factory_name(slug: str, category: str) -> str:
    register_mod = _register_module(slug, category)
    register_fn = getattr(register_mod, f"register_{slug}_integration")
    source = inspect.getsource(register_fn)
    register_source = inspect.getsource(register_mod)
    contract_name = f"create_{slug}_{category}_integration"
    manifest_match = re.search(r"register_from_manifest\(\s*[^,]+,\s*(create_\w+)", source)
    if manifest_match:
        name = manifest_match.group(1)
        if name != contract_name:
            return name
    catalog_match = re.search(r"factory=(create_\w+)", source)
    if catalog_match:
        name = catalog_match.group(1)
        if name != contract_name:
            return name
    bundle_import_match = re.search(
        r"from .+\.bundle import (create_\w+)",
        register_source,
    )
    if bundle_import_match:
        name = bundle_import_match.group(1)
        if name != contract_name:
            return name
    bundle = _bundle_module(slug, category)
    for name in getattr(bundle, "__all__", ()):
        if name.startswith("create_") and name != contract_name:
            return name
    msg = f"{slug}: legacy factory not found"
    raise AssertionError(msg)


def _legacy_factory(slug: str, category: str) -> Any:
    return getattr(_bundle_module(slug, category), _legacy_factory_name(slug, category))


def _provider_id_const(slug: str, category: str) -> str:
    const_name = f"{slug.upper()}_{category.upper()}_PROVIDER_ID"
    return getattr(_integration_module(slug, category), const_name)


def _expected_integration_kind(category: str) -> str:
    return PlatformIntegrationKind(category).value


class _FakeClient:
    async def ping(self) -> None:
        return None


def test_completeness_expected_slugs_match_migrated_registry() -> None:
    expected_migrated = EXPECTED_NON_OBSERVABILITY_SLUGS - DEFERRED_WITH_REASON.keys()
    assert MIGRATED_SLUGS == expected_migrated
    assert MIGRATED_SLUGS | DEFERRED_WITH_REASON.keys() == EXPECTED_NON_OBSERVABILITY_SLUGS
    assert not (MIGRATED_SLUGS & DEFERRED_WITH_REASON.keys())


def test_deferred_list_documents_reasons() -> None:
    for slug, reason in DEFERRED_WITH_REASON.items():
        assert reason.strip(), f"missing defer reason for {slug}"
        assert SLUG_CATEGORY[slug] != "observability_backend"


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_migrated_slug_has_integration_module(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_path = (
        _PROJECT_ROOT
        / "intergrax"
        / "integrations"
        / "providers"
        / category
        / slug
        / "integration.py"
    )
    assert integration_path.is_file(), f"missing integration.py for {category}/{slug}"


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_integration_derives_from_category_contract(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    contract_cls = PROVIDER_CATEGORY_CONTRACT_REGISTRY[category]
    integration = _integration_class(slug, category).from_client(_FakeClient())

    assert isinstance(integration, contract_cls)


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_provider_id_matches_slug(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration = _integration_class(slug, category).from_client(_FakeClient())

    assert integration.provider_id == slug
    assert _provider_id_const(slug, category) == slug


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_integration_kind_matches_category(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration = _integration_class(slug, category).from_client(_FakeClient())

    assert integration.integration_kind == _expected_integration_kind(category)


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_contract_factory_exists(slug: str) -> None:
    factory = _contract_factory(slug, SLUG_CATEGORY[slug])
    assert callable(factory)


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_legacy_factory_importable(slug: str) -> None:
    assert callable(_legacy_factory(slug, SLUG_CATEGORY[slug]))


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_register_remains_legacy_compatible(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    register_fn = getattr(_register_module(slug, category), f"register_{slug}_integration")
    register_mod_source = inspect.getsource(_register_module(slug, category))
    source = inspect.getsource(register_fn)
    legacy_name = _legacy_factory_name(slug, category)
    contract_name = f"create_{slug}_{category}_integration"
    assert legacy_name in register_mod_source
    assert contract_name not in register_mod_source


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_init_lazy_api_exports_contract_symbols(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    prefix = _class_prefix(slug, category)
    pkg = importlib.import_module(_provider_pkg(slug, category))
    const_name = f"{slug.upper()}_{category.upper()}_PROVIDER_ID"
    assert getattr(pkg, const_name) == slug
    assert getattr(pkg, f"{prefix}Integration").__name__ == f"{prefix}Integration"
    assert getattr(pkg, f"{prefix}IntegrationConfig").__name__ == f"{prefix}IntegrationConfig"
    assert getattr(pkg, f"{prefix}Client").__name__ == f"{prefix}Client"
    assert callable(getattr(pkg, f"create_{slug}_{category}_integration"))
    assert callable(getattr(pkg, _legacy_factory_name(slug, category)))
    assert callable(getattr(pkg, f"register_{slug}_integration"))


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_integration_module_has_no_vendor_sdk_imports(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    path = (
        _PROJECT_ROOT
        / "intergrax"
        / "integrations"
        / "providers"
        / category
        / slug
        / "integration.py"
    )
    allowed_roots = frozenset({"__future__", "typing", "pydantic", "intergrax"})
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("from "):
            root = stripped.split()[1].split(".")[0]
            assert root in allowed_roots, f"unexpected import root {root!r} in {slug}"
        elif stripped.startswith("import ") and not stripped.startswith("import ("):
            root = stripped.split()[1].split(".")[0]
            assert root in allowed_roots, f"unexpected import root {root!r} in {slug}"


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_disabled_integration_without_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration = _contract_factory(slug, category)(enabled=False, client=None)
    assert integration.config.enabled is False
    assert integration.client is None


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_enabled_integration_without_client_raises(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    with pytest.raises(IntegrationConfigurationError, match="client"):
        _contract_factory(slug, category)(enabled=True, client=None)


@pytest.mark.parametrize("slug", sorted(MIGRATED_SLUGS))
def test_enabled_integration_with_fake_client(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    client = _FakeClient()
    integration = _contract_factory(slug, category)(enabled=True, client=client)
    assert integration.client is client
    assert integration.config.enabled is True
