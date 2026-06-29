# © Artur Czarnecki. All rights reserved.

"""Regression guards — non-observability integrations must not hide legacy runtime delegates."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.providers.layout import SLUG_CATEGORY

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PROVIDERS_ROOT = _PROJECT_ROOT / "intergrax" / "integrations" / "providers"

DEFERRED_LLM_GUARDRAIL_SLUGS: frozenset[str] = frozenset(
    {
        "llm_guard",
        "guardrails_ai",
        "nemo_guardrails",
        "openguardrails",
        "presidio",
        "llama_guard",
        "lakera",
        "azure_content_safety",
        "bedrock_guardrails",
    }
)

# Vector stores with intentional typed RAG inner-store bridge (pre-migrated reference pattern).
DEFERRED_VECTOR_BRIDGE_SLUGS: frozenset[str] = frozenset({"qdrant", "pinecone"})

INLINE_WAVE_SLUGS: tuple[str, ...] = tuple(
    sorted(
        slug
        for slug, category in SLUG_CATEGORY.items()
        if category not in {"observability_backend", "llm_guardrail"}
        and slug not in DEFERRED_LLM_GUARDRAIL_SLUGS
        and slug not in DEFERRED_VECTOR_BRIDGE_SLUGS
        and (_PROVIDERS_ROOT / category / slug / "integration.py").is_file()
    )
)

_FORBIDDEN_INTEGRATION_SNIPPETS: tuple[str, ...] = (
    "__pydantic_private__",
    "def _require_runtime",
    "def __getattr__(self, name",
    "_backend: Any",
    "_runtime: Any",
    "_inner: Any",
    "from_backend(",
    "from_runtime(",
    "from_inner(",
)

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


def _slug_to_class(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def _category_to_class(category: str) -> str:
    return "".join(part.capitalize() for part in category.split("_"))


def _integration_class(slug: str, category: str) -> type:
    mod = importlib.import_module(f"intergrax.integrations.providers.{category}.{slug}.integration")
    if category == "observability_backend":
        return getattr(mod, f"{_slug_to_class(slug)}ObservabilityIntegration")
    return getattr(mod, f"{_slug_to_class(slug)}{_category_to_class(category)}Integration")


def _legacy_factory(slug: str, category: str) -> Any:
    bundle = importlib.import_module(f"intergrax.integrations.providers.{category}.{slug}.bundle")
    names = [name for name in dir(bundle) if name.startswith("create_") and name.endswith("_integration") is False]
    preferred = f"create_{slug}_"
    for name in sorted(names):
        if name.startswith(preferred):
            return getattr(bundle, name)
    for name in sorted(names):
        if category.replace("_", "") in name.replace("_", ""):
            return getattr(bundle, name)
    return getattr(bundle, sorted(names)[0])


class _FakeCatalogClient:
    def connect(self) -> None:
        return None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        del sql, params

    def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        del sql, params
        return []

    def close(self) -> None:
        return None


def test_provider_integration_modules_forbid_legacy_delegation_patterns() -> None:
    violations: list[str] = []
    for slug in INLINE_WAVE_SLUGS:
        category = SLUG_CATEGORY[slug]
        path = _PROVIDERS_ROOT / category / slug / "integration.py"
        source = path.read_text(encoding="utf-8")
        for snippet in _FORBIDDEN_INTEGRATION_SNIPPETS:
            if snippet in source:
                violations.append(f"{path.relative_to(_PROJECT_ROOT).as_posix()}: {snippet!r}")
    assert violations == []


@pytest.mark.parametrize("slug", INLINE_WAVE_SLUGS)
def test_integration_has_no_runtime_delegate_surface(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = _integration_class(slug, category)
    assert not hasattr(integration_cls, "_require_runtime")
    assert "_require_runtime" not in inspect.getsource(integration_cls)
    assert "def __getattr__" not in inspect.getsource(integration_cls)
    assert getattr(integration_cls, "from_runtime", None) is None
    instance = integration_cls.from_client(_FakeCatalogClient())  # type: ignore[arg-type]
    assert getattr(instance, "_runtime", "missing") in (None, "missing")
    assert getattr(instance, "_backend", "missing") in (None, "missing")


@pytest.mark.parametrize("slug", ("duckdb", "auth0", "kafka", "filesystem"))
def test_representative_integration_requires_client_explicitly(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = _integration_class(slug, category)
    bare = integration_cls.for_provider(
        provider_id=slug,
        display_name=slug,
        config=integration_cls.model_fields["config"].default.__class__(enabled=True),  # type: ignore[attr-defined]
    )
    with pytest.raises(Exception) as exc_info:
        if category == "relational_store":
            bare.connect()
        elif category == "identity_provider":
            bare.verify_token("token")
        elif category == "message_bus":
            bare.publish("topic", b"payload")
        elif category == "object_storage":
            bare.put("key", b"data")
        else:
            pytest.fail(f"unexpected representative slug {slug!r}")
    assert "catalog client" in str(exc_info.value).lower()


@pytest.mark.parametrize("slug", ("duckdb", "auth0"))
def test_representative_legacy_factory_returns_public_integration(slug: str) -> None:
    category = SLUG_CATEGORY[slug]
    integration_cls = _integration_class(slug, category)
    client = _FakeCatalogClient() if category == "relational_store" else MagicMock()
    integration = integration_cls.from_client(client)  # type: ignore[arg-type]
    assert isinstance(integration, integration_cls)
    assert type(integration).__name__ == integration_cls.__name__
    assert not type(integration).__name__.startswith("_")
    if category == "relational_store":
        integration.connect()
        assert integration._client is client


def test_deferred_vector_bridge_slugs_use_typed_inner_not_runtime_delegate() -> None:
    for slug in sorted(DEFERRED_VECTOR_BRIDGE_SLUGS):
        category = SLUG_CATEGORY[slug]
        path = _PROVIDERS_ROOT / category / slug / "integration.py"
        source = path.read_text(encoding="utf-8")
        assert "def _require_runtime" not in source
        assert "_inner: VectorStore | None" in source
        assert "def _require_inner" in source
