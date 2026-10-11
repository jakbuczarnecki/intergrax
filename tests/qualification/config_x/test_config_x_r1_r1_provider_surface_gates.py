# © Artur Czarnecki. All rights reserved.

"""CONFIG-X-FINAL-R1-R1 — vector provider configuration surface remediation gates."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.chroma.config import (
    ENV_CHROMA_TENANT_ID,
    ChromaIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.pinecone.config import (
    ENV_PINECONE_TENANT_ID,
    PineconeIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.qdrant.config import (
    ENV_QDRANT_TENANT_ID,
    QdrantIntegrationConfig,
)
from intergrax.integrations.providers.vector_store.vespa.config import (
    ENV_VESPA_TENANT,
    ENV_VESPA_URL,
    VespaIntegrationConfig,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_VECTOR_CONFIG_PATHS = (
    "intergrax/integrations/providers/vector_store/chroma/config.py",
    "intergrax/integrations/providers/vector_store/qdrant/config.py",
    "intergrax/integrations/providers/vector_store/pinecone/config.py",
    "intergrax/integrations/providers/vector_store/vespa/config.py",
    "intergrax/integrations/providers/vector_store/weaviate/schema.py",
)


@pytest.mark.parametrize("rel_path", _VECTOR_CONFIG_PATHS)
def test_vector_provider_surfaces_no_ambient_default_tenant_literal(rel_path: str) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    text = (root / rel_path).read_text(encoding="utf-8")
    assert 'tenant_id: str = "default"' not in text
    assert 'DEFAULT_TENANT_ID = "default"' not in text
    assert ', "default").strip() or "default"' not in text


def test_qdrant_from_env_missing_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_QDRANT_TENANT_ID, raising=False)
    with pytest.raises(IntegrationConfigurationError, match=ENV_QDRANT_TENANT_ID):
        QdrantIntegrationConfig.from_env()


def test_chroma_from_env_missing_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_CHROMA_TENANT_ID, raising=False)
    with pytest.raises(IntegrationConfigurationError, match=ENV_CHROMA_TENANT_ID):
        ChromaIntegrationConfig.from_env()


def test_pinecone_from_env_missing_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_PINECONE_TENANT_ID, raising=False)
    with pytest.raises(IntegrationConfigurationError, match=ENV_PINECONE_TENANT_ID):
        PineconeIntegrationConfig.from_env()


def test_vespa_from_env_missing_tenant_or_url_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_VESPA_TENANT, raising=False)
    monkeypatch.delenv(ENV_VESPA_URL, raising=False)
    with pytest.raises(IntegrationConfigurationError, match=ENV_VESPA_TENANT):
        VespaIntegrationConfig.from_env()
    monkeypatch.setenv(ENV_VESPA_TENANT, "tenant-a")
    with pytest.raises(IntegrationConfigurationError, match=ENV_VESPA_URL):
        VespaIntegrationConfig.from_env()


def test_explicit_vector_provider_tenant_still_materializes() -> None:
    cfg = QdrantIntegrationConfig(
        collection_name="coll",
        tenant_id="tenant-explicit",
        url="http://127.0.0.1:6333",
    )
    assert cfg.require_tenant_id() == "tenant-explicit"
