"""Unit tests for scenario-owned embedding configuration."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.rag.embedding.registry.profile import EmbeddingProfile

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    VPI_REFERENCE_EMBEDDING_DIMENSION,
    VPI_REFERENCE_EMBEDDING_MODEL,
    VPI_REFERENCE_EMBEDDING_PROVIDER,
    VpiEmbeddingConfiguration,
    VpiEmbeddingDimensionMismatchError,
    VpiIndexEmbeddingIdentity,
    load_vpi_embedding_configuration,
    validate_resolved_provider_dimension,
)

pytestmark = pytest.mark.unit


def test_reference_defaults_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPI_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("VPI_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("VPI_EMBEDDING_DIMENSION", raising=False)

    configuration = load_vpi_embedding_configuration()

    assert configuration.provider == VPI_REFERENCE_EMBEDDING_PROVIDER
    assert configuration.model == VPI_REFERENCE_EMBEDDING_MODEL
    assert configuration.expected_dimension == VPI_REFERENCE_EMBEDDING_DIMENSION


def test_vendor_swap_through_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPI_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("VPI_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("VPI_EMBEDDING_DIMENSION", "768")

    configuration = load_vpi_embedding_configuration()

    assert configuration.profile == EmbeddingProfile(
        provider="ollama",
        model="nomic-embed-text",
    )
    assert configuration.expected_dimension == 768


def test_openai_provider_string_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPI_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("VPI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("VPI_EMBEDDING_DIMENSION", "1536")

    configuration = load_vpi_embedding_configuration()

    assert configuration.provider == "openai"
    assert configuration.model == "text-embedding-3-small"
    assert configuration.expected_dimension == 1536


def test_dimension_mismatch_fails_closed() -> None:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model=VPI_REFERENCE_EMBEDDING_MODEL),
        expected_dimension=1024,
    )

    with pytest.raises(VpiEmbeddingDimensionMismatchError, match="dimension mismatch"):
        validate_resolved_provider_dimension(
            configuration=configuration,
            resolved_dimension=2560,
        )


def test_matching_dimension_passes_validation() -> None:
    configuration = VpiEmbeddingConfiguration(
        profile=EmbeddingProfile(provider="hf", model=VPI_REFERENCE_EMBEDDING_MODEL),
        expected_dimension=1024,
    )

    validate_resolved_provider_dimension(
        configuration=configuration,
        resolved_dimension=1024,
    )


def test_bootstrap_manifest_identity_from_configuration() -> None:
    configuration = load_vpi_embedding_configuration()
    identity = VpiIndexEmbeddingIdentity.from_configuration(
        configuration,
        dataset_checksum="sha256:example",
    )

    assert identity.embedding_provider == VPI_REFERENCE_EMBEDDING_PROVIDER
    assert identity.embedding_model == VPI_REFERENCE_EMBEDDING_MODEL
    assert identity.embedding_dimension == VPI_REFERENCE_EMBEDDING_DIMENSION
    assert identity.embedding_configuration_version == EMBEDDING_CONFIGURATION_VERSION
    assert identity.search_representation_derivation_version == "v2"
    assert identity.dataset_checksum == "sha256:example"


def test_invalid_dimension_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPI_EMBEDDING_DIMENSION", "0")

    with pytest.raises(ValueError, match="VPI_EMBEDDING_DIMENSION"):
        load_vpi_embedding_configuration()


def test_config_module_has_no_vendor_imports() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    module_path = (
        repo_root
        / "platform_proofs/scenarios/verified_product_identification/application/config/embedding_configuration.py"
    )
    forbidden_tokens = (
        "openai",
        "sentence_transformers",
        "ollama",
        "qdrant_client",
        "qdrant",
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.casefold() not in forbidden_tokens
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.casefold() not in forbidden_tokens
