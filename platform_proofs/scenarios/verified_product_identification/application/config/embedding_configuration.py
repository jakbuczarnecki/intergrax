"""Scenario-owned embedding configuration — swappable via ``VPI_EMBEDDING_*`` env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from intergrax.rag.embedding.registry.profile import EmbeddingProfile

from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)

EMBEDDING_CONFIGURATION_VERSION = "v1"
VPI_EMBEDDING_ENV_PREFIX = "VPI_EMBEDDING"

VPI_REFERENCE_EMBEDDING_PROVIDER = "hf"
VPI_REFERENCE_EMBEDDING_MODEL = "BAAI/bge-m3"
VPI_REFERENCE_EMBEDDING_DIMENSION = 1024


class VpiEmbeddingDimensionMismatchError(RuntimeError):
    """Raised when a provider's resolved dimension disagrees with configured expectation."""


@dataclass(frozen=True, slots=True)
class VpiEmbeddingConfiguration:
    """Immutable VPI embedding settings independent of vendor imports."""

    profile: EmbeddingProfile
    expected_dimension: int

    @property
    def provider(self) -> str:
        return self.profile.provider

    @property
    def model(self) -> str | None:
        return self.profile.model

    def __post_init__(self) -> None:
        if self.expected_dimension <= 0:
            msg = "expected_dimension must be > 0"
            raise ValueError(msg)
        if self.profile.model is None or not self.profile.model.strip():
            msg = "embedding model must be a non-empty string for VPI configuration"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class VpiIndexEmbeddingIdentity:
    """Embedding identity fields for a future scenario bootstrap manifest."""

    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    search_representation_derivation_version: str
    dataset_checksum: str
    embedding_configuration_version: str

    @classmethod
    def from_configuration(
        cls,
        configuration: VpiEmbeddingConfiguration,
        *,
        dataset_checksum: str,
        search_representation_derivation_version: str = SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version: str = EMBEDDING_CONFIGURATION_VERSION,
    ) -> VpiIndexEmbeddingIdentity:
        model = configuration.model
        if model is None:
            msg = "embedding model is required for bootstrap manifest identity"
            raise ValueError(msg)
        return cls(
            embedding_provider=configuration.provider,
            embedding_model=model,
            embedding_dimension=configuration.expected_dimension,
            search_representation_derivation_version=search_representation_derivation_version,
            dataset_checksum=dataset_checksum,
            embedding_configuration_version=embedding_configuration_version,
        )


def _parse_expected_dimension(raw_value: str | None) -> int:
    if raw_value is None or not raw_value.strip():
        return VPI_REFERENCE_EMBEDDING_DIMENSION
    try:
        parsed = int(raw_value.strip())
    except ValueError as exc:
        msg = "VPI_EMBEDDING_DIMENSION must be a positive integer"
        raise ValueError(msg) from exc
    if parsed <= 0:
        msg = "VPI_EMBEDDING_DIMENSION must be > 0"
        raise ValueError(msg)
    return parsed


def load_vpi_embedding_configuration(
    *,
    prefix: str = VPI_EMBEDDING_ENV_PREFIX,
) -> VpiEmbeddingConfiguration:
    """
    Load VPI embedding configuration from the process environment.

    Precedence for values is owned by the canonical proof environment loader
    (``load_proof_environment``): process environment wins over scenario ``.env``.
    When unset, scenario reference defaults apply (not global ``INTERGRAX_EMBEDDING_*``).
    """
    provider_raw = os.getenv(
        f"{prefix}_PROVIDER",
        VPI_REFERENCE_EMBEDDING_PROVIDER,
    )
    model_raw = os.getenv(f"{prefix}_MODEL", VPI_REFERENCE_EMBEDDING_MODEL)
    dimension_raw = os.getenv(f"{prefix}_DIMENSION")
    profile = EmbeddingProfile(provider=provider_raw, model=model_raw)
    expected_dimension = _parse_expected_dimension(dimension_raw)
    return VpiEmbeddingConfiguration(
        profile=profile,
        expected_dimension=expected_dimension,
    )


def validate_resolved_provider_dimension(
    *,
    configuration: VpiEmbeddingConfiguration,
    resolved_dimension: int,
) -> None:
    """Fail closed when provider ``dimension()`` disagrees with configured expectation."""
    if resolved_dimension <= 0:
        msg = f"provider reported invalid embedding dimension: {resolved_dimension}"
        raise VpiEmbeddingDimensionMismatchError(msg)
    if resolved_dimension != configuration.expected_dimension:
        msg = (
            "embedding dimension mismatch: configured "
            f"{configuration.expected_dimension}, provider resolved {resolved_dimension}; "
            "re-embedding and Qdrant index rebuild are required"
        )
        raise VpiEmbeddingDimensionMismatchError(msg)
