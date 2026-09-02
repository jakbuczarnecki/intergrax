# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-neutral vector index administration (control plane)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.integrations.contracts.base import HealthStatus, IntegrationError
from intergrax.rag.vectorstore.config.vector_config import Metric


class VectorSearchCapability(str, Enum):
    """Logical search channels an index may expose."""

    DENSE = "dense"
    SPARSE_LEXICAL = "sparse_lexical"


class VectorIndexPrepareOutcome(str, Enum):
    """Idempotent prepare result."""

    CREATED = "created"
    ALREADY_COMPATIBLE = "already_compatible"


class VectorIndexCompatibilityError(IntegrationError):
    """Persisted index state is incompatible with the requested specification."""


@dataclass(frozen=True, slots=True)
class VectorIndexIdentity:
    """Logical index identity before provider-specific physical naming."""

    logical_name: str
    tenant_id: str


@dataclass(frozen=True, slots=True)
class DenseVectorChannelSpec:
    """Dense vector channel requirements."""

    channel_name: str
    dimension: int
    metric: Metric


@dataclass(frozen=True, slots=True)
class SparseLexicalChannelSpec:
    """Sparse lexical channel requirements."""

    channel_name: str


@dataclass(frozen=True, slots=True)
class VectorIndexSpec:
    """Provider-neutral required index shape."""

    identity: VectorIndexIdentity
    dense: DenseVectorChannelSpec
    required_capabilities: frozenset[VectorSearchCapability]
    sparse_lexical: SparseLexicalChannelSpec | None = None

    def __post_init__(self) -> None:
        if self.dense.dimension <= 0:
            raise ValueError("dense.dimension must be positive")
        if not self.dense.channel_name.strip():
            raise ValueError("dense.channel_name must be non-empty")
        if VectorSearchCapability.SPARSE_LEXICAL in self.required_capabilities:
            if self.sparse_lexical is None:
                raise ValueError(
                    "sparse_lexical spec is required when SPARSE_LEXICAL capability is requested"
                )
            if not self.sparse_lexical.channel_name.strip():
                raise ValueError("sparse_lexical.channel_name must be non-empty")


@dataclass(frozen=True, slots=True)
class VectorIndexDescription:
    """Immutable persisted index description without vendor leakage."""

    identity: VectorIndexIdentity
    exists: bool
    reachable: bool
    point_count: int
    dense_dimension: int | None
    present_capabilities: frozenset[VectorSearchCapability]
    dense_channel_name: str | None
    sparse_lexical_channel_name: str | None


@dataclass(frozen=True, slots=True)
class VectorIndexPrepareResult:
    """Typed prepare outcome plus post-prepare description."""

    outcome: VectorIndexPrepareOutcome
    description: VectorIndexDescription


def validate_spec_against_description(
    spec: VectorIndexSpec,
    description: VectorIndexDescription,
) -> None:
    """Fail closed when persisted state does not satisfy ``spec``."""
    if not description.exists:
        raise VectorIndexCompatibilityError(
            f"vector index {description.identity.logical_name!r} does not exist"
        )
    if description.dense_dimension is None:
        raise VectorIndexCompatibilityError(
            f"vector index {description.identity.logical_name!r} has no dense channel"
        )
    if description.dense_dimension != spec.dense.dimension:
        raise VectorIndexCompatibilityError(
            "vector index dense dimension "
            f"{description.dense_dimension} != expected {spec.dense.dimension}"
        )
    missing = spec.required_capabilities - description.present_capabilities
    if missing:
        missing_labels = ", ".join(sorted(capability.value for capability in missing))
        raise VectorIndexCompatibilityError(
            f"vector index missing required capabilities: {missing_labels}"
        )


@runtime_checkable
class VectorIndexAdministration(Protocol):
    """Control-plane port for vector/search index lifecycle and inspection."""

    def probe(self) -> HealthStatus: ...

    def describe_index(self, identity: VectorIndexIdentity) -> VectorIndexDescription: ...

    def prepare_index(self, spec: VectorIndexSpec) -> VectorIndexPrepareResult: ...

    def close(self) -> None: ...


__all__ = [
    "DenseVectorChannelSpec",
    "SparseLexicalChannelSpec",
    "VectorIndexAdministration",
    "VectorIndexCompatibilityError",
    "VectorIndexDescription",
    "VectorIndexIdentity",
    "VectorIndexPrepareOutcome",
    "VectorIndexPrepareResult",
    "VectorIndexSpec",
    "VectorSearchCapability",
    "validate_spec_against_description",
]
