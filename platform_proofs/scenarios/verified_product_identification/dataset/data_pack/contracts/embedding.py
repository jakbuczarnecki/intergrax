"""Canonical embedding artifact row."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


@dataclass(frozen=True, slots=True)
class EmbeddingDataPackRecord:
    """Provider-neutral dense embedding representation for one source offer."""

    logical_point_id: str
    source_ref: SourceRecordRef
    derivation_version: str
    semantic_text_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str | None
    embedding_dimension: int
    dense_embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.logical_point_id.strip():
            raise ValueError("logical_point_id must be non-empty")
        if not self.derivation_version.strip():
            raise ValueError("derivation_version must be non-empty")
        if not self.semantic_text_hash.strip():
            raise ValueError("semantic_text_hash must be non-empty")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be > 0")
        if len(self.dense_embedding) != self.embedding_dimension:
            msg = (
                f"dense_embedding length {len(self.dense_embedding)} "
                f"!= embedding_dimension {self.embedding_dimension}"
            )
            raise ValueError(msg)
