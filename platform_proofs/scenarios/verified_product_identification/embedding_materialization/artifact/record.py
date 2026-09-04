"""One materialized search representation row with dense embedding."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


@dataclass(frozen=True, slots=True)
class EmbeddingArtifactRecord:
    """Materialized semantic + lexical representation and dense embedding for one offer."""

    global_row_index: int
    logical_point_id: str
    catalog_id: str
    offer_id: str
    source_revision: str | None
    derivation_version: str
    semantic_text: str
    lexical_text: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    dense_embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.global_row_index < 0:
            msg = "global_row_index must be >= 0"
            raise ValueError(msg)
        if len(self.dense_embedding) != self.embedding_dimension:
            msg = (
                f"dense_embedding length {len(self.dense_embedding)} "
                f"!= embedding_dimension {self.embedding_dimension}"
            )
            raise ValueError(msg)

    def source_ref(self) -> SourceRecordRef:
        """Canonical source reference carried by the materialized artifact row."""
        return SourceRecordRef(
            offer_id=ProductOfferId(self.offer_id),
            catalog_id=self.catalog_id,
            source_revision=self.source_revision,
        )
