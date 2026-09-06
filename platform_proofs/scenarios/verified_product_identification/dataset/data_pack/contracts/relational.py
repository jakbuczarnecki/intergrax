"""Canonical relational artifact row."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


@dataclass(frozen=True, slots=True)
class RelationalDataPackRecord:
    """Provider-neutral relational representation for one source offer."""

    global_row_index: int
    source_ref: SourceRecordRef
    record_json: str
    derivation_version: str
    semantic_text: str
    semantic_text_hash: str
    title: str | None
    brand: str | None
    category: str | None
    description: str | None
    has_identifiers: bool
    has_spec_table: bool
    has_structured_attributes: bool

    def __post_init__(self) -> None:
        if self.global_row_index < 0:
            raise ValueError("global_row_index must be >= 0")
        if not self.record_json.strip():
            raise ValueError("record_json must be non-empty")
        if not self.derivation_version.strip():
            raise ValueError("derivation_version must be non-empty")
        if not self.semantic_text_hash.strip():
            raise ValueError("semantic_text_hash must be non-empty")
