"""Immutable contracts for bounded semantic representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


class TruncationStrategy(StrEnum):
    """Controlled compression strategy applied when representation exceeds budget."""

    PRIORITY_COMPRESSION = "priority_compression"


class RepresentationSectionKind(StrEnum):
    """Priority-ordered sections contributing to bounded semantic text."""

    IDENTITY = "identity"
    TITLE = "title"
    ATTRIBUTES = "attributes"
    DESCRIPTION = "description"
    SPECS = "specs"


@dataclass(frozen=True, slots=True)
class SemanticRepresentationPolicy:
    """Budget and preservation rules for one bounded representation profile."""

    max_characters: int | None
    max_tokens: int | None
    preserved_fields: tuple[str, ...]
    truncation_strategy: TruncationStrategy

    def __post_init__(self) -> None:
        if self.max_characters is not None and self.max_characters <= 0:
            msg = "max_characters must be > 0 when set"
            raise ValueError(msg)
        if self.max_tokens is not None and self.max_tokens <= 0:
            msg = "max_tokens must be > 0 when set"
            raise ValueError(msg)
        if self.max_characters is None and self.max_tokens is None:
            msg = "at least one of max_characters or max_tokens must be set"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SemanticSection:
    """One logical section before budget application."""

    kind: RepresentationSectionKind
    source_field: str
    text: str
    priority: int

    def __post_init__(self) -> None:
        if not self.text.strip():
            msg = "SemanticSection.text must be non-empty"
            raise ValueError(msg)
        if not self.source_field.strip():
            msg = "SemanticSection.source_field must be non-empty"
            raise ValueError(msg)
        if self.priority < 0:
            msg = "SemanticSection.priority must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SemanticRepresentationBuildOutput:
    """Bounded semantic text plus surviving section metadata."""

    result: SemanticRepresentationResult
    final_sections: tuple[SemanticSection, ...]


@dataclass(frozen=True, slots=True)
class SemanticRepresentationResult:
    """Bounded semantic text derived from one catalog record."""

    source_ref: SourceRecordRef
    original_character_count: int
    final_character_count: int
    estimated_tokens: int
    truncated: bool
    preserved_field_count: int
    representation_text: str

    def __post_init__(self) -> None:
        if self.original_character_count < 0:
            msg = "original_character_count must be >= 0"
            raise ValueError(msg)
        if self.final_character_count < 0:
            msg = "final_character_count must be >= 0"
            raise ValueError(msg)
        if self.preserved_field_count < 0:
            msg = "preserved_field_count must be >= 0"
            raise ValueError(msg)
        if self.estimated_tokens < 0:
            msg = "estimated_tokens must be >= 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RepresentationReductionMetrics:
    """Before/after reduction statistics for one representation transform."""

    before_chars: int
    after_chars: int
    reduction_ratio: float
    before_tokens: int
    after_tokens: int

    def __post_init__(self) -> None:
        if self.before_chars < 0 or self.after_chars < 0:
            msg = "character counts must be >= 0"
            raise ValueError(msg)
        if self.before_tokens < 0 or self.after_tokens < 0:
            msg = "token counts must be >= 0"
            raise ValueError(msg)
        if self.reduction_ratio < 0.0:
            msg = "reduction_ratio must be >= 0"
            raise ValueError(msg)
