# © Artur Czarnecki. All rights reserved.

"""Product-level Ask scope contracts for source-scoped indexed retrieval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_KNOWLEDGE_ITEM_IDS = 20
_MAX_KNOWLEDGE_ITEM_ID_LEN = 256
_MAX_SOURCE_IDS = 20
_MAX_SOURCE_ID_LEN = 256


def _dedupe_sorted(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in sorted(values):
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


class KnowledgeAskScopeV1(BaseModel):
    """Application-level Ask scope using canonical knowledge_item_id references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_item_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_KNOWLEDGE_ITEM_IDS,
    )

    @field_validator("knowledge_item_ids")
    @classmethod
    def _validate_knowledge_item_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw in value:
            item_id = str(raw).strip()
            if not item_id:
                raise ValueError("knowledge_item_id must not be blank")
            if len(item_id) > _MAX_KNOWLEDGE_ITEM_ID_LEN:
                raise ValueError("knowledge_item_id exceeds maximum length")
            normalized.append(item_id)
        if not normalized:
            raise ValueError("knowledge scope must not be empty")
        return _dedupe_sorted(tuple(normalized))


class KnowledgeRetrievalScopeV1(BaseModel):
    """Validated retrieval authorization scope derived from indexed inventory items."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_source_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=_MAX_SOURCE_IDS,
    )

    @classmethod
    def from_validated_source_ids(
        cls,
        source_ids: tuple[str, ...],
    ) -> KnowledgeRetrievalScopeV1:
        normalized: list[str] = []
        for raw in source_ids:
            source_id = str(raw).strip()
            if not source_id:
                raise ValueError("source_id must not be blank")
            if len(source_id) > _MAX_SOURCE_ID_LEN:
                raise ValueError("source_id exceeds maximum length")
            normalized.append(source_id)
        if not normalized:
            raise ValueError("retrieval scope must not be empty")
        return cls(allowed_source_ids=_dedupe_sorted(tuple(normalized)))


class KnowledgeAskScopeError(RuntimeError):
    """Fail-closed Ask scope validation error with a bounded product code."""

    def __init__(self, error_code: str, message: str = "") -> None:
        self.error_code = error_code
        self.message = message or error_code
        super().__init__(self.message)
