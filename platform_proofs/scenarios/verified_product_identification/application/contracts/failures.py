"""Provider-neutral catalog/search failure semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CatalogSearchFailureKind(StrEnum):
    """Bounded failure kinds for catalog search ports."""

    INVALID_QUERY = "invalid_query"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class CatalogSearchFailure:
    kind: CatalogSearchFailureKind
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("CatalogSearchFailure.message must be a non-empty string")
