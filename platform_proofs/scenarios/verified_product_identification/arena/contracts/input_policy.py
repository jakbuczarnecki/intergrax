"""Provider-neutral embedding input transformation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class EmbeddingInputRole(str, Enum):
    QUERY = "QUERY"
    DOCUMENT = "DOCUMENT"


@dataclass(frozen=True, slots=True)
class EmbeddingInputPolicyRef:
    """Versioned reference to a candidate-specific input policy."""

    policy_id: str
    policy_version: str
    query_instruction_summary: str
    document_instruction_summary: str


class EmbeddingInputTransformation(Protocol):
    """Transform canonical semantic text into model-specific embedding input."""

    @property
    def policy_ref(self) -> EmbeddingInputPolicyRef:
        """Return the versioned policy metadata for evidence recording."""

    def transform(self, role: EmbeddingInputRole, canonical_text: str) -> str:
        """Apply role-specific transformation without mutating canonical text."""
