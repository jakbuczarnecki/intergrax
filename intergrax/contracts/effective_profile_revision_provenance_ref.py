# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference to Profile Resolution-owned effective revision identity (read-only provenance)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffectiveProfileRevisionProvenanceRef:
    """
    Opaque reference to Profile Resolution-owned effective revision identity.

    This contract does not define or validate Profile Resolution identity grammar.
    Canonical revision identity semantics remain owned by ``EffectiveProfileRevisionId``.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", validate_effective_profile_revision_provenance_ref(self.value))


def validate_effective_profile_revision_provenance_ref(value: object) -> str:
    if isinstance(value, EffectiveProfileRevisionProvenanceRef):
        return value.value
    if type(value) is not str:
        raise TypeError(
            f"EffectiveProfileRevisionProvenanceRef must be str, got {type(value).__name__}"
        )
    normalized = value.strip()
    if not normalized:
        raise ValueError("EffectiveProfileRevisionProvenanceRef must be non-empty")
    if normalized != value:
        raise ValueError("EffectiveProfileRevisionProvenanceRef must not contain surrounding whitespace")
    return normalized


__all__ = [
    "EffectiveProfileRevisionProvenanceRef",
    "validate_effective_profile_revision_provenance_ref",
]
