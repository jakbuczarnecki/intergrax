# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reference to Profile Resolution-owned effective revision identity (read-only provenance)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_REVISION_ID_PREFIX = "effprof_rev_"
_REVISION_ID_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class EffectiveProfileRevisionProvenanceRef:
    """
    Immutable reference to an effective profile revision identity.

    Does not mint or resolve revision truth — Profile Resolution remains authority.
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
    if not value.startswith(_REVISION_ID_PREFIX):
        raise ValueError(
            f"EffectiveProfileRevisionProvenanceRef must start with {_REVISION_ID_PREFIX!r}"
        )
    suffix = value[len(_REVISION_ID_PREFIX) :]
    if not _REVISION_ID_SUFFIX.fullmatch(suffix):
        raise ValueError("EffectiveProfileRevisionProvenanceRef suffix must match [0-9a-f]{32}")
    return value


__all__ = [
    "EffectiveProfileRevisionProvenanceRef",
    "validate_effective_profile_revision_provenance_ref",
]
