# © Artur Czarnecki. All rights reserved.

"""Effective profile revision identity (P1.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import uuid4

_REVISION_ID_PREFIX = "effprof_rev_"
_REVISION_ID_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class EffectiveProfileRevisionId:
    """Immutable identity of one admitted effective profile revision.

    Distinct from semantic ``fingerprint``, ``RunId``, and ``ExecutionId``.
    """

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", validate_effective_profile_revision_id(self.value))


def validate_effective_profile_revision_id(value: object) -> str:
    if isinstance(value, EffectiveProfileRevisionId):
        return value.value
    if type(value) is not str:
        raise TypeError(
            f"EffectiveProfileRevisionId must be str, got {type(value).__name__}"
        )
    if not value.startswith(_REVISION_ID_PREFIX):
        raise ValueError(
            f"EffectiveProfileRevisionId must start with {_REVISION_ID_PREFIX!r}"
        )
    suffix = value[len(_REVISION_ID_PREFIX) :]
    if not _REVISION_ID_SUFFIX.fullmatch(suffix):
        raise ValueError("EffectiveProfileRevisionId suffix must match [0-9a-f]{32}")
    return value


def mint_effective_profile_revision_id() -> EffectiveProfileRevisionId:
    """Mint a new explicit revision identity."""
    return EffectiveProfileRevisionId(f"{_REVISION_ID_PREFIX}{uuid4().hex}")
