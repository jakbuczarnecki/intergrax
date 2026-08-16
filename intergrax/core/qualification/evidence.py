# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical qualification evidence record shape."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

KindT = TypeVar("KindT", bound=StrEnum)


@dataclass(frozen=True, slots=True)
class QualificationEvidence(Generic[KindT]):
    """Safe, immutable evidence metadata (no secrets or raw payloads)."""

    kind: KindT
    code: str
    ref: str | None = None
    label: str | None = None
