# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Qualification evidence validity contracts (PROVIDER-QUAL-2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType
from uuid import uuid4

QualificationRunId = NewType("QualificationRunId", str)

_CANONICAL_RUN_ID_SUFFIX = re.compile(r"^[0-9a-f]{32}$")


class QualificationEvidenceValidity(StrEnum):
    """Current admission interpretation for qualification evidence."""

    CURRENT = "current"
    STALE = "stale"
    REVOKED = "revoked"


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")


def _require_aware_instant(value: object, *, field_name: str) -> None:
    if type(value) is not datetime:
        raise TypeError(f"{field_name} must be datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware datetime")


def validate_qualification_run_id(value: object) -> QualificationRunId:
    if type(value) is not str:
        raise TypeError(f"qualification_run_id must be str, got {type(value).__name__}")
    _require_non_empty_text(value, field_name="qualification_run_id")
    if not value.startswith("qual_run_"):
        raise ValueError("qualification_run_id must start with 'qual_run_'")
    suffix = value[len("qual_run_") :]
    if not _CANONICAL_RUN_ID_SUFFIX.fullmatch(suffix):
        raise ValueError("qualification_run_id suffix must match [0-9a-f]{32}")
    return QualificationRunId(value)


def new_qualification_run_id() -> QualificationRunId:
    """Mint a new execution-owned provider qualification run identity."""
    return QualificationRunId(f"qual_run_{uuid4().hex}")


@dataclass(frozen=True, slots=True)
class QualificationValidityRecord:
    """Append-only validity evaluation referencing an immutable qualification run."""

    qualification_run_id: QualificationRunId
    validity: QualificationEvidenceValidity
    evaluated_at: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        validate_qualification_run_id(self.qualification_run_id)
        if not isinstance(self.validity, QualificationEvidenceValidity):
            raise TypeError("validity must be QualificationEvidenceValidity")
        _require_aware_instant(self.evaluated_at, field_name="evaluated_at")
        if self.reason is not None:
            _require_non_empty_text(self.reason, field_name="reason")
