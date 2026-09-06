# © Artur Czarnecki. All rights reserved.

"""Typed inspection inconsistency evidence (P1.4)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class InspectionInconsistencyKind(StrEnum):
    """Explicit corruption or absence — inspection does not repair."""

    NOT_FOUND = "not_found"
    INCOMPLETE = "incomplete"
    MISSING_REVISION = "missing_revision"
    FINGERPRINT_MISMATCH = "fingerprint_mismatch"
    APPLICATION_SCOPE_MISMATCH = "application_scope_mismatch"
    TENANT_SCOPE_MISMATCH = "tenant_scope_mismatch"


class InspectionInconsistency(BaseModel):
    """Typed inconsistency between canonical facts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: InspectionInconsistencyKind
    message: str = Field(min_length=1)
    field: str | None = None
