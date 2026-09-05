# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Minimal cross-domain provenance fields (CAPABILITY-CATALOG-1 Stage 1)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from intergrax.contracts.capability_catalog._validation import normalize_optional_text
from intergrax.contracts.capability_catalog.identity import CapabilitySourceIdentity

SCHEMA_CAPABILITY_PROVENANCE_V1: Final = "capability_provenance.v1"


class CapabilityProvenance(BaseModel):
    """Optional provenance fields — absent domain data stays absent (no empty masking)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_provenance.v1"] = SCHEMA_CAPABILITY_PROVENANCE_V1
    source: CapabilitySourceIdentity
    version_label: str | None = None
    package_reference: str | None = None
    content_digest: str | None = None
    publisher: str | None = None

    @field_validator(
        "version_label",
        "package_reference",
        "content_digest",
        "publisher",
    )
    @classmethod
    def _validate_optional_text(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        return normalize_optional_text(value, label=str(info.field_name))
