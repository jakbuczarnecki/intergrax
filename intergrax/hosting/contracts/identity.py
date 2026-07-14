# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosted application identity contracts (APP-HOST-1A.1)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

_APPLICATION_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_APPLICATION_FACTORY_ID_MAX_LENGTH = 256


def normalize_application_id(value: str) -> str:
    """Normalize and validate a Tier-3 application identifier slug."""
    slug = value.strip().lower()
    if not _APPLICATION_ID_RE.match(slug):
        raise ValueError("application_id must be lowercase slug: [a-z][a-z0-9_]*")
    return slug


def validate_application_factory_id(value: str) -> str:
    """Normalize and validate a stable application factory identifier."""
    factory_id = value.strip()
    if not factory_id:
        raise ValueError("application_factory_id must not be empty")
    if len(factory_id) > _APPLICATION_FACTORY_ID_MAX_LENGTH:
        raise ValueError(
            f"application_factory_id must be at most {_APPLICATION_FACTORY_ID_MAX_LENGTH} characters"
        )
    if any(character.isspace() or ord(character) < 32 for character in factory_id):
        raise ValueError("application_factory_id must not contain whitespace or control characters")
    return factory_id


class HostedApplicationIdentity(BaseModel):
    """Public hosted-application identity without runtime-only references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    application_factory_id: str

    @field_validator("application_id")
    @classmethod
    def _validate_application_id(cls, value: str) -> str:
        return normalize_application_id(value)

    @field_validator("application_factory_id")
    @classmethod
    def _validate_application_factory_id(cls, value: str) -> str:
        return validate_application_factory_id(value)
