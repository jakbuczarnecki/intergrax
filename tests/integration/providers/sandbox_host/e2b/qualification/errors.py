# © Artur Czarnecki. All rights reserved.

"""Qualification harness errors — E2B credential boundary."""

from __future__ import annotations

from tests.integration.providers.sandbox_host.qualification.errors import QualificationError


class QualificationCredentialUnavailable(QualificationError):
    """E2B credentials are not available for physical qualification."""
