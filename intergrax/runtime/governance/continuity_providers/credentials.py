# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Credential boundary for continuity providers.

This module never contacts GCP or Azure, never reads a secret, and never
returns a token. Workload federation remains an unexecuted name until a later
qualified binding exists. Absence of that binding is a refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LiveCredentialRefusal(StrEnum):
    BINDING_ABSENT = "binding_absent"
    LIVE_ACQUISITION_FORBIDDEN = "live_acquisition_forbidden"
    QUALIFICATION_ABSENT = "qualification_absent"


@dataclass(frozen=True, slots=True)
class CredentialBoundaryDecision:
    """A refusal. ``token`` stays None. A token is not a cloud qualification."""

    refused: LiveCredentialRefusal
    token: None = None


class OfflineProviderRefusal(RuntimeError):
    """Cloud SDK session refused before any network call."""


def evaluate_continuity_credentials(binding: object) -> CredentialBoundaryDecision:
    """Refuse live acquisition. No branch in this function returns a token."""

    if binding is None:
        return CredentialBoundaryDecision(refused=LiveCredentialRefusal.BINDING_ABSENT)
    return CredentialBoundaryDecision(refused=LiveCredentialRefusal.QUALIFICATION_ABSENT)
