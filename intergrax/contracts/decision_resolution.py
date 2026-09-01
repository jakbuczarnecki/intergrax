# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System resolution contracts (DS-CORE-04).

Typed Decision Resolution semantics separate substantive decision outcomes from
execution termination. ``ACCEPTED`` remains represented by
``AuthoritativeAcceptedDecision`` (DS-CORE-02); non-accepted terminal outcomes
use ``AuthoritativeResolutionRecord``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.contracts.decision_identity import DecisionIdentity


class DecisionResolution(str, Enum):
    """Canonical substantive decision outcome for one decision scope."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class AuthoritativeResolutionRecord:
    """Terminal authoritative outcome when no Decision Version was accepted.

    ``identity.version`` is the latest evaluated decision version at resolution
    time — not an accepted version.
    """

    identity: DecisionIdentity
    resolution: DecisionResolution

    def __post_init__(self) -> None:
        if type(self.identity) is not DecisionIdentity:
            raise TypeError(
                "AuthoritativeResolutionRecord.identity must be DecisionIdentity",
            )
        if type(self.resolution) is not DecisionResolution:
            raise TypeError(
                "AuthoritativeResolutionRecord.resolution must be DecisionResolution",
            )
        if self.resolution is DecisionResolution.ACCEPTED:
            raise ValueError(
                "AuthoritativeResolutionRecord cannot represent ACCEPTED; "
                "use AuthoritativeAcceptedDecision",
            )


def validate_authoritative_resolution_record(
    record: AuthoritativeResolutionRecord,
) -> AuthoritativeResolutionRecord:
    """Re-validate an authoritative resolution record invariant."""
    if type(record) is not AuthoritativeResolutionRecord:
        raise TypeError("record must be AuthoritativeResolutionRecord")
    return record
