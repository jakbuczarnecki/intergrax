# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Low-level event_kind naming contract shared by registry and signal emitters."""

from __future__ import annotations

import re

_EVENT_KIND_RE = re.compile(
    r"^(agents|applications|platform|intergrax)\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
)


class DomainSignalError(ValueError):
    """Raised when a domain signal kind or payload is invalid."""


def validate_event_kind(kind: str) -> None:
    if not _EVENT_KIND_RE.match(kind):
        raise DomainSignalError(
            "event_kind must be a namespaced lowercase id "
            "(e.g. agents.legal.clause_flagged)"
        )
