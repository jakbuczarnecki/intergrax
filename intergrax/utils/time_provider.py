# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class TimeProvider(ABC):
    """
    Contract for providing timezone-aware timestamps.

    Design goals:
    - single authoritative time source for the framework,
    - timezone-aware datetimes only (no naive datetime),
    - easy to swap implementations (tests, simulations, replay, per-tenant time rules).
    """

    @classmethod
    @abstractmethod
    def utc_now(cls) -> datetime:
        """
        Return the current time as a timezone-aware UTC datetime.
        """
        raise NotImplementedError


class SystemTimeProvider(TimeProvider):
    """
    Production time provider: uses the system clock.

    This is the default implementation to replace deprecated datetime.utcnow().
    """

    @classmethod
    def utc_now(cls) -> datetime:
        return datetime.now(timezone.utc)
