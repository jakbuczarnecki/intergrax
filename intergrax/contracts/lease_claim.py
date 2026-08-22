# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StaleClaimError(RuntimeError):
    """Raised when a terminal mutation uses a superseded owner or fence."""


class LeaseOwnership(BaseModel):
    """Typed ownership identity for claim/lease/fence coordination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str
    lease_expires_at: datetime
    fence: int
