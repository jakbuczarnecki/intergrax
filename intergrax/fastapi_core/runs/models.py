# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class CreateRunRequest:
    """
    Request to create a new run.

    Payload is opaque to API Core and will be
    passed to runtime in later stages.
    """
    payload: Dict[str, Any]


@dataclass(frozen=True)
class RunResponse:
    run_id: str
    status: RunStatus
    result: Optional[Dict[str, Any]] = None
