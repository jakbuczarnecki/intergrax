# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(slots=True)
class TraceEventDTO:
    run_id: str
    step_id: Optional[str]
    event_type: str
    timestamp: float
    payload: Dict[str, Any]
