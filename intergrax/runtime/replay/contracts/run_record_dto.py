# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(slots=True)
class RunRecordDTO:
    run_id: str
    started_at: float
    finished_at: Optional[float]
    status: str
    final_answer: Optional[str]
    metadata: Optional[Dict[str, Any]] = None
