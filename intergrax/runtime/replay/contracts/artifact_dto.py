# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(slots=True)
class ArtifactDTO:
    artifact_id: str
    name: Optional[str]
    type: Optional[str]
    produced_by_step: Optional[str]
    metadata: Optional[Dict[str, Any]]
