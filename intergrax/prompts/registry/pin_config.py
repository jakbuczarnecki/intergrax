# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class PromptPinConfig:
    """
    Mapping prompt_id -> pinned version.
    """
    pins: Dict[str, int]

    def get(self, prompt_id: str) -> Optional[int]:
        return self.pins.get(prompt_id)
