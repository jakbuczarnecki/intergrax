# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse


class LLMStreamEventKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class LLMStreamEvent:
    """Streaming chunk emitted by ``stream_messages`` / ``stream_with_tools``."""

    kind: LLMStreamEventKind
    delta_content: str = ""
    response: LLMAdapterResponse | None = None

    @property
    def is_final(self) -> bool:
        return self.kind == LLMStreamEventKind.FINAL
