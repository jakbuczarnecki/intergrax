# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LLMStructuredResult(Generic[T]):
    """Structured output paired with the raw adapter response envelope."""

    parsed: T
    response: LLMAdapterResponse
