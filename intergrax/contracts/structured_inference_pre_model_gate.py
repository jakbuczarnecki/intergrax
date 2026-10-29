# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-facing PRE_MODEL gate port for structured inference (NPSC42 / GR-10-R2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from intergrax.contracts.inference_profile_id import InferenceProfileId
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


@runtime_checkable
class StructuredInferencePreModelGate(Protocol):
    """Mandatory governance evaluation before provider structured I/O."""

    def __call__(
        self,
        *,
        adapter: LLMAdapter,
        messages: Sequence[ChatMessage],
        inference_profile_id: InferenceProfileId | None,
    ) -> None: ...
