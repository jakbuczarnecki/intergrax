# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional

from intergrax.rag.answers.builders.default_prompts import default_rag_system_instruction


class PromptBuilder:
    """
    Builds final LLM prompt for RAG answering.

    Combines user query and retrieved context
    into a single prompt string.
    """

    def __init__(
        self,
        *,
        system_prompt: Optional[str] = None,
    ) -> None:

        self._system_prompt: str = system_prompt or default_rag_system_instruction()

    def build(
        self,
        *,
        query: str,
        context: str,
    ) -> str:

        return (
            f"{self._system_prompt}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{query}\n\n"
            f"Answer:"
        )