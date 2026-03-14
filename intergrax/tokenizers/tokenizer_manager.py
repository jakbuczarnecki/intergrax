# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.tokenizers.contracts.base_tokenizer_manager import BaseTokenizerManager
from intergrax.tokenizers.engine.tokenizer_engine import TokenizerEngine


class TokenizerManager(BaseTokenizerManager):
    """
    Public entry point for tokenization operations.

    Responsibilities
    ----------------
    - expose simplified API for token counting
    - hide engine and registry implementation
    """

    def __init__(
        self,
        engine: TokenizerEngine,
    ) -> None:
        self._engine = engine

    def count_tokens(
        self,
        text: str,
        *,
        tokenizer_id: str | None = None,
    ) -> int:
        """
        Count tokens in the provided text.

        If tokenizer_id is None, default tokenizer will be used.
        """

        return self._engine.count_tokens(
            tokenizer_id=tokenizer_id,
            text=text,
        )