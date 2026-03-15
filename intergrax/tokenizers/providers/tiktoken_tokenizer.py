# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import tiktoken

from intergrax.tokenizers.contracts.tokenizer import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """
    Production tokenizer based on OpenAI tiktoken.

    Provides accurate token counting for GPT models and
    any encoding supported by tiktoken.
    """

    id: str = "tiktoken"

    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
    ) -> None:

        self._encoding_name = encoding_name
        self._encoding = None

    def _ensure_loaded(self):

        if self._encoding is None:
            self._encoding = tiktoken.get_encoding(self._encoding_name)

    def count_tokens(self, *, text: str) -> int:

        if not text:
            return 0

        self._ensure_loaded()

        tokens = self._encoding.encode(text)

        return len(tokens)