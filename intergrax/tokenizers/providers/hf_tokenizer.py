# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from transformers import AutoTokenizer

from intergrax.tokenizers.contracts.tokenizer import Tokenizer


class HFTokenizer(Tokenizer):
    """
    HuggingFace tokenizer provider.

    Supports tokenization for models available through the
    HuggingFace Transformers ecosystem (Llama, Mistral, Gemma, etc.).
    """

    id: str = "hf"

    def __init__(
        self,
        *,
        model_name: str = "meta-llama/Llama-2-7b-hf",
    ) -> None:

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

    def count_tokens(self, text: str) -> int:

        if not text:
            return 0

        tokens = self._tokenizer.encode(text)

        return len(tokens)