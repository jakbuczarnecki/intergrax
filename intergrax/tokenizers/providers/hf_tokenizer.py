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
        model: str = "meta-llama/Llama-2-7b-hf",
    ) -> None:
        self._model = model
        self._tokenizer = None

    def _ensure_loaded(self):

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model
            )

    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        return self._tokenizer.encode(
            text,
            add_special_tokens=False
        )

    def count_tokens(self, *, text: str) -> int:
        return len(self.encode(text))