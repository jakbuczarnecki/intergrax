# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry

class TokenizerEngine:
    
    def __init__(
        self,
        registry: TokenizerRegistry,
    ) -> None:
        self._registry = registry


    def count_tokens(self, tokenizer_id: str, text: str) -> int:
        tokenizer = self._registry.get(tokenizer_id)
        return tokenizer.count_tokens(text)