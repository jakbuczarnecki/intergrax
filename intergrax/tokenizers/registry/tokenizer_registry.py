# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Dict, Iterable

from intergrax.tokenizers.contracts.tokenizer import Tokenizer


class TokenizerRegistry:
    """
    Registry responsible for managing tokenizer providers.

    Tokenizers are Tier-0 components and therefore:
    - stateless
    - pure functional
    - runtime independent
    """

    def __init__(
        self,
        tokenizers: Iterable[Tokenizer] | None = None,
        *,
        default_tokenizer_id: str | None = None,
    ) -> None:
        self._tokenizers: Dict[str, Tokenizer] = {}
        self._default_tokenizer_id: str | None = default_tokenizer_id

        if tokenizers:
            for tokenizer in tokenizers:
                self.register(tokenizer)

    
    def register(self, tokenizer: Tokenizer) -> None:
        """
        Register tokenizer instance.
        """

        name = tokenizer.id

        if not name:
            raise ValueError("Tokenizer id must be defined")

        if name in self._tokenizers:
            raise ValueError(f"Tokenizer already registered: {name}")

        self._tokenizers[name] = tokenizer

    
    def get(self, name: str | None) -> Tokenizer:
        """
        Return tokenizer by name.
        """

        if name is None:
            tokenizer = self.default()
        else:
            tokenizer = self._tokenizers.get(name)

        if tokenizer is None:
            raise ValueError(f"Tokenizer not found: {name}")

        return tokenizer


    def default(self) -> Tokenizer:
        """
        Return explicitly configured default tokenizer (never registration-order implicit).
        """

        if not self._tokenizers:
            raise ValueError("No tokenizer registered")

        if self._default_tokenizer_id is None:
            raise ValueError("No default tokenizer configured")

        tokenizer = self._tokenizers.get(self._default_tokenizer_id)
        if tokenizer is None:
            raise ValueError(f"Default tokenizer not found: {self._default_tokenizer_id}")

        return tokenizer


    def available(self) -> Dict[str, Tokenizer]:
        """
        Return all registered tokenizers.
        """

        return dict(self._tokenizers)
