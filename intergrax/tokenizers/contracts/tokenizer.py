# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """
    Contract for token counting and tokenization.

    Tokenizers are deterministic Tier-0 components used by
    prompt builders, context builders and budget controllers.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Returns number of tokens produced from the input text.
        """
        raise NotImplementedError