# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.tokenizers.contracts.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    """
    Minimal whitespace tokenizer.

    Used as default fallback tokenizer for:
    - bootstrap
    - unit tests
    - local development
    """

    def count_tokens(self, text: str) -> int:

        if not text:
            return 0

        return len(text.split())