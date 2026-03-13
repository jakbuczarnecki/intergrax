# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.tokenizers.contracts.tokenizer import Tokenizer
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer



def create_default_tokenizer(
    *,
    tokenizer: Tokenizer | None = None,
) -> Tokenizer:
    """
    Create default tokenizer used by RAG bootstrap.

    Allows dependency override.
    """

    if tokenizer is not None:
        return tokenizer

    return SimpleTokenizer()