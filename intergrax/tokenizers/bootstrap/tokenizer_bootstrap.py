# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.tokenizers.contracts.base_tokenizer_manager import BaseTokenizerManager
from intergrax.tokenizers.contracts.tokenizer import Tokenizer
from intergrax.tokenizers.engine.tokenizer_engine import TokenizerEngine
from intergrax.tokenizers.providers.hf_tokenizer import HFTokenizer
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tokenizers.tokenizer_manager import TokenizerManager


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

    return TiktokenTokenizer()


def create_default_tokenizer_engine(
    registry: TokenizerRegistry | None = None,
) -> TokenizerEngine:

    if registry is None:
        registry = TokenizerRegistry()

        registry.register(SimpleTokenizer())
        registry.register(TiktokenTokenizer())
        registry.register(HFTokenizer())

    return TokenizerEngine(
        registry=registry,
    )

def create_default_tokenizer_manager(
    engine: TokenizerEngine | None = None
)-> BaseTokenizerManager:
    
    if engine is None:
        engine = create_default_tokenizer_engine()
    
    manager = TokenizerManager(
        engine=engine
    )

    return manager