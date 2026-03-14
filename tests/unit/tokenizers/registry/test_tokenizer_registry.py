import pytest

from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer


pytestmark = pytest.mark.unit


def test_registry_register_and_get():

    registry = TokenizerRegistry()

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    result = registry.get("simple")

    assert result is tokenizer


def test_registry_default():

    registry = TokenizerRegistry()

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    assert registry.get(None) is tokenizer


def test_registry_duplicate_registration():

    registry = TokenizerRegistry()

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    with pytest.raises(ValueError):
        registry.register(tokenizer)