import pytest

from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer


pytestmark = pytest.mark.unit


def test_registry_register_and_get():

    registry = TokenizerRegistry()

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    result = registry.get("simple")

    assert result is tokenizer


def test_registry_default_requires_explicit_default_id():

    registry = TokenizerRegistry(default_tokenizer_id="simple")

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    assert registry.get(None) is tokenizer


def test_registry_default_not_configured_fails_closed():

    registry = TokenizerRegistry()

    registry.register(SimpleTokenizer())

    with pytest.raises(ValueError, match="No default tokenizer configured"):
        registry.default()


def test_registry_registration_order_does_not_select_default():

    registry = TokenizerRegistry(default_tokenizer_id="tiktoken")
    registry.register(SimpleTokenizer())
    registry.register(TiktokenTokenizer())

    assert registry.get(None).id == "tiktoken"

    registry_reversed = TokenizerRegistry(default_tokenizer_id="tiktoken")
    registry_reversed.register(TiktokenTokenizer())
    registry_reversed.register(SimpleTokenizer())

    assert registry_reversed.get(None).id == "tiktoken"


def test_registry_duplicate_registration():

    registry = TokenizerRegistry()

    tokenizer = SimpleTokenizer()

    registry.register(tokenizer)

    with pytest.raises(ValueError):
        registry.register(tokenizer)
