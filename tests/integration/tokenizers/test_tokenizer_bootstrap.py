import pytest

from intergrax.tokenizers.bootstrap.tokenizer_bootstrap import (
    create_default_tokenizer_manager,
)


pytestmark = pytest.mark.integration


def test_default_tokenizer_manager_pipeline():

    manager = create_default_tokenizer_manager()

    text = "Intergrax builds AI agent systems"

    tokens = manager.count_tokens(text)

    assert isinstance(tokens, int)
    assert tokens > 0