from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from intergrax.rag.embedding.providers._openai_compatible import (
    embed_openai_compatible,
)


pytestmark = pytest.mark.unit


def response_item(index: int, embedding: list[float]) -> SimpleNamespace:
    return SimpleNamespace(index=index, embedding=embedding)


def test_transport_batches_and_orders_response() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            response_item(2, [7.0, 8.0]),
            response_item(0, [1.0, 2.0]),
            response_item(1, [4.0, 5.0]),
        ]
    )

    vectors = embed_openai_compatible(
        client,
        model="model",
        texts=["a", "b", "c"],
    )

    np.testing.assert_array_equal(vectors, [[1, 2], [4, 5], [7, 8]])
    assert vectors.shape == (3, 2)
    assert vectors.dtype == np.float32
    client.embeddings.create.assert_called_once_with(
        model="model",
        input=["a", "b", "c"],
    )


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (
            [response_item(0, [1.0])],
            "count",
        ),
        (
            [response_item(0, [1.0]), response_item(0, [2.0])],
            "Duplicate",
        ),
        (
            [SimpleNamespace(embedding=[1.0]), response_item(1, [2.0])],
            "invalid index",
        ),
        (
            [response_item(0, [1.0]), response_item(3, [2.0])],
            "out of range",
        ),
    ],
)
def test_transport_rejects_malformed_response(
    items: list[SimpleNamespace],
    message: str,
) -> None:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(data=items)

    with pytest.raises(ValueError, match=message):
        embed_openai_compatible(client, model="model", texts=["a", "b"])


def test_transport_rejects_ragged_vectors() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            response_item(0, [1.0, 2.0]),
            response_item(1, [3.0]),
        ]
    )

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        embed_openai_compatible(client, model="model", texts=["a", "b"])


def test_transport_rejects_non_numeric_vector() -> None:
    client = MagicMock()
    client.embeddings.create.return_value = SimpleNamespace(
        data=[response_item(0, ["not-a-number"])]
    )

    with pytest.raises(ValueError, match="numeric"):
        embed_openai_compatible(client, model="model", texts=["a"])


def test_embedding_providers_import_without_langchain() -> None:
    script = """
import importlib.abc
import sys

class LangChainBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "langchain_openai" or fullname.startswith(
            ("langchain_core", "langchain_community")
        ):
            raise AssertionError(f"blocked import: {fullname}")
        return None

sys.meta_path.insert(0, LangChainBlocker())
from intergrax.rag.embedding.providers.llama_cpp_embedding_provider import (
    LlamaCppEmbeddingProvider,
)
from intergrax.rag.embedding.providers.openai_embedding_provider import (
    OpenAIEmbeddingProvider,
)
from intergrax.rag.embedding.providers.vllm_embedding_provider import (
    VllmEmbeddingProvider,
)

assert OpenAIEmbeddingProvider().provider_name() == "openai"
assert VllmEmbeddingProvider().provider_name() == "vllm"
assert LlamaCppEmbeddingProvider().provider_name() == "llama_cpp"
"""
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
