# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalQueryRequest,
    ManagedRetrievalQueryError,
    ManagedRetrievalResourceNotFoundError,
)
from intergrax.integrations.providers.managed_retrieval.openai.adapter import (
    OpenAIManagedRetrievalAdapter,
)
from intergrax.integrations.providers.managed_retrieval.openai.config import (
    OpenAIManagedRetrievalConfig,
)

pytestmark = pytest.mark.unit


def _fake_openai_client(
    *,
    answer: str = "adapter answer",
    file_ids: list[str] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.vector_stores.retrieve.return_value = SimpleNamespace(id="vs_test")
    page = SimpleNamespace(
        data=[SimpleNamespace(id=fid) for fid in (file_ids or [])],
        has_more=False,
    )
    client.vector_stores.files.list.return_value = page
    client.responses.create.return_value = SimpleNamespace(output_text=answer)

    def _files_create(*, file, purpose):  # noqa: ANN001
        _ = file, purpose
        return SimpleNamespace(id="file-new")

    def _files_retrieve(file_id: str):  # noqa: ARG001
        return SimpleNamespace(status="processed")

    client.files.create.side_effect = _files_create
    client.files.retrieve.side_effect = _files_retrieve
    return client


def test_adapter_maps_query_to_responses_file_search() -> None:
    client = _fake_openai_client(answer="mapped answer")
    adapter = OpenAIManagedRetrievalAdapter(
        client,
        config=OpenAIManagedRetrievalConfig(api_key="test-key"),
    )
    answer = adapter.query(
        ManagedRetrievalQueryRequest(
            store_id="vs_1",
            question="What?",
            model="gpt-4o-mini",
            instructions="strict",
            max_results=5,
            score_threshold=0.3,
        )
    )
    assert answer == "mapped answer"
    client.responses.create.assert_called_once()
    call_kwargs = client.responses.create.call_args.kwargs
    assert call_kwargs["tools"][0]["type"] == "file_search"
    assert call_kwargs["tools"][0]["vector_store_ids"] == ["vs_1"]


def test_adapter_resource_not_found() -> None:
    client = _fake_openai_client()
    client.vector_stores.retrieve.side_effect = RuntimeError("missing")
    adapter = OpenAIManagedRetrievalAdapter(
        client,
        config=OpenAIManagedRetrievalConfig(api_key="test-key"),
    )
    with pytest.raises(ManagedRetrievalResourceNotFoundError):
        adapter.ensure_store_exists("vs_missing")


def test_adapter_query_failure() -> None:
    client = _fake_openai_client()
    client.responses.create.side_effect = RuntimeError("boom")
    adapter = OpenAIManagedRetrievalAdapter(
        client,
        config=OpenAIManagedRetrievalConfig(api_key="test-key"),
    )
    with pytest.raises(ManagedRetrievalQueryError):
        adapter.query(
            ManagedRetrievalQueryRequest(
                store_id="vs_1",
                question="q",
                model="gpt-4o-mini",
                instructions="i",
                max_results=3,
                score_threshold=0.2,
            )
        )
