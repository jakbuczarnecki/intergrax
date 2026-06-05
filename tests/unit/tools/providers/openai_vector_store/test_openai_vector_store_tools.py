# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.openai_vector_store.contracts import (
    OpenAiFileSearchQueryInput,
    OpenAiVectorStoreClearInput,
    OpenAiVectorStoreUploadInput,
)
from intergrax.tools.providers.openai_vector_store.handlers import (
    OpenAiFileSearchQueryHandler,
    OpenAiVectorStoreClearHandler,
    OpenAiVectorStoreUploadHandler,
)
from intergrax.tools.providers.openai_vector_store.service import (
    OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
    OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
    OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
    perform_openai_file_search_query,
    perform_openai_vector_store_clear,
    perform_openai_vector_store_upload,
)
from intergrax.tools.providers.openai_vector_store.bundle import (
    register_openai_vector_store_tools,
)
from intergrax.tools.registry.bootstrap import register_default_tools, reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog, get_bundle, list_catalog_tool_ids
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


def _fake_client(
    *,
    answer: str = "Grounded answer.",
    file_ids: list[str] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.vector_stores.retrieve.return_value = SimpleNamespace(id="vs_test")
    page = SimpleNamespace(data=[SimpleNamespace(id=fid) for fid in (file_ids or [])], has_more=False)
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


def test_bundle_registers_three_tools() -> None:
    register_default_tools()
    entry = get_bundle("openai_vector_store")
    assert entry.tool_ids == (
        OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
        OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
        OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
    )
    assert OPENAI_FILE_SEARCH_QUERY_TOOL_ID in list_catalog_tool_ids()


def test_file_search_query_success() -> None:
    ctx = ToolWiringContext(
        extras={
            "openai_client": _fake_client(answer="Answer from docs."),
            "openai_vector_store_id": "vs_abc",
        }
    )
    out = perform_openai_file_search_query(
        ctx,
        OpenAiFileSearchQueryInput(query="What is Intergrax?"),
    )
    assert out.used is True
    assert out.answer_text == "Answer from docs."
    assert out.context_text == "Answer from docs."
    assert out.vector_store_id == "vs_abc"
    assert out.reason == "ok"


def test_file_search_query_missing_client() -> None:
    ctx = ToolWiringContext(extras={"openai_vector_store_id": "vs_abc"})
    out = perform_openai_file_search_query(ctx, OpenAiFileSearchQueryInput(query="test"))
    assert out.used is False
    assert out.reason == "openai_client_not_configured"


def test_vector_store_clear_deletes_files() -> None:
    client = _fake_client(file_ids=["f1", "f2"])
    ctx = ToolWiringContext(
        extras={"openai_client": client, "openai_vector_store_id": "vs_clear"},
    )
    out = perform_openai_vector_store_clear(ctx, OpenAiVectorStoreClearInput())
    assert out.used is True
    assert out.deleted_count == 2
    assert client.vector_stores.files.delete.call_count == 2
    assert client.files.delete.call_count == 2


def test_vector_store_upload_from_folder(tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello", encoding="utf-8")
    client = _fake_client()
    ctx = ToolWiringContext(
        extras={"openai_client": client, "openai_vector_store_id": "vs_up"},
    )
    out = perform_openai_vector_store_upload(
        ctx,
        OpenAiVectorStoreUploadInput(folder_path=str(tmp_path), patterns=("*.txt",)),
    )
    assert out.used is True
    assert out.uploaded_count == 1
    assert out.file_names == ["note.txt"]
    client.files.create.assert_called_once()
    client.vector_stores.files.create.assert_called_once()


def test_handlers_delegate_to_service() -> None:
    client = _fake_client(answer="via handler")
    ctx = ToolWiringContext(
        extras={"openai_client": client, "openai_vector_store_id": "vs_h"},
    )
    registry = ToolRegistry()
    register_openai_vector_store_tools(registry, ctx)
    handler = OpenAiFileSearchQueryHandler(ctx)
    out = handler.execute(
        ToolExecutionRequest(
            run_id="run-1",
            step_id="step-1",
            tool_id=OPENAI_FILE_SEARCH_QUERY_TOOL_ID,
            input=OpenAiFileSearchQueryInput(query="q"),
        )
    )
    assert out.used is True
    assert out.answer_text == "via handler"

    clear_handler = OpenAiVectorStoreClearHandler(ctx)
    clear_out = clear_handler.execute(
        ToolExecutionRequest(
            run_id="run-1",
            step_id="step-2",
            tool_id=OPENAI_VECTOR_STORE_CLEAR_TOOL_ID,
            input=OpenAiVectorStoreClearInput(),
        )
    )
    assert clear_out.used is True

    upload_handler = OpenAiVectorStoreUploadHandler(ctx)
    upload_out = upload_handler.execute(
        ToolExecutionRequest(
            run_id="run-1",
            step_id="step-3",
            tool_id=OPENAI_VECTOR_STORE_UPLOAD_TOOL_ID,
            input=OpenAiVectorStoreUploadInput(folder_path="/nonexistent"),
        )
    )
    assert upload_out.used is False
    assert upload_out.reason == "folder_not_found"
