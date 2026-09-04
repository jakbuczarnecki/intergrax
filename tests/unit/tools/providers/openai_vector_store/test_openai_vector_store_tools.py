# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalBackend,
    ManagedRetrievalQueryRequest,
    ManagedRetrievalUploadResult,
)
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


class FakeManagedRetrievalBackend:
    """Second-provider stand-in proving tool layer is vendor-agnostic."""

    def __init__(
        self,
        *,
        answer: str = "Grounded answer.",
        file_ids: list[str] | None = None,
        upload_names: list[str] | None = None,
    ) -> None:
        self.answer = answer
        self.file_ids = list(file_ids or [])
        self.upload_names = list(upload_names or [])
        self.ensure_calls: list[str] = []
        self.clear_calls: list[str] = []
        self.query_requests: list[ManagedRetrievalQueryRequest] = []

    def ensure_store_exists(self, store_id: str) -> None:
        self.ensure_calls.append(store_id)

    def list_attached_file_ids(self, store_id: str) -> list[str]:
        return list(self.file_ids)

    def upload_folder(
        self,
        store_id: str,
        folder: str | Path,
        *,
        patterns: tuple[str, ...] | list[str],
    ) -> ManagedRetrievalUploadResult:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Directory does not exist: {folder_path}")
        _ = store_id, patterns
        if self.upload_names:
            return ManagedRetrievalUploadResult(
                uploaded_names=tuple(self.upload_names),
                failed_names=(),
            )
        folder_path = Path(folder)
        names = tuple(p.name for p in folder_path.glob("*.txt"))
        return ManagedRetrievalUploadResult(uploaded_names=names, failed_names=())

    def clear_store(self, store_id: str) -> int:
        self.clear_calls.append(store_id)
        count = len(self.file_ids)
        self.file_ids.clear()
        return count

    def query(self, request: ManagedRetrievalQueryRequest) -> str:
        self.query_requests.append(request)
        return self.answer


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_tool_catalog()
    reset_default_tools_bootstrap()
    yield
    clear_tool_catalog()
    reset_default_tools_bootstrap()


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
    backend = FakeManagedRetrievalBackend(answer="Answer from docs.")
    ctx = ToolWiringContext(
        managed_retrieval=backend,
        extras={"openai_vector_store_id": "vs_abc"},
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
    assert backend.query_requests[0].question == "What is Intergrax?"


def test_file_search_query_missing_provider() -> None:
    ctx = ToolWiringContext(extras={"openai_vector_store_id": "vs_abc"})
    out = perform_openai_file_search_query(ctx, OpenAiFileSearchQueryInput(query="test"))
    assert out.used is False
    assert out.reason == "managed_retrieval_not_configured"


def test_vector_store_clear_deletes_files() -> None:
    backend = FakeManagedRetrievalBackend(file_ids=["f1", "f2"])
    ctx = ToolWiringContext(
        managed_retrieval=backend,
        extras={"openai_vector_store_id": "vs_clear"},
    )
    out = perform_openai_vector_store_clear(ctx, OpenAiVectorStoreClearInput())
    assert out.used is True
    assert out.deleted_count == 2
    assert backend.clear_calls == ["vs_clear"]


def test_vector_store_upload_from_folder(tmp_path: Path) -> None:
    doc = tmp_path / "note.txt"
    doc.write_text("hello", encoding="utf-8")
    backend = FakeManagedRetrievalBackend()
    ctx = ToolWiringContext(
        managed_retrieval=backend,
        extras={"openai_vector_store_id": "vs_up"},
    )
    out = perform_openai_vector_store_upload(
        ctx,
        OpenAiVectorStoreUploadInput(folder_path=str(tmp_path), patterns=("*.txt",)),
    )
    assert out.used is True
    assert out.uploaded_count == 1
    assert out.file_names == ["note.txt"]


def test_handlers_delegate_to_service() -> None:
    backend = FakeManagedRetrievalBackend(answer="via handler")
    ctx = ToolWiringContext(
        managed_retrieval=backend,
        extras={"openai_vector_store_id": "vs_h"},
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


def test_provider_substitution_without_tool_changes() -> None:
    vendor_b = FakeManagedRetrievalBackend(answer="vendor-b answer")
    ctx = ToolWiringContext(
        managed_retrieval=vendor_b,
        extras={"openai_vector_store_id": "store-b"},
    )
    out = perform_openai_file_search_query(ctx, OpenAiFileSearchQueryInput(query="hello"))
    assert out.answer_text == "vendor-b answer"
    assert isinstance(vendor_b, ManagedRetrievalBackend)
