# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_builder import ContextBuilder
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID


@pytest.mark.gate
def test_context_builder_enables_rag_via_allowed_tools() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    session = ChatSession(id="s1", tenant_id="t1")
    request = RuntimeRequest(
        agent_id="a",
        user_id="u1",
        session_id="s1",
        message="q",
        tenant_id="t1",
        metadata={"allowed_tools": [RAG_RETRIEVE_TOOL_ID]},
    )
    use_rag, reason = builder._should_use_rag(session, request)
    assert use_rag is True
    assert reason == "rag_via_allowed_tools"


@pytest.mark.gate
def test_context_builder_disables_rag_when_tool_list_excludes_retrieve() -> None:
    config = RuntimeConfig(llm_adapter=MagicMock(), enable_rag=True)
    builder = ContextBuilder(config, MagicMock())
    session = ChatSession(id="s1", tenant_id="t1")
    request = RuntimeRequest(
        agent_id="a",
        user_id="u1",
        session_id="s1",
        message="q",
        tenant_id="t1",
        metadata={"allowed_tools": ["websearch.query"]},
    )
    use_rag, reason = builder._should_use_rag(session, request)
    assert use_rag is False
    assert reason == "rag_not_in_allowed_tools"


def _retrieval_builder(
    *,
    tenant_id: str | None = "tenant-a",
    workspace_id: str | None = "workspace-a",
    bound_scope: VectorStoreScope | None = None,
) -> tuple[ContextBuilder, MagicMock]:
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = MagicMock(
        chunks=[],
        used=False,
        reason="no_hits",
        trace=MagicMock(),
    )
    vectorstore = MagicMock()
    vectorstore.bound_scope = bound_scope
    config = RuntimeConfig(
        llm_adapter=MagicMock(),
        enable_rag=True,
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        retrieval_service=retrieval_service,
        embedding_manager=MagicMock(),
    )
    return ContextBuilder(config, vectorstore), retrieval_service


def _request(*, tenant_id: str | None = "tenant-a", workspace_id: str | None = "workspace-a") -> RuntimeRequest:
    return RuntimeRequest(
        agent_id="agent",
        user_id="user-a",
        session_id="session-a",
        message="hello",
        tenant_id=tenant_id,
        workspace_id=workspace_id,
    )


@pytest.mark.gate
def test_context_builder_separates_operation_scope_from_metadata_filter() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(
        id="session-a",
        user_id="user-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    _, reason = builder._retrieve_for_session(session, _request())

    assert reason == "no_hits"
    request = retrieval_service.retrieve.call_args.args[0]
    assert request.scope == VectorStoreScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
    assert request.metadata_filter.conditions == {
        "session_id": "session-a",
        "user_id": "user-a",
    }


@pytest.mark.gate
def test_context_builder_rejects_conflicting_workspace_before_retrieval() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(
        id="session-a",
        user_id="user-a",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )

    chunks, reason = builder._retrieve_for_session(
        session,
        _request(workspace_id="workspace-b"),
    )

    assert chunks == []
    assert reason == "workspace_scope_conflict"
    retrieval_service.retrieve.assert_not_called()


@pytest.mark.gate
def test_context_builder_rejects_conflicting_tenant_before_retrieval() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(id="session-a", tenant_id="tenant-a")

    chunks, reason = builder._retrieve_for_session(
        session,
        _request(tenant_id="tenant-b", workspace_id=None),
    )

    assert chunks == []
    assert reason == "tenant_scope_conflict"
    retrieval_service.retrieve.assert_not_called()


@pytest.mark.gate
def test_context_builder_preserves_bound_namespace() -> None:
    builder, retrieval_service = _retrieval_builder(
        workspace_id=None,
        bound_scope=VectorStoreScope(
            tenant_id="tenant-a",
            namespace="configured",
            workspace_id="workspace-bound",
        ),
    )
    session = ChatSession(id="session-a", tenant_id="tenant-a")

    builder._retrieve_for_session(session, _request(workspace_id=None))

    request = retrieval_service.retrieve.call_args.args[0]
    assert request.scope == VectorStoreScope(
        tenant_id="tenant-a",
        namespace="configured",
        workspace_id="workspace-bound",
    )


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("request", 123),
        ("config", 123),
        ("session", 123),
    ],
)
def test_context_builder_rejects_non_string_tenant(
    source: str,
    value: object,
) -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(id="session-a", tenant_id="tenant-a")
    request = _request()

    if source == "request":
        request.tenant_id = value  # type: ignore[assignment]
    elif source == "config":
        builder._config.tenant_id = value  # type: ignore[assignment]
    else:
        session.tenant_id = value  # type: ignore[assignment]

    chunks, reason = builder._retrieve_for_session(session, request)

    assert chunks == []
    assert reason == "tenant_scope_invalid"
    retrieval_service.retrieve.assert_not_called()


def test_context_builder_rejects_blank_request_tenant_without_fallback() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(id="session-a", tenant_id="tenant-a")

    chunks, reason = builder._retrieve_for_session(
        session,
        _request(tenant_id="   "),
    )

    assert chunks == []
    assert reason == "tenant_scope_invalid"
    retrieval_service.retrieve.assert_not_called()


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("request", True),
        ("config", True),
        ("session", True),
    ],
)
def test_context_builder_rejects_non_string_workspace(
    source: str,
    value: object,
) -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(id="session-a", tenant_id="tenant-a")
    request = _request()

    if source == "request":
        request.workspace_id = value  # type: ignore[assignment]
    elif source == "config":
        builder._config.workspace_id = value  # type: ignore[assignment]
    else:
        session.workspace_id = value  # type: ignore[assignment]

    chunks, reason = builder._retrieve_for_session(session, request)

    assert chunks == []
    assert reason == "workspace_scope_invalid"
    retrieval_service.retrieve.assert_not_called()


def test_context_builder_canonicalizes_valid_routing_strings() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(
        id="session-a",
        tenant_id=" tenant-a ",
        workspace_id=" workspace-a ",
    )

    _, reason = builder._retrieve_for_session(
        session,
        _request(tenant_id=" tenant-a ", workspace_id=" workspace-a "),
    )

    assert reason == "no_hits"
    request = retrieval_service.retrieve.call_args.args[0]
    assert request.scope == VectorStoreScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )


def test_context_builder_rejects_invalid_bound_scope_before_retrieval() -> None:
    builder, retrieval_service = _retrieval_builder(
        bound_scope=object(),  # type: ignore[arg-type]
    )
    session = ChatSession(id="session-a", tenant_id="tenant-a")

    chunks, reason = builder._retrieve_for_session(session, _request())

    assert chunks == []
    assert reason == "tenant_scope_invalid"
    retrieval_service.retrieve.assert_not_called()


def test_context_builder_does_not_use_routing_keys_from_request_metadata() -> None:
    builder, retrieval_service = _retrieval_builder()
    session = ChatSession(id="session-a", tenant_id="tenant-a")
    request = _request()
    request.metadata = {
        "tenant_id": "spoofed-tenant",
        "namespace": "spoofed-namespace",
        "workspace_id": "spoofed-workspace",
    }

    _, reason = builder._retrieve_for_session(session, request)

    assert reason == "no_hits"
    retrieval_request = retrieval_service.retrieve.call_args.args[0]
    assert retrieval_request.scope == VectorStoreScope(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
    )
