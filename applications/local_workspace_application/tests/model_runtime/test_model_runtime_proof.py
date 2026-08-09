# © Artur Czarnecki. All rights reserved.

"""Unit and orchestration tests for LKW model runtime portability proof."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_workspace_application.model_runtime_proof.aggregation import (
    index_invariance_passes,
    provider_qualification_passes,
)
from local_workspace_application.model_runtime_proof.config import (
    ModelRuntimeProofConfig,
    classify_endpoint,
    load_proof_config_from_env,
    materialize_provider_env,
)
from local_workspace_application.model_runtime_proof.contracts import (
    BASIC_GENERATION_MARKER,
    EmbeddingIdentityRecord,
    FIXTURE_MARKER,
    IndexIdentityRecord,
    IndexInvarianceResult,
    ProofFailureCode,
    ProofOverallStatus,
    ProviderQualificationResult,
    StageStatus,
)
from local_workspace_application.model_runtime_proof.index_identity import (
    compare_embedding_identity,
    compare_index_identity,
    index_identity_is_complete,
    resolve_collection_identity,
)
from local_workspace_application.model_runtime_proof.report import (
    render_markdown,
    render_terminal_summary,
    serialize_result_json,
)
from local_workspace_application.model_runtime_proof.runner import (
    ModelRuntimeProofRunner,
)
from local_workspace_application.model_runtime_proof.safety import (
    assert_no_secret_leak,
    redact_text,
    safe_error_excerpt,
)
from local_workspace_application.model_runtime_proof.stages import (
    run_grounded_ask,
    validate_tool_call,
    _resolve_tool_choice,
)

pytestmark = pytest.mark.unit


def test_proof_config_validation_rejects_empty_model() -> None:
    config = ModelRuntimeProofConfig(
        ollama_model="",
        vllm_model="m",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant",
        data_home="/tmp/proof",
        timeout_seconds=30.0,
    )
    assert ProofFailureCode.CONFIG_INVALID in config.validate()


def test_proof_config_allows_empty_data_home_for_runner_default() -> None:
    config = ModelRuntimeProofConfig(
        ollama_model="llama3.1:8b",
        vllm_model="Qwen/Qwen2.5-3B-Instruct",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant",
        data_home="",
        timeout_seconds=30.0,
    )
    assert config.validate() == []


def test_materialize_provider_env_sets_canonical_names() -> None:
    config = ModelRuntimeProofConfig(
        ollama_model="llama3.1:latest",
        vllm_model="Qwen/Qwen2.5-7B-Instruct",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant",
        data_home="/tmp/proof",
        timeout_seconds=30.0,
    )
    ollama_env = materialize_provider_env(provider="ollama", config=config, target={})
    assert ollama_env["INTERGRAX_LLM_PROVIDER"] == "ollama"
    assert ollama_env["INTERGRAX_DEFAULT_OLLAMA_MODEL"] == "llama3.1:latest"
    vllm_env = materialize_provider_env(
        provider="vllm",
        config=config,
        target={"OLLAMA_HOST": "http://127.0.0.1:11434"},
    )
    assert vllm_env["INTERGRAX_DEFAULT_VLLM_MODEL"] == "Qwen/Qwen2.5-7B-Instruct"
    assert "OLLAMA_HOST" not in vllm_env


def test_secret_redaction() -> None:
    text = "authorization: Bearer sk-abcdefghijklmnopqrstuvwxyz"
    redacted = redact_text(text)
    assert "sk-" not in redacted
    assert "[REDACTED]" in redacted


def test_safe_error_excerpt_bounded() -> None:
    excerpt = safe_error_excerpt(RuntimeError("x" * 500), limit=50)
    assert len(excerpt) <= 50


def test_assert_no_secret_leak_detects_token() -> None:
    assert (
        assert_no_secret_leak({"token": "Bearer abc"}) == "proof_secret_leak_detected"
    )


def test_validate_tool_call_accepts_llm_tool_call_arguments_json() -> None:
    from intergrax.llm_adapters.contracts.tool_call import LLMToolCall

    calls = [
        LLMToolCall(
            id="call-1",
            name="local.workspace.search",
            arguments_json='{"workspace_id": "ws-1", "query": "marker"}',
        )
    ]
    args, code, _ = validate_tool_call(calls, workspace_id="ws-1")
    assert code is None
    assert args == {"workspace_id": "ws-1", "query": "marker"}


def test_resolve_tool_choice_uses_required_for_ollama() -> None:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

    class _OllamaStub:
        provider = LLMProvider.OLLAMA

    choice, mode = _resolve_tool_choice(_OllamaStub(), force_tool_choice=True)  # type: ignore[arg-type]
    assert choice == "required"
    assert mode == "forced"


def test_validate_tool_call_rejects_multiple() -> None:
    calls = [
        type("C", (), {"name": "local.workspace.search", "arguments": {}})(),
        type("C", (), {"name": "local.workspace.search", "arguments": {}})(),
    ]
    _, code, _ = validate_tool_call(calls, workspace_id="ws-1")
    assert code is ProofFailureCode.TOOL_CALL_MULTIPLE


def test_validate_tool_call_rejects_unexpected_tool() -> None:
    calls = [type("C", (), {"name": "shell.run", "arguments": {"cmd": "ls"}})()]
    _, code, _ = validate_tool_call(calls, workspace_id="ws-1")
    assert code is ProofFailureCode.TOOL_CALL_UNEXPECTED_TOOL


def test_validate_tool_call_rejects_invalid_workspace() -> None:
    calls = [
        type(
            "C",
            (),
            {
                "name": "local.workspace.search",
                "arguments": {"workspace_id": "other", "query": "marker"},
            },
        )()
    ]
    _, code, _ = validate_tool_call(calls, workspace_id="ws-1")
    assert code is ProofFailureCode.TOOL_CALL_INVALID


def test_embedding_identity_comparison() -> None:
    before = EmbeddingIdentityRecord(provider="ollama", model="embed", dimensions=768)
    after = EmbeddingIdentityRecord(provider="ollama", model="embed", dimensions=768)
    assert compare_embedding_identity(before, after) is True
    changed = EmbeddingIdentityRecord(provider="vllm", model="embed", dimensions=768)
    assert compare_embedding_identity(before, changed) is False


def test_index_identity_comparison() -> None:
    before = IndexIdentityRecord(
        tenant_id="t",
        workspace_id="w",
        source_id="source-1",
        document_id="doc-1",
        content_hash="hash-1",
        collection_identity="inmemory:tenant",
        vector_count=3,
        chunk_count=3,
        embedding=EmbeddingIdentityRecord(provider="ollama", model="e", dimensions=1),
    )
    after = before.model_copy()
    comparison = compare_index_identity(before, after)
    assert (
        comparison.collection_identity
        and comparison.vector_count
        and comparison.document_id
        and comparison.source_id
        and comparison.content_hash
        and comparison.chunk_count
    )


def test_result_schema_serializes() -> None:
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
    )

    result = ModelRuntimeProofResult(
        proof_id="proof-1",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.FAIL,
    )
    payload = json.loads(serialize_result_json(result))
    assert payload["schema_version"] == "lkw.model_runtime_portability.proof.v2"


def test_markdown_render_includes_marker_constants() -> None:
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
    )

    result = ModelRuntimeProofResult(
        proof_id="proof-1",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.FAIL,
    )
    text = render_markdown(result)
    assert "lkw.model_runtime_portability.proof.v2" in text


def test_terminal_summary_shows_overall() -> None:
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
        ProviderQualificationResult,
    )

    result = ModelRuntimeProofResult(
        proof_id="proof-1",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.PASS,
        provider_results={
            "ollama": ProviderQualificationResult(
                provider="ollama",
                configured_model="m",
                health_status=StageStatus.PASS,
                basic_generation_status=StageStatus.PASS,
                structured_planning_status=StageStatus.PASS,
                tool_call_status=StageStatus.PASS,
                tool_execution_status=StageStatus.PASS,
                grounded_ask_status=StageStatus.PASS,
                citation_status=StageStatus.PASS,
                ask_run_persisted=True,
                resolved_through_canonical_resolver=True,
            )
        },
    )
    summary = render_terminal_summary(result)
    assert "OVERALL:" in summary
    assert "PASS" in summary


def test_classify_endpoint_loopback() -> None:
    assert classify_endpoint("http://127.0.0.1:8100/v1") == "loopback"


class _FakeAdapter:
    provider = "fake"
    model = "fake"

    def supports_structured_output(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def generate_messages(self, messages, **kwargs):
        from intergrax.llm_adapters._shared.adapter_response_builders import (
            build_adapter_response,
        )

        return build_adapter_response(content=f"marker {BASIC_GENERATION_MARKER}")

    def generate_with_tools(self, messages, tools_schema, **kwargs):
        from intergrax.llm_adapters._shared.adapter_response_builders import (
            build_adapter_response,
        )
        from intergrax.llm_adapters.contracts.tool_call import LLMToolCall

        return build_adapter_response(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="local.workspace.search",
                    arguments_json=json.dumps(
                        {"workspace_id": "ws-proof", "query": FIXTURE_MARKER}
                    ),
                ),
            ),
        )


def test_orchestration_with_fake_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    config = ModelRuntimeProofConfig(
        ollama_model="fake",
        vllm_model="fake",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant-proof",
        data_home=str(tmp_path / "proof-data"),
        timeout_seconds=60.0,
        vector_store="inmemory",
        require_live_providers=False,
        vllm_provisioning_classification="committed_compose_sufficient",
    )

    async def _fake_qualify(self, *, provider, fixture, index_identity):
        from local_workspace_application.model_runtime_proof.contracts import (
            ProviderQualificationResult,
        )

        return ProviderQualificationResult(
            provider=provider,
            configured_model="fake",
            resolved_model="fake",
            server_model="fake",
            adapter_class="_FakeAdapter",
            health_status=StageStatus.PASS,
            basic_generation_status=StageStatus.PASS,
            structured_planning_status=StageStatus.PASS,
            tool_call_status=StageStatus.PASS,
            tool_execution_status=StageStatus.PASS,
            grounded_ask_status=StageStatus.PASS,
            citation_status=StageStatus.PASS,
            ask_run_persisted=True,
            resolved_through_canonical_resolver=True,
            session_adapter_object_id=f"{provider}-adapter",
        )

    monkeypatch.setattr(ModelRuntimeProofRunner, "_qualify_provider", _fake_qualify)
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.runner.index_identity_is_complete",
        lambda identity: True,
    )
    import asyncio

    runner = ModelRuntimeProofRunner(config)
    result = asyncio.run(runner.run())
    assert result.fixture.workspace_id is not None
    assert result.index_invariance.no_reindex is StageStatus.PASS


def test_load_config_reads_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_LKW_MODEL_RUNTIME_PROOF", "1")
    monkeypatch.setenv("LKW_MODEL_RUNTIME_PROOF_OLLAMA_MODEL", "llama3.1:latest")
    config = load_proof_config_from_env()
    assert config.require_live_providers is True


def test_provider_qualification_fails_when_stage_not_run() -> None:
    result = ProviderQualificationResult(
        provider="ollama",
        configured_model="m",
        health_status=StageStatus.PASS,
        basic_generation_status=StageStatus.NOT_RUN,
        resolved_through_canonical_resolver=True,
        ask_run_persisted=True,
    )
    assert provider_qualification_passes(result) is False


def test_provider_qualification_fails_when_ask_not_persisted() -> None:
    result = ProviderQualificationResult(
        provider="ollama",
        configured_model="m",
        health_status=StageStatus.PASS,
        basic_generation_status=StageStatus.PASS,
        structured_planning_status=StageStatus.PASS,
        tool_call_status=StageStatus.PASS,
        tool_execution_status=StageStatus.PASS,
        grounded_ask_status=StageStatus.PASS,
        citation_status=StageStatus.PASS,
        resolved_through_canonical_resolver=True,
        ask_run_persisted=False,
    )
    assert provider_qualification_passes(result) is False


def test_index_invariance_fails_on_document_change() -> None:
    inv = IndexInvarianceResult(
        embedding_identity=StageStatus.PASS,
        collection_identity=StageStatus.PASS,
        vector_count=StageStatus.PASS,
        source_identity=StageStatus.PASS,
        document_identity=StageStatus.FAIL,
        content_hash=StageStatus.PASS,
        chunk_count=StageStatus.PASS,
        no_reindex=StageStatus.PASS,
    )
    assert index_invariance_passes(inv) is False


def test_index_invariance_fails_on_unknown_collection() -> None:
    before = IndexIdentityRecord(
        tenant_id="t",
        workspace_id="w",
        source_id="s",
        document_id="d",
        content_hash="h",
        collection_identity="inmemory:tenant",
        vector_count=1,
        chunk_count=1,
        embedding=EmbeddingIdentityRecord(provider="ollama", model="e", dimensions=1),
    )
    after = before.model_copy(update={"collection_identity": "unknown"})
    comparison = compare_index_identity(before, after)
    assert comparison.collection_identity is False


def test_index_identity_incomplete_rejects_unknown_collection() -> None:
    identity = IndexIdentityRecord(
        tenant_id="t",
        workspace_id="w",
        source_id="s",
        document_id="d",
        content_hash="h",
        collection_identity="unknown",
        vector_count=1,
        chunk_count=1,
        embedding=EmbeddingIdentityRecord(provider="ollama", model="e", dimensions=1),
    )
    assert index_identity_is_complete(identity) is False


def test_resolve_collection_identity_from_manager() -> None:
    class _Store:
        def list_collections(self):
            return ["inmemory:tenant-proof"]

    class _Manager:
        _collection_name = None
        _store = _Store()

        def list_collections(self):
            return self._store.list_collections()

    class _Context:
        vectorstore_manager = _Manager()
        integration_profile = None
        extras: dict[str, object] = {}

    identity = resolve_collection_identity(_Context(), "tenant-proof")  # type: ignore[arg-type]
    assert identity == "inmemory:tenant-proof"


@pytest.mark.asyncio
async def test_grounded_ask_fails_when_run_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from intergrax.integrations._shared.in_memory_document_store import (
        InMemoryDocumentStore,
    )

    from local_workspace_application.model_runtime_proof.contracts import ASK_QUESTION
    from local_workspace_application.workspaces.repository import (
        ManagedWorkspaceRepository,
    )

    app = FastAPI()

    @app.post("/v1/local_workspace/workspaces/{workspace_id}/ask")
    def _ask(workspace_id: str):
        return {
            "run_id": "run-missing",
            "workspace_id": workspace_id,
            "status": "completed",
            "question": ASK_QUESTION,
            "answer": f"The code is {FIXTURE_MARKER}",
            "citations": [
                {
                    "source_id": "source-1",
                    "excerpt": FIXTURE_MARKER,
                }
            ],
            "created_at": "2026-07-30T00:00:00Z",
        }

    store = InMemoryDocumentStore()
    repository = ManagedWorkspaceRepository(store)
    client = TestClient(app)
    ok, _, _, persisted, failure, detail, status = await run_grounded_ask(
        client,
        tenant_id="tenant",
        workspace_id="ws-1",
        source_id="source-1",
        repository=repository,
    )
    assert ok is False
    assert persisted is False
    assert failure is ProofFailureCode.GROUNDED_ASK_FAILED
    assert detail == "ask_not_persisted"
    assert status == 200


def test_runner_uses_canonical_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    calls: list[str] = []

    def _fake_resolve(env):
        calls.append("resolve")
        return _FakeAdapter()

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.runtime.resolve_llm_adapter",
        _fake_resolve,
    )
    from local_workspace_application.model_runtime_proof.config import (
        ModelRuntimeProofConfig,
    )
    from local_workspace_application.model_runtime_proof.runtime import (
        build_proof_runtime_session,
    )

    config = ModelRuntimeProofConfig(
        ollama_model="fake",
        vllm_model="fake",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant-proof",
        data_home=str(tmp_path / "proof-data"),
        timeout_seconds=60.0,
        vector_store="inmemory",
        require_live_providers=False,
    )
    session = build_proof_runtime_session(config, provider="ollama")
    try:
        assert calls == ["resolve"]
        assert session.llm_adapter is not None
        assert session.ask_service.llm_adapter is session.llm_adapter
    finally:
        session.close()


def test_global_document_store_resolver_unchanged_after_session_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_workspace_application.serving import workspace_routes
    from local_workspace_application.model_runtime_proof.config import (
        ModelRuntimeProofConfig,
    )
    from local_workspace_application.model_runtime_proof.runtime import (
        build_proof_runtime_session,
    )

    original = workspace_routes.resolve_managed_workspace_document_store
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    config = ModelRuntimeProofConfig(
        ollama_model="fake",
        vllm_model="fake",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant-proof",
        data_home=str(tmp_path / "proof-data"),
        timeout_seconds=60.0,
        vector_store="inmemory",
        require_live_providers=False,
    )
    session = build_proof_runtime_session(config, provider=None)
    try:
        assert workspace_routes.resolve_managed_workspace_document_store is original
    finally:
        session.close()
    assert workspace_routes.resolve_managed_workspace_document_store is original


def test_repository_state_clean_when_porcelain_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: "abc123",
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(available=True, lines=()),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "clean"


def test_repository_state_task_owned_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: "abc123",
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(
            available=True,
            lines=(
                " M applications/local_workspace_application/model_runtime_proof/runner.py",
            ),
        ),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "task_owned_changes"


def test_repository_state_unrelated_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: "abc123",
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(
            available=True,
            lines=(" M README.md",),
        ),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "unrelated_changes"


def test_repository_state_mixed_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: "abc123",
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(
            available=True,
            lines=(
                " M applications/local_workspace_application/model_runtime_proof/runner.py",
                " M README.md",
            ),
        ),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "task_owned_and_unrelated_changes"


def test_repository_state_unavailable_on_porcelain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: "abc123",
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(available=False, lines=()),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "unavailable"


def test_repository_state_unavailable_on_rev_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.repository_state import (
        GitPorcelainResult,
        capture_repository_state,
    )

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_head",
        lambda repo_root=None: None,
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.repository_state._git_porcelain",
        lambda repo_root=None: GitPorcelainResult(available=True, lines=()),
    )
    state = capture_repository_state()
    assert state.working_tree_classification == "unavailable"


def test_vllm_provisioning_defaults_to_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION", raising=False
    )
    from local_workspace_application.model_runtime_proof.config import (
        load_vllm_provisioning_classification_from_env,
    )

    assert load_vllm_provisioning_classification_from_env() == "unverified"


def test_vllm_provisioning_explicit_committed_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION",
        "committed_compose_sufficient",
    )
    config = load_proof_config_from_env()
    assert config.vllm_provisioning_classification == "committed_compose_sufficient"


def test_vllm_provisioning_explicit_external_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION",
        "external_runtime",
    )
    config = load_proof_config_from_env()
    assert config.vllm_provisioning_classification == "external_runtime"


def test_vllm_provisioning_unsupported_value_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "LKW_MODEL_RUNTIME_PROOF_VLLM_PROVISIONING_CLASSIFICATION",
        "made_up",
    )
    config = load_proof_config_from_env()
    assert ProofFailureCode.CONFIG_INVALID in config.validate()


def test_unverified_vllm_provisioning_blocks_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")

    async def _fake_qualify(self, *, provider, fixture, index_identity):
        return ProviderQualificationResult(
            provider=provider,
            configured_model="fake",
            resolved_model="fake",
            server_model="fake",
            adapter_class="_FakeAdapter",
            health_status=StageStatus.PASS,
            basic_generation_status=StageStatus.PASS,
            structured_planning_status=StageStatus.PASS,
            tool_call_status=StageStatus.PASS,
            tool_execution_status=StageStatus.PASS,
            grounded_ask_status=StageStatus.PASS,
            citation_status=StageStatus.PASS,
            ask_run_persisted=True,
            resolved_through_canonical_resolver=True,
            session_adapter_object_id=f"{provider}-adapter",
        )

    monkeypatch.setattr(ModelRuntimeProofRunner, "_qualify_provider", _fake_qualify)
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.runner.index_identity_is_complete",
        lambda identity: True,
    )
    config = ModelRuntimeProofConfig(
        ollama_model="fake",
        vllm_model="fake",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant-proof",
        data_home=str(tmp_path / "proof-data"),
        timeout_seconds=60.0,
        vector_store="inmemory",
        require_live_providers=False,
        vllm_provisioning_classification="unverified",
    )
    import asyncio

    result = asyncio.run(ModelRuntimeProofRunner(config).run())
    assert result.overall_status is ProofOverallStatus.FAIL


def test_stale_v1_evidence_removed_after_failed_write(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
    )
    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "lkw.model_runtime_portability.proof.v1",
                "overall_status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    markdown_path.write_text("# stale", encoding="utf-8")
    result = ModelRuntimeProofResult(
        proof_id="proof-fail",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.FAIL,
    )
    write_evidence(result, json_path=json_path, markdown_path=markdown_path)
    assert not json_path.is_file()
    assert not markdown_path.is_file()


def _pass_evidence_result(**updates):
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
        ProviderQualificationResult,
    )

    digest = "sha256:" + ("a" * 64)
    result = ModelRuntimeProofResult(
        proof_id="proof-pass",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.PASS,
        provider_results={
            "ollama": ProviderQualificationResult(
                provider="ollama",
                configured_model="qwen2.5:14b",
                resolved_model="qwen2.5:14b",
                server_model="qwen2.5:14b",
                server_model_digest=digest,
            ),
            "vllm": ProviderQualificationResult(
                provider="vllm",
                configured_model="Qwen/Qwen2.5-3B-Instruct",
                resolved_model="Qwen/Qwen2.5-3B-Instruct",
                server_model="Qwen/Qwen2.5-3B-Instruct",
            ),
        },
        vllm_provisioning_classification="committed_compose_sufficient",
    )
    if updates:
        return result.model_copy(update=updates)
    return result


def _assert_no_evidence_residue(directory: Path) -> None:
    for path in directory.iterdir():
        name = path.name
        assert not name.endswith(".tmp")
        assert ".rollback." not in name


def test_evidence_publication_fails_before_canonical_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    original_json = json.dumps(
        {
            "schema_version": "lkw.model_runtime_portability.proof.v1",
            "overall_status": "PASS",
        }
    )
    original_markdown = "# stale"
    json_path.write_text(original_json, encoding="utf-8")
    markdown_path.write_text(original_markdown, encoding="utf-8")

    def _fail_temp_write(*args, **kwargs):
        raise OSError("temp write failed")

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.report._write_validated_temp_file",
        _fail_temp_write,
    )
    with pytest.raises(OSError):
        write_evidence(
            _pass_evidence_result(),
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert json_path.read_text(encoding="utf-8") == original_json
    assert markdown_path.read_text(encoding="utf-8") == original_markdown
    _assert_no_evidence_residue(tmp_path)


def test_evidence_publication_rolls_back_after_second_replacement_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    original_json = json.dumps(
        {
            "schema_version": "lkw.model_runtime_portability.proof.v1",
            "overall_status": "PASS",
        }
    )
    original_markdown = "# stale"
    json_path.write_text(original_json, encoding="utf-8")
    markdown_path.write_text(original_markdown, encoding="utf-8")

    original_replace = os.replace
    calls = {"count": 0}

    def _replace_with_second_failure(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("markdown replacement failed")
        return original_replace(src, dst)

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.report.os.replace",
        _replace_with_second_failure,
    )
    with pytest.raises(OSError):
        write_evidence(
            _pass_evidence_result(),
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert json_path.read_text(encoding="utf-8") == original_json
    assert markdown_path.read_text(encoding="utf-8") == original_markdown
    _assert_no_evidence_residue(tmp_path)


def test_evidence_publication_rolls_back_when_no_original_files_and_second_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    original_replace = os.replace
    calls = {"count": 0}

    def _replace_with_second_failure(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("markdown replacement failed")
        return original_replace(src, dst)

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.report.os.replace",
        _replace_with_second_failure,
    )
    with pytest.raises(OSError):
        write_evidence(
            _pass_evidence_result(),
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert not json_path.is_file()
    assert not markdown_path.is_file()
    _assert_no_evidence_residue(tmp_path)


def test_evidence_publication_rolls_back_when_only_json_existed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    original_json = json.dumps({"schema_version": "v1"})
    json_path.write_text(original_json, encoding="utf-8")
    original_replace = os.replace
    calls = {"count": 0}

    def _replace_with_second_failure(src, dst):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("markdown replacement failed")
        return original_replace(src, dst)

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.report.os.replace",
        _replace_with_second_failure,
    )
    with pytest.raises(OSError):
        write_evidence(
            _pass_evidence_result(),
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert json_path.read_text(encoding="utf-8") == original_json
    assert not markdown_path.is_file()
    _assert_no_evidence_residue(tmp_path)


def test_atomic_evidence_replacement_on_pass(tmp_path: Path) -> None:
    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "lkw.model_runtime_portability.proof.v1",
                "overall_status": "PASS",
            }
        ),
        encoding="utf-8",
    )
    markdown_path.write_text("# stale", encoding="utf-8")
    result = _pass_evidence_result()
    write_evidence(result, json_path=json_path, markdown_path=markdown_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert payload["schema_version"] == "lkw.model_runtime_portability.proof.v2"
    assert payload["overall_status"] == "PASS"
    assert payload["proof_id"] == result.proof_id
    assert "lkw.model_runtime_portability.proof.v2" in markdown_text
    assert f"proof_id: `{result.proof_id}`" in markdown_text
    assert payload["provider_results"]["ollama"]["configured_model"] == "qwen2.5:14b"
    assert (
        payload["provider_results"]["vllm"]["configured_model"]
        == "Qwen/Qwen2.5-3B-Instruct"
    )
    digest = payload["provider_results"]["ollama"]["server_model_digest"]
    assert digest
    assert f"server_model_digest: `{digest}`" in markdown_text
    _assert_no_evidence_residue(tmp_path)


def test_evidence_publication_rolls_back_on_post_publication_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_workspace_application.model_runtime_proof.report import write_evidence

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    original_json = json.dumps(
        {
            "schema_version": "lkw.model_runtime_portability.proof.v1",
            "overall_status": "PASS",
        }
    )
    original_markdown = "# stale"
    json_path.write_text(original_json, encoding="utf-8")
    markdown_path.write_text(original_markdown, encoding="utf-8")
    calls = {"count": 0}
    original_validate = "local_workspace_application.model_runtime_proof.report._validate_evidence_payload"

    def _validate_with_post_failure(json_text, markdown_text):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise ValueError("post_publication_validation_failed")
        import local_workspace_application.model_runtime_proof.report as report

        return report._validate_evidence_payload(json_text, markdown_text)

    monkeypatch.setattr(original_validate, _validate_with_post_failure)
    with pytest.raises(ValueError):
        write_evidence(
            _pass_evidence_result(),
            json_path=json_path,
            markdown_path=markdown_path,
        )
    assert json_path.read_text(encoding="utf-8") == original_json
    assert markdown_path.read_text(encoding="utf-8") == original_markdown
    _assert_no_evidence_residue(tmp_path)


def test_invalidation_rolls_back_when_second_unlink_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_workspace_application.model_runtime_proof.report import (
        invalidate_canonical_evidence,
    )

    json_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.json"
    markdown_path = tmp_path / "LKW_MODEL_RUNTIME_PORTABILITY.md"
    json_path.write_text('{"schema":"v1"}', encoding="utf-8")
    markdown_path.write_text("# stale", encoding="utf-8")
    original_unlink = Path.unlink
    calls = {"count": 0}

    def _unlink_with_second_failure(self, missing_ok=False):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("markdown unlink failed")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", _unlink_with_second_failure)
    with pytest.raises(OSError):
        invalidate_canonical_evidence(json_path=json_path, markdown_path=markdown_path)
    assert json_path.is_file()
    assert markdown_path.is_file()


def test_provider_qualification_fails_when_ollama_digest_missing() -> None:
    result = ProviderQualificationResult(
        provider="ollama",
        configured_model="qwen2.5:14b",
        health_status=StageStatus.PASS,
        basic_generation_status=StageStatus.PASS,
        structured_planning_status=StageStatus.PASS,
        tool_call_status=StageStatus.PASS,
        tool_execution_status=StageStatus.PASS,
        grounded_ask_status=StageStatus.PASS,
        citation_status=StageStatus.PASS,
        resolved_through_canonical_resolver=True,
        ask_run_persisted=True,
    )
    assert provider_qualification_passes(result) is False


def test_normalize_model_digest_accepts_prefixed_and_bare_hex() -> None:
    from local_workspace_application.model_runtime_proof.health import (
        normalize_model_digest,
    )

    bare = "a" * 64
    assert normalize_model_digest(bare) == f"sha256:{bare}"
    assert normalize_model_digest(f"sha256:{bare}") == f"sha256:{bare}"
    assert normalize_model_digest("sha256:short") is None
    assert normalize_model_digest("") is None


def test_ollama_health_requires_exact_model_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.config import (
        ModelRuntimeProofConfig,
    )
    from local_workspace_application.model_runtime_proof.health import (
        probe_ollama_health,
    )

    config = ModelRuntimeProofConfig(
        ollama_model="qwen2.5:14b",
        vllm_model="Qwen/Qwen2.5-3B-Instruct",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant",
        data_home="/tmp/proof",
        timeout_seconds=30.0,
    )

    def _fake_http_json(url: str, *, timeout: float):
        if url.endswith("/api/version"):
            return {"version": "0.32.5"}
        return {
            "models": [
                {
                    "name": "qwen2.5:14b",
                    "digest": "sha256:" + ("b" * 64),
                    "size": 9000,
                }
            ]
        }

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health._http_json",
        _fake_http_json,
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health.materialize_provider_env",
        lambda **kwargs: {
            "INTERGRAX_LLM_MODEL": "qwen2.5:14b",
            "OLLAMA_HOST": "http://127.0.0.1:11434",
        },
    )

    class _Adapter:
        model = "qwen2.5:14b"

        def __class_getitem__(cls, item):
            return cls

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health.unwrap_catalog_capability_adapter",
        lambda adapter: _Adapter(),
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health.LLMAdapterRegistry.create",
        lambda *args, **kwargs: _Adapter(),
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health.NativeOllamaAdapter",
        _Adapter,
    )
    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health.OllamaModelCapabilityResolver",
        lambda **kwargs: type(
            "Resolver",
            (),
            {
                "resolve": lambda self, model: type(
                    "Caps", (), {"resolved": True, "capabilities": ()}
                )()
            },
        )(),
    )

    snapshot, failure, detail = probe_ollama_health(config)
    assert failure is None
    assert snapshot is not None
    assert snapshot.server_model_digest == "sha256:" + ("b" * 64)


def test_ollama_health_fails_when_digest_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_workspace_application.model_runtime_proof.config import (
        ModelRuntimeProofConfig,
    )
    from local_workspace_application.model_runtime_proof.health import (
        probe_ollama_health,
    )

    config = ModelRuntimeProofConfig(
        ollama_model="qwen2.5:14b",
        vllm_model="Qwen/Qwen2.5-3B-Instruct",
        ollama_base_url="http://127.0.0.1:11434",
        vllm_base_url="http://127.0.0.1:8100/v1",
        tenant_id="tenant",
        data_home="/tmp/proof",
        timeout_seconds=30.0,
    )

    def _fake_http_json(url: str, *, timeout: float):
        if url.endswith("/api/version"):
            return {"version": "0.32.5"}
        return {"models": [{"name": "qwen2.5:14b", "size": 9000}]}

    monkeypatch.setattr(
        "local_workspace_application.model_runtime_proof.health._http_json",
        _fake_http_json,
    )

    snapshot, failure, detail = probe_ollama_health(config)
    assert snapshot is None
    assert failure is ProofFailureCode.PROVIDER_IDENTITY_MISMATCH
    assert detail == "model_digest_missing"


def test_qualified_ollama_model_stored_in_v2_result() -> None:
    from datetime import UTC, datetime

    from local_workspace_application.model_runtime_proof.contracts import (
        ModelRuntimeProofResult,
        ProviderQualificationResult,
    )

    result = ModelRuntimeProofResult(
        proof_id="proof-pass",
        started_at=datetime.now(UTC),
        overall_status=ProofOverallStatus.PASS,
        provider_results={
            "ollama": ProviderQualificationResult(
                provider="ollama",
                configured_model="llama3.1:8b",
                resolved_model="llama3.1:8b",
            )
        },
    )
    payload = json.loads(serialize_result_json(result))
    assert payload["provider_results"]["ollama"]["configured_model"] == "llama3.1:8b"
    assert payload["provider_results"]["ollama"]["resolved_model"] == "llama3.1:8b"


def test_failed_candidate_not_reported_as_qualified() -> None:
    result = ProviderQualificationResult(
        provider="ollama",
        configured_model="qwen2.5:7b",
        resolved_model="qwen2.5:7b",
        health_status=StageStatus.PASS,
        basic_generation_status=StageStatus.PASS,
        structured_planning_status=StageStatus.PASS,
        tool_call_status=StageStatus.PASS,
        tool_execution_status=StageStatus.PASS,
        grounded_ask_status=StageStatus.FAIL,
        citation_status=StageStatus.FAIL,
        resolved_through_canonical_resolver=True,
        ask_run_persisted=False,
    )
    assert provider_qualification_passes(result) is False
