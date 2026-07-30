# © Artur Czarnecki. All rights reserved.

"""Unit and orchestration tests for LKW model runtime portability proof."""

from __future__ import annotations

import json

import pytest

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
    ProofFailureCode,
    ProofOverallStatus,
    StageStatus,
)
from local_workspace_application.model_runtime_proof.index_identity import (
    compare_embedding_identity,
    compare_index_identity,
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

    adapter = type("A", (), {"provider": LLMProvider.OLLAMA})()
    choice, mode = _resolve_tool_choice(adapter, force_tool_choice=True)
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
        collection_identity="w",
        vector_count=3,
        embedding=EmbeddingIdentityRecord(provider="ollama", model="e", dimensions=1),
    )
    after = before.model_copy()
    collection_ok, vector_ok, document_ok = compare_index_identity(before, after)
    assert collection_ok and vector_ok and document_ok


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
    assert payload["schema_version"] == "lkw.model_runtime_portability.proof.v1"


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
    assert "lkw.model_runtime_portability.proof.v1" in text


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
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    name="local.workspace.search",
                    arguments_json=json.dumps(
                        {"workspace_id": "ws-proof", "query": FIXTURE_MARKER}
                    ),
                )
            ],
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
        )

    monkeypatch.setattr(ModelRuntimeProofRunner, "_qualify_provider", _fake_qualify)
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
