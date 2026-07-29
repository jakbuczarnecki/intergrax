# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall

from local_workspace_application.benchmarks.local_model_qualification.config import load_config
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    FailurePhase,
    ModelProvisioningStatus,
    ModelStatus,
    ProtocolStatus,
    ProvisionedModel,
    ProvisioningResult,
    SafeErrorCode,
    SchemaProbeStatus,
    StructuralFailureCategory,
    WarmupStatus,
)
from local_workspace_application.benchmarks.local_model_qualification.protocols import (
    SUBMIT_DRAFT_TOOL_NAME,
    classify_model_preparation_error,
)
from local_workspace_application.benchmarks.local_model_qualification.runner import (
    compute_exit_code,
    run_benchmark,
    run_protocol_benchmark,
)
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    DraftWebUrlSource,
    KnowledgeAddSourcesDraftAction,
    NameDraftWorkspaceReference,
    WorkspaceListDraftAction,
)
from local_workspace_application.conversation.interaction_models import WorkspaceReferenceKind

_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "local-model-qualification.toml"
)


def _provisioning(config) -> ProvisioningResult:
    required = tuple(model.name for model in config.models if model.enabled)
    return ProvisioningResult(
        runtime="docker",
        compose_file=config.ollama.compose_file,
        compose_service=config.ollama.compose_service,
        container_name=config.ollama.container_name,
        persistent_model_volume="intergrax-ollama-models",
        readiness_result="READY",
        required_models=required,
        models=tuple(
            ProvisionedModel(model=name, status=ModelProvisioningStatus.ALREADY_AVAILABLE)
            for name in required
        ),
    )


@dataclass
class ScriptedAdapter:
    calls: list[str]
    ps_observer: list[str] = field(default_factory=list)

    def supports_structured_output(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: float | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        self.calls.append(run_id or "")
        draft = ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
        return LLMStructuredResult(parsed=draft, response=LLMAdapterResponse(content=""))

    def generate_with_tools(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs.get("run_id", ""))
        draft = ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
        return LLMAdapterResponse(
            content="",
            finish_reason=LLMFinishReason.TOOL_CALLS,
            tool_calls=(
                LLMToolCall(
                    id="1",
                    name=SUBMIT_DRAFT_TOOL_NAME,
                    arguments_json=draft.model_dump_json(),
                ),
            ),
        )


class FakeOllamaClient:
    def __init__(self, installed: set[str]) -> None:
        self._installed = installed

    def version(self) -> dict[str, str]:
        return {"version": "0.5.0"}

    def list(self) -> dict[str, list[dict[str, str | int]]]:
        return {
            "models": [
                {"name": name, "digest": f"sha256:{name}", "size": 1000}
                for name in self._installed
            ]
        }

    def show(self, *, model: str) -> dict[str, object]:
        return {
            "details": {"parameter_size": "14B", "quantization_level": "Q4", "family": "qwen2"},
        }

    def ps(self) -> dict[str, list[dict[str, int | str]]]:
        return {"models": []}


def _client_factory(installed: set[str]):
    def factory(host: str) -> FakeOllamaClient:
        return FakeOllamaClient(installed)

    return factory


class CapabilityFailAdapter(ScriptedAdapter):
    @property
    def model_capabilities(self) -> object:
        raise RuntimeError("capability resolution failed")


@dataclass
class CountingWarmupAdapter:
    fail_after_successes: int = 1
    structured_error: Exception | None = None
    tools_error: Exception | None = None
    tools_result: LLMAdapterResponse | None = None
    structured_result: ConversationInteractionDraft | None = None
    warmup_attempts: int = 0

    def supports_structured_output(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def _warmup_index(self, run_id: str | None) -> int | None:
        if not run_id:
            return None
        repetition = run_id.rsplit(":", 1)[-1]
        if not repetition.startswith("-"):
            return None
        return abs(int(repetition))

    def _should_fail_warmup(self, run_id: str | None) -> bool:
        warmup_index = self._warmup_index(run_id)
        if warmup_index is None:
            return False
        self.warmup_attempts = max(self.warmup_attempts, warmup_index)
        return warmup_index >= self.fail_after_successes

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        if self.structured_error is not None and self._should_fail_warmup(run_id):
            raise self.structured_error
        if self.structured_result is not None and self._should_fail_warmup(run_id):
            draft = self.structured_result
        else:
            draft = ConversationInteractionDraft(
                actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
            )
        return LLMStructuredResult(parsed=draft, response=LLMAdapterResponse(content=""))

    def generate_with_tools(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        run_id = kwargs.get("run_id")
        if self.tools_error is not None and self._should_fail_warmup(run_id):
            raise self.tools_error
        if self.tools_result is not None and self._should_fail_warmup(run_id):
            return self.tools_result
        draft = ConversationInteractionDraft(
            actions=(WorkspaceListDraftAction(action_type="workspace.list"),)
        )
        return LLMAdapterResponse(
            content="",
            finish_reason=LLMFinishReason.TOOL_CALLS,
            tool_calls=(
                LLMToolCall(
                    id="1",
                    name=SUBMIT_DRAFT_TOOL_NAME,
                    arguments_json=draft.model_dump_json(),
                ),
            ),
        )


def test_adapter_construction_failure_preserves_protocol_records() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]
    failed_model = enabled[0]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == failed_model:
            raise RuntimeError("adapter construction failed")
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert len(result.models) == 5
    failed = next(model for model in result.models if model.name == failed_model)
    assert len(failed.protocols) == 2
    assert [protocol.protocol for protocol in failed.protocols] == [
        "structured_output",
        "single_plan_tool",
    ]
    for protocol in failed.protocols:
        assert protocol.qualification_status == ProtocolStatus.PROVIDER_ERROR
        assert protocol.probe_failure_phase == FailurePhase.ADAPTER_CONSTRUCTION.value
        assert protocol.probe_safe_error_code == SafeErrorCode.OLLAMA_ADAPTER_CONSTRUCTION_FAILED.value
        assert protocol.provider_failure_count == 1
        assert protocol.failure_category_counts == {"PROVIDER_ERROR": 1}
    assert failed.metadata.digest
    assert failed.metadata.artifact_size_bytes == 1000


def test_adapter_construction_resource_failure() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]
    failed_model = enabled[0]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == failed_model:
            raise MemoryError("out of memory during adapter construction")
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    failed = next(model for model in result.models if model.name == failed_model)
    assert failed.status == ModelStatus.RESOURCE_LIMIT
    for protocol in failed.protocols:
        assert protocol.qualification_status == ProtocolStatus.RESOURCE_LIMIT
        assert protocol.provider_failure_count == 0
        assert protocol.failure_category_counts == {"RESOURCE_LIMIT": 1}
        assert protocol.probe_safe_error_code == SafeErrorCode.OLLAMA_RESOURCE_LIMIT.value


def test_capability_inspection_failure() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]
    failed_model = enabled[0]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == failed_model:
            return CapabilityFailAdapter(calls=[])
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    failed = next(model for model in result.models if model.name == failed_model)
    assert len(failed.protocols) == 2
    for protocol in failed.protocols:
        assert protocol.probe_failure_phase == FailurePhase.CAPABILITY_RESOLUTION.value
        assert protocol.probe_safe_error_code == SafeErrorCode.OLLAMA_CAPABILITY_RESOLUTION_FAILED.value


def test_metadata_phase_failure() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    with patch(
        "local_workspace_application.benchmarks.local_model_qualification.runner.build_inventory_metadata",
        side_effect=RuntimeError("metadata preparation failed"),
    ):
        result = run_benchmark(
            config,
            _provisioning(config),
            client_factory=_client_factory(enabled),
            adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
            generated_at_utc="2026-01-01T00:00:00+00:00",
            generated_from_commit="test",
        )
    for model in result.models:
        assert len(model.protocols) == 2
        for protocol in model.protocols:
            assert protocol.probe_failure_phase == FailurePhase.MODEL_METADATA.value
            assert protocol.probe_safe_error_code == SafeErrorCode.OLLAMA_MODEL_METADATA_FAILED.value


def test_summary_pair_count_with_one_failed_model() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]
    failed_model = enabled[0]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == failed_model:
            raise RuntimeError("adapter construction failed")
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert result.summary.expected_model_protocol_pairs == 10
    assert result.summary.attempted_model_protocol_pairs == 10
    assert result.summary.actual_scored_call_count < result.summary.expected_scored_call_count


def test_exit_code_two_model_level_provider_failure() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == enabled[0]:
            raise RuntimeError("adapter construction failed")
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 2


def test_exit_code_two_model_level_resource_failure() -> None:
    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        if model_name == enabled[0]:
            raise MemoryError("out of memory")
        return ScriptedAdapter(calls=[])

    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 2


def test_warmup_provider_failure_diagnostics() -> None:
    config = load_config(_CONFIG)
    model_cfg = next(model for model in config.models if model.enabled)
    adapter = CountingWarmupAdapter(
        fail_after_successes=1,
        structured_error=RuntimeError("provider transport failed"),
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="structured_output",
        adapter=adapter,
    )
    assert result.warmup_status == WarmupStatus.FAILED
    assert result.qualification_status == ProtocolStatus.WARMUP_FAILED
    assert result.provider_failure_count == 1
    assert result.failure_category_counts == {"PROVIDER_ERROR": 1}
    assert result.warmup_failure_category == StructuralFailureCategory.PROVIDER_ERROR.value
    assert result.warmup_failure_phase == FailurePhase.PROVIDER_INVOKE.value
    assert result.warmup_error_type == "RuntimeError"
    assert result.warmup_safe_error_code == SafeErrorCode.UNKNOWN_PROVIDER_FAILURE.value
    assert result.warmup_failure_repetition == 1
    assert result.warmup_failure_latency_ms is not None
    assert result.case_count == 0


def test_warmup_resource_failure_diagnostics() -> None:
    config = load_config(_CONFIG)
    model_cfg = next(model for model in config.models if model.enabled)
    adapter = CountingWarmupAdapter(
        fail_after_successes=1,
        structured_error=MemoryError("CUDA out of memory"),
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="structured_output",
        adapter=adapter,
    )
    assert result.qualification_status == ProtocolStatus.RESOURCE_LIMIT
    assert result.provider_failure_count == 0
    assert result.failure_category_counts == {"RESOURCE_LIMIT": 1}
    assert result.warmup_safe_error_code == SafeErrorCode.OLLAMA_RESOURCE_LIMIT.value


def test_warmup_missing_tool_call_diagnostics() -> None:
    config = load_config(_CONFIG)
    model_cfg = next(model for model in config.models if model.enabled)
    adapter = CountingWarmupAdapter(
        fail_after_successes=1,
        tools_result=LLMAdapterResponse(content="", tool_calls=()),
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="single_plan_tool",
        adapter=adapter,
    )
    assert result.qualification_status == ProtocolStatus.WARMUP_FAILED
    assert result.provider_failure_count == 0
    assert result.failure_category_counts == {"MISSING_PLAN_TOOL_CALL": 1}
    assert result.warmup_failure_phase == FailurePhase.TOOL_CALL_VALIDATION.value


def test_warmup_draft_compilation_failure_diagnostics() -> None:
    config = load_config(_CONFIG)
    model_cfg = next(model for model in config.models if model.enabled)
    bad_draft = ConversationInteractionDraft(
        actions=(
            KnowledgeAddSourcesDraftAction(
                action_type="knowledge.add_sources",
                workspace=NameDraftWorkspaceReference(
                    kind=WorkspaceReferenceKind.name,
                    value="finanse",
                ),
                sources=(
                    DraftWebUrlSource(
                        object_type="web_url",
                        value="https://not-in-message.example",
                    ),
                ),
            ),
        )
    )
    adapter = CountingWarmupAdapter(
        fail_after_successes=1,
        structured_result=bad_draft,
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="structured_output",
        adapter=adapter,
    )
    assert result.failure_category_counts == {"DRAFT_COMPILATION_FAILED": 1}
    assert result.warmup_failure_phase == FailurePhase.DRAFT_COMPILATION.value
    assert result.provider_failure_count == 0


def test_warmup_latency_excluded_from_scored_percentiles() -> None:
    config = load_config(_CONFIG)
    model_cfg = next(model for model in config.models if model.enabled)
    adapter = CountingWarmupAdapter(
        fail_after_successes=1,
        structured_error=RuntimeError("warmup failed"),
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="structured_output",
        adapter=adapter,
    )
    assert result.warmup_failure_latency_ms is not None
    assert result.warmup_failure_latency_ms > 0
    assert result.latency_ms.minimum == 0.0
    assert result.latency_ms.median == 0.0
    assert result.latency_ms.p95 == 0.0


def test_later_warmup_failure_stops_subsequent_warmups() -> None:
    config = load_config(_CONFIG)
    config = config.model_copy(
        update={"benchmark": config.benchmark.model_copy(update={"warmup_runs": 3})}
    )
    model_cfg = next(model for model in config.models if model.enabled)
    adapter = CountingWarmupAdapter(
        fail_after_successes=2,
        structured_error=RuntimeError("warmup 2 failed"),
    )
    result = run_protocol_benchmark(
        config=config,
        model=model_cfg,
        protocol="structured_output",
        adapter=adapter,
    )
    assert result.warmup_failure_repetition == 2
    assert adapter.warmup_attempts == 2


def test_exit_code_two_warmup_provider_failure() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: CountingWarmupAdapter(
            fail_after_successes=1,
            structured_error=RuntimeError("warmup provider failed"),
        ),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 2


def test_exit_code_two_warmup_resource_failure() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: CountingWarmupAdapter(
            fail_after_successes=1,
            structured_error=MemoryError("out of memory"),
        ),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 2


def test_exit_code_two_warmup_structural_failure() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: CountingWarmupAdapter(
            fail_after_successes=1,
            tools_result=LLMAdapterResponse(content="", tool_calls=()),
        ),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 2


def test_classify_model_preparation_error_by_phase() -> None:
    category, code = classify_model_preparation_error(
        RuntimeError("fail"),
        phase=FailurePhase.MODEL_METADATA,
    )
    assert category == StructuralFailureCategory.PROVIDER_ERROR
    assert code == SafeErrorCode.OLLAMA_MODEL_METADATA_FAILED

    category, code = classify_model_preparation_error(
        MemoryError("out of memory"),
        phase=FailurePhase.ADAPTER_CONSTRUCTION,
    )
    assert category == StructuralFailureCategory.RESOURCE_LIMIT
    assert code == SafeErrorCode.OLLAMA_RESOURCE_LIMIT
