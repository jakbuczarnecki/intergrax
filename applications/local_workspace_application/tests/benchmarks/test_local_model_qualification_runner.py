# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

from local_workspace_application.benchmarks.local_model_qualification.config import load_config
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    ModelProvisioningStatus,
    ModelStatus,
    ProtocolStatus,
    ProvisionedModel,
    ProvisioningResult,
    RESULT_SCHEMA_VERSION,
)
from local_workspace_application.benchmarks.local_model_qualification.provisioning import ProvisioningError
from local_workspace_application.benchmarks.local_model_qualification.runner import (
    compute_exit_code,
    run_benchmark,
    run_from_config,
    write_artifacts,
)
from local_workspace_application.conversation.interaction_draft_models import (
    ConversationInteractionDraft,
    WorkspaceListDraftAction,
)

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
        from intergrax.llm_adapters.contracts.finish_reason import LLMFinishReason
        from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
        from local_workspace_application.benchmarks.local_model_qualification.protocols import (
            SUBMIT_DRAFT_TOOL_NAME,
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
        self.ps_calls = 0

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
        self.ps_calls += 1
        return {"models": [{"model": "qwen2.5:14b", "size": 1000, "size_vram": 1000}]}


def _client_factory(installed: set[str]):
    def factory(host: str) -> FakeOllamaClient:
        return FakeOllamaClient(installed)

    return factory


def test_models_processed_sequentially() -> None:
    order: list[str] = []

    def adapter_factory(model_name: str) -> ScriptedAdapter:
        order.append(model_name)
        return ScriptedAdapter(calls=[])

    config = load_config(_CONFIG)
    enabled = [model.name for model in config.models if model.enabled]
    run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert order == enabled


def test_five_enabled_models_produce_five_results() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert len(result.models) == 5
    assert all(model.installed for model in result.models)
    assert all(len(model.protocols) == 2 for model in result.models)


def test_ps_called_after_inference() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    client = FakeOllamaClient(enabled)
    run_benchmark(
        config,
        _provisioning(config),
        client_factory=lambda _host: client,
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert client.ps_calls >= 1


def test_report_rebuilt_after_normal_model_failures(tmp_path: Path) -> None:
    config = load_config(_CONFIG)
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    config = config.model_copy(
        update={
            "results_json_path": results_path,
            "report_markdown_path": report_path,
        }
    )
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    write_artifacts(config, result)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == RESULT_SCHEMA_VERSION


def test_provisioning_failure_preserves_old_artifacts(tmp_path: Path) -> None:
    config = load_config(_CONFIG)
    results_path = tmp_path / "results.json"
    report_path = tmp_path / "report.md"
    preserved_json = '{"existing": true}\n'
    preserved_md = "# preserved\n"
    results_path.write_text(preserved_json, encoding="utf-8")
    report_path.write_text(preserved_md, encoding="utf-8")
    config = config.model_copy(
        update={
            "results_json_path": results_path,
            "report_markdown_path": report_path,
        }
    )

    def fail_provision(_config, **kwargs):
        raise ProvisioningError("DOCKER_OLLAMA_START_FAILED")

    with pytest.raises(ProvisioningError):
        run_from_config(config, provision=fail_provision)
    assert results_path.read_text(encoding="utf-8") == preserved_json
    assert report_path.read_text(encoding="utf-8") == preserved_md


def test_exit_code_zero_semantics() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert compute_exit_code(result) == 0


def test_exit_code_two_for_provider_failure() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    protocol = result.models[0].protocols[0].model_copy(
        update={"qualification_status": ProtocolStatus.PROVIDER_ERROR}
    )
    model = result.models[0].model_copy(update={"protocols": (protocol,)})
    modified = result.model_copy(update={"models": (model,)})
    assert compute_exit_code(modified) == 2


def test_all_models_have_digest_and_size() -> None:
    config = load_config(_CONFIG)
    enabled = {model.name for model in config.models if model.enabled}
    result = run_benchmark(
        config,
        _provisioning(config),
        client_factory=_client_factory(enabled),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    for model in result.models:
        assert model.metadata.digest
        assert model.metadata.artifact_size_bytes and model.metadata.artifact_size_bytes > 0
        assert model.status in {ModelStatus.COMPLETED, ModelStatus.COMPLETED_WITH_FAILURES}
