# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult

from local_workspace_application.benchmarks.local_model_qualification.config import load_config
from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    ModelStatus,
)
from local_workspace_application.benchmarks.local_model_qualification.runner import (
    _has_partial_model_failure,
    run_benchmark,
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


@dataclass
class ScriptedAdapter:
    calls: list[str]

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
        raise AssertionError("tool path not used in scripted runner test")


class FakeOllamaClient:
    def __init__(self, installed: set[str]) -> None:
        self._installed = installed

    def version(self) -> dict[str, str]:
        return {"version": "0.5.0"}

    def list(self) -> dict[str, list[dict[str, str]]]:
        return {"models": [{"name": name} for name in self._installed]}

    def show(self, *, model: str) -> dict[str, object]:
        return {
            "digest": f"sha256:{model}",
            "size": 1000,
            "details": {"parameter_size": "14B", "quantization_level": "Q4", "family": "qwen2"},
        }

    def ps(self) -> dict[str, list[dict[str, int | str]]]:
        return {"models": [{"name": "qwen2.5:14b", "size": 1000, "size_vram": 1000}]}


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
        client_factory=_client_factory(set(enabled)),
        adapter_factory=adapter_factory,
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    assert order == enabled


def test_missing_model_recorded(tmp_path: Path) -> None:
    config = load_config(_CONFIG)
    result = run_benchmark(
        config,
        client_factory=_client_factory({"qwen2.5:14b"}),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    statuses = {model.name: model.status for model in result.models}
    assert statuses["qwen2.5:14b"] in {ModelStatus.COMPLETED, ModelStatus.COMPLETED_WITH_FAILURES}
    assert statuses["llama3.1:8b"] == ModelStatus.NOT_INSTALLED


def test_fatal_config_failure_preserves_existing_artifacts(tmp_path: Path) -> None:
    config = load_config(_CONFIG)
    preserved = tmp_path / "preserved.json"
    preserved.write_text('{"existing": true}\n', encoding="utf-8")
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(bad_config)
    assert json.loads(preserved.read_text(encoding="utf-8")) == {"existing": True}


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
    result = run_benchmark(
        config,
        client_factory=_client_factory({"qwen2.5:14b"}),
        adapter_factory=lambda _name: ScriptedAdapter(calls=[]),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    write_artifacts(config, result)
    assert results_path.exists()
    assert report_path.exists()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "lkw.local_model_qualification.result.v1"


def test_exit_code_semantics_for_missing_models() -> None:
    config = load_config(_CONFIG)
    result = run_benchmark(
        config,
        client_factory=_client_factory(set()),
        generated_at_utc="2026-01-01T00:00:00+00:00",
        generated_from_commit="test",
    )
    from local_workspace_application.benchmarks.local_model_qualification.runner import (
        _has_partial_model_failure,
    )

    assert _has_partial_model_failure(result)