# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from local_workspace_application.benchmarks.local_model_qualification.config import OllamaConfig
from local_workspace_application.benchmarks.local_model_qualification.contracts import ObservedExecutionMode
from local_workspace_application.benchmarks.local_model_qualification.environment import (
    ModelInventoryRecord,
    build_inventory_metadata,
    derive_execution_mode,
    fetch_model_inventory,
    fetch_runtime_metadata,
)


def _ollama_config() -> OllamaConfig:
    return OllamaConfig(
        host="http://localhost:11434",
        runtime="docker",
        compose_file="../../../infra/docker/ollama/docker-compose.yml",
        compose_service="ollama",
        container_name="intergrax-ollama",
        continue_on_model_error=True,
        keep_alive="10m",
        startup_timeout_seconds=120,
        model_pull_timeout_seconds=7200,
        readiness_poll_seconds=1.0,
    )


@dataclass
class TypedListResponse:
    models: list[object]


@dataclass
class TypedInventoryItem:
    name: str
    digest: str
    size: int
    details: dict[str, str] | None = None


@dataclass
class TypedPsItem:
    model: str
    size: int
    size_vram: int


class MappingListClient:
    def list(self) -> dict[str, list[dict[str, object]]]:
        return {
            "models": [
                {"name": "qwen2.5:14b", "digest": "sha256:abc", "size": 5000},
            ]
        }

    def ps(self) -> dict[str, list[dict[str, object]]]:
        return {"models": [{"name": "qwen2.5:14b", "size": 1000, "size_vram": 950}]}


class TypedListClient:
    def list(self) -> TypedListResponse:
        return TypedListResponse(
            models=[
                TypedInventoryItem(name="qwen2.5:14b", digest="sha256:typed", size=6000),
            ]
        )

    def ps(self) -> TypedListResponse:
        return TypedListResponse(
            models=[TypedPsItem(model="qwen2.5:14b", size=2000, size_vram=1900)]
        )


class NameFieldPsClient:
    def list(self) -> dict[str, list[dict[str, object]]]:
        return {"models": [{"model": "qwen2.5:14b", "digest": "sha256:name", "size": 7000}]}

    def ps(self) -> dict[str, list[dict[str, object]]]:
        return {"models": [{"name": "qwen2.5:14b", "size": 3000, "size_vram": 0}]}


class ShowClient:
    def show(self, *, model: str) -> SimpleNamespace:
        return SimpleNamespace(
            model_dump=lambda: {
                "details": {
                    "parameter_size": "14B",
                    "quantization_level": "Q4_K_M",
                    "family": "qwen2",
                },
                "model_info": {"context_length": 32768},
            }
        )


def test_mapping_client_list_normalized() -> None:
    inventory = fetch_model_inventory(_ollama_config(), client_factory=lambda _host: MappingListClient())
    assert inventory["qwen2.5:14b"].digest == "sha256:abc"
    assert inventory["qwen2.5:14b"].artifact_size_bytes == 5000


def test_typed_client_list_normalized() -> None:
    inventory = fetch_model_inventory(_ollama_config(), client_factory=lambda _host: TypedListClient())
    assert inventory["qwen2.5:14b"].digest == "sha256:typed"
    assert inventory["qwen2.5:14b"].artifact_size_bytes == 6000


def test_digest_from_inventory_not_modelfile() -> None:
    client = MappingListClient()

    class ShowWithModelfile:
        def list(self):
            return client.list()

        def show(self, *, model: str) -> dict[str, object]:
            return {"modelfile": "FROM qwen2.5", "size": 1}

    inventory = fetch_model_inventory(_ollama_config(), client_factory=lambda _host: ShowWithModelfile())
    metadata = build_inventory_metadata(inventory["qwen2.5:14b"], {"modelfile": "FROM qwen2.5"})
    assert metadata.digest == "sha256:abc"


def test_typed_client_show_normalized() -> None:
    show = ShowClient().show(model="qwen2.5:14b")
    payload = show.model_dump()
    record = ModelInventoryRecord(name="qwen2.5:14b", digest="sha256:typed", artifact_size_bytes=1)
    metadata = build_inventory_metadata(record, payload)
    assert metadata.parameter_size == "14B"
    assert metadata.quantization_level == "Q4_K_M"
    assert metadata.model_family == "qwen2"
    assert metadata.context_length == 32768


def test_ps_supports_model_field() -> None:
    loaded, vram = fetch_runtime_metadata(
        _ollama_config(),
        "qwen2.5:14b",
        client_factory=lambda _host: TypedListClient(),
    )
    assert loaded == 2000
    assert vram == 1900


def test_ps_supports_name_field() -> None:
    loaded, vram = fetch_runtime_metadata(
        _ollama_config(),
        "qwen2.5:14b",
        client_factory=lambda _host: NameFieldPsClient(),
    )
    assert loaded == 3000
    assert vram == 0


def test_full_gpu_only_from_measured_allocation() -> None:
    metadata = build_inventory_metadata(
        ModelInventoryRecord(name="m", digest="sha256:x", artifact_size_bytes=1),
        {},
    )
    metadata = metadata.model_copy(update={"loaded_size_bytes": 1000, "size_vram_bytes": 960})
    assert derive_execution_mode(metadata) == ObservedExecutionMode.FULL_GPU


def test_partial_gpu_offload_derived_correctly() -> None:
    metadata = build_inventory_metadata(
        ModelInventoryRecord(name="m", digest="sha256:x", artifact_size_bytes=1),
        {},
    )
    metadata = metadata.model_copy(update={"loaded_size_bytes": 1000, "size_vram_bytes": 400})
    assert derive_execution_mode(metadata) == ObservedExecutionMode.PARTIAL_GPU_OFFLOAD


def test_cpu_only_derived_correctly() -> None:
    metadata = build_inventory_metadata(
        ModelInventoryRecord(name="m", digest="sha256:x", artifact_size_bytes=1),
        {},
    )
    metadata = metadata.model_copy(update={"loaded_size_bytes": 1000, "size_vram_bytes": 0})
    assert derive_execution_mode(metadata) == ObservedExecutionMode.CPU_ONLY


def test_unknown_when_process_metadata_unavailable() -> None:
    metadata = build_inventory_metadata(
        ModelInventoryRecord(name="m", digest="sha256:x", artifact_size_bytes=1),
        {},
    )
    assert derive_execution_mode(metadata) == ObservedExecutionMode.UNKNOWN
