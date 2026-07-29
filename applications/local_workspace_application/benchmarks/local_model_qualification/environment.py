# © Artur Czarnecki. All rights reserved.

"""Environment and Ollama metadata collection for qualification runs."""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    HostMetadata,
    ModelMetadata,
    ObservedExecutionMode,
    OllamaEnvironment,
)
from local_workspace_application.benchmarks.local_model_qualification.config import OllamaConfig


@dataclass(frozen=True, slots=True)
class ModelInventoryRecord:
    name: str
    digest: str
    artifact_size_bytes: int


def collect_host_metadata() -> HostMetadata:
    cpu_description: str | None = None
    if platform.processor():
        cpu_description = platform.processor()
    elif sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["wmic", "cpu", "get", "Name"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            lines = [
                line.strip()
                for line in output.splitlines()
                if line.strip() and line.strip() != "Name"
            ]
            if lines:
                cpu_description = lines[0]
        except (OSError, subprocess.SubprocessError):
            cpu_description = None

    total_ram: int | None = None
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            lines = [line.strip() for line in output.splitlines() if line.strip().isdigit()]
            if lines:
                total_ram = int(lines[0])
        except (OSError, subprocess.SubprocessError, ValueError):
            total_ram = None

    gpu_name: str | None = None
    gpu_vram: int | None = None
    driver_version: str | None = None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        first_line = output.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if parts:
            gpu_name = parts[0]
        if len(parts) > 1:
            try:
                gpu_vram = int(float(parts[1]) * 1024 * 1024)
            except ValueError:
                gpu_vram = None
        if len(parts) > 2:
            driver_version = parts[2]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass

    return HostMetadata(
        operating_system=platform.system(),
        os_release=platform.release(),
        machine_architecture=platform.machine(),
        python_version=platform.python_version(),
        cpu_description=cpu_description,
        total_system_ram_bytes=total_ram,
        gpu_name=gpu_name,
        gpu_total_vram_bytes=gpu_vram,
        nvidia_driver_version=driver_version,
    )


def _client_factory(host: str, timeout: float | None = None) -> Any:
    from ollama import Client

    if timeout is not None:
        return Client(host=host, timeout=timeout)
    return Client(host=host)


def _normalize_models_list(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("models", []))
    models = getattr(response, "models", [])
    return list(models) if models is not None else []


def _model_name(item: Any) -> str | None:
    if isinstance(item, dict):
        name = item.get("name") or item.get("model")
    else:
        name = getattr(item, "model", None) or getattr(item, "name", None)
    return str(name) if name else None


def _to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "__dict__"):
        return {key: value for key, value in vars(item).items() if not key.startswith("_")}
    return {}


def collect_ollama_environment(
    config: OllamaConfig,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> OllamaEnvironment:
    factory = client_factory or (lambda host: _client_factory(host))
    version: str | None = None
    try:
        client = factory(config.host)
        if hasattr(client, "version"):
            version_info = client.version()
            if isinstance(version_info, dict):
                version = version_info.get("version")
            else:
                version = getattr(version_info, "version", None)
    except Exception:
        version = None
    if version is None:
        try:
            import httpx

            response = httpx.get(f"{config.host.rstrip('/')}/api/version", timeout=5.0)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                version = payload.get("version")
        except Exception:
            version = None
    return OllamaEnvironment(version=version, host=config.host)


def list_installed_models(
    config: OllamaConfig,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> set[str]:
    factory = client_factory or (lambda host: _client_factory(host))
    client = factory(config.host)
    response = client.list()
    names: set[str] = set()
    for item in _normalize_models_list(response):
        name = _model_name(item)
        if name:
            names.add(name)
    return names


def fetch_model_inventory(
    config: OllamaConfig,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, ModelInventoryRecord]:
    factory = client_factory or (lambda host: _client_factory(host))
    client = factory(config.host)
    response = client.list()
    inventory: dict[str, ModelInventoryRecord] = {}
    for item in _normalize_models_list(response):
        data = _to_dict(item)
        name = data.get("name") or data.get("model")
        if not name:
            continue
        name = str(name)
        digest = data.get("digest")
        size = data.get("size")
        if isinstance(digest, str) and digest.strip() and isinstance(size, int) and size > 0:
            inventory[name] = ModelInventoryRecord(
                name=name,
                digest=digest.strip(),
                artifact_size_bytes=size,
            )
    return inventory


def pull_model(
    config: OllamaConfig,
    model_name: str,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> None:
    factory = client_factory or (
        lambda host: _client_factory(host, timeout=float(config.model_pull_timeout_seconds))
    )
    client = factory(config.host)
    client.pull(model=model_name, stream=False)


def fetch_show_metadata(
    config: OllamaConfig,
    model_name: str,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    factory = client_factory or (lambda host: _client_factory(host))
    client = factory(config.host)
    try:
        show = client.show(model=model_name)
    except Exception:
        return {}
    if not isinstance(show, dict):
        show = show.model_dump() if hasattr(show, "model_dump") else {}
    return show if isinstance(show, dict) else {}


def build_inventory_metadata(
    inventory_record: ModelInventoryRecord,
    show: dict[str, Any],
) -> ModelMetadata:
    details = show.get("details", {}) if isinstance(show.get("details"), dict) else {}
    model_info = show.get("model_info", {}) if isinstance(show.get("model_info"), dict) else {}

    parameter_size = details.get("parameter_size") or model_info.get("parameter_size")
    quantization = details.get("quantization_level") or model_info.get("quantization_level")
    family = details.get("family") or model_info.get("family")

    context_length = None
    for key in ("context_length", "num_ctx"):
        if key in model_info:
            try:
                context_length = int(model_info[key])
                break
            except (TypeError, ValueError):
                pass

    return ModelMetadata(
        digest=inventory_record.digest,
        artifact_size_bytes=inventory_record.artifact_size_bytes,
        parameter_size=str(parameter_size) if parameter_size else None,
        quantization_level=str(quantization) if quantization else None,
        model_family=str(family) if family else None,
        context_length=context_length,
    )


def fetch_runtime_metadata(
    config: OllamaConfig,
    model_name: str,
    *,
    client_factory: Callable[[str], Any] | None = None,
) -> tuple[int | None, int | None]:
    factory = client_factory or (lambda host: _client_factory(host))
    client = factory(config.host)
    try:
        ps = client.ps()
        for item in _normalize_models_list(ps):
            data = _to_dict(item)
            name = data.get("name") or data.get("model")
            if name != model_name:
                continue
            loaded_size = data.get("size")
            size_vram = data.get("size_vram")
            return (
                int(loaded_size) if isinstance(loaded_size, int) else None,
                int(size_vram) if isinstance(size_vram, int) else None,
            )
    except Exception:
        pass
    return None, None


def merge_runtime_metadata(
    metadata: ModelMetadata,
    loaded_size: int | None,
    size_vram: int | None,
) -> ModelMetadata:
    return metadata.model_copy(
        update={
            "loaded_size_bytes": loaded_size,
            "size_vram_bytes": size_vram,
        }
    )


def derive_execution_mode(metadata: ModelMetadata) -> ObservedExecutionMode:
    loaded = metadata.loaded_size_bytes
    vram = metadata.size_vram_bytes
    if loaded is None or vram is None:
        return ObservedExecutionMode.UNKNOWN
    if loaded <= 0:
        return ObservedExecutionMode.UNKNOWN
    if vram == 0:
        return ObservedExecutionMode.CPU_ONLY
    if vram >= 0.95 * loaded:
        return ObservedExecutionMode.FULL_GPU
    if 0 < vram < 0.95 * loaded:
        return ObservedExecutionMode.PARTIAL_GPU_OFFLOAD
    return ObservedExecutionMode.UNKNOWN


def check_ollama_readiness(
    config: OllamaConfig,
    *,
    client_factory: Callable[[str], Any] | None = None,
    http_get: Callable[[str], Any] | None = None,
) -> bool:
    factory = client_factory or (lambda host: _client_factory(host))
    try:
        client = factory(config.host)
        if hasattr(client, "version"):
            client.version()
            return True
    except Exception:
        pass
    if http_get is not None:
        try:
            response = http_get(f"{config.host.rstrip('/')}/api/version")
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            return True
        except Exception:
            return False
    try:
        import httpx

        response = httpx.get(f"{config.host.rstrip('/')}/api/version", timeout=5.0)
        response.raise_for_status()
        return True
    except Exception:
        return False
