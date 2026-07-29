# © Artur Czarnecki. All rights reserved.

"""TOML configuration loading and validation for local model qualification."""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_APPLICATION_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_CONFIG_PATH = _APPLICATION_ROOT / "scripts" / "local-model-qualification.toml"
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class _FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OllamaConfig(_FrozenConfig):
    host: str
    runtime: Literal["docker"]
    compose_file: str
    compose_service: str
    container_name: str
    continue_on_model_error: bool
    keep_alive: str
    startup_timeout_seconds: int
    model_pull_timeout_seconds: int
    readiness_poll_seconds: float

    @field_validator("host", "compose_service", "container_name")
    @classmethod
    def _validate_nonblank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("field must be nonblank")
        return trimmed

    @field_validator("compose_file")
    @classmethod
    def _validate_compose_file_relative(cls, value: str) -> str:
        if Path(value).is_absolute() or value.startswith(("/", "\\")):
            raise ValueError("compose_file must be relative")
        return value

    @field_validator("startup_timeout_seconds", "model_pull_timeout_seconds")
    @classmethod
    def _validate_timeout_seconds(cls, value: int) -> int:
        if value < 1:
            raise ValueError("timeout must be >= 1")
        return value

    @field_validator("readiness_poll_seconds")
    @classmethod
    def _validate_readiness_poll(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("readiness_poll_seconds must be > 0")
        return value


class BenchmarkConfig(_FrozenConfig):
    repetitions: int
    warmup_runs: int
    temperature: float
    max_tokens: int
    randomize_case_order: bool

    @field_validator("repetitions")
    @classmethod
    def _validate_repetitions(cls, value: int) -> int:
        if value < 1:
            raise ValueError("repetitions must be >= 1")
        return value

    @field_validator("warmup_runs")
    @classmethod
    def _validate_warmup_runs(cls, value: int) -> int:
        if value < 0:
            raise ValueError("warmup_runs must be >= 0")
        return value

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if value != value:  # NaN
            raise ValueError("temperature must be finite")
        if value in (float("inf"), float("-inf")):
            raise ValueError("temperature must be finite")
        return value

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, value: int) -> int:
        if value < 128:
            raise ValueError("max_tokens must be >= 128")
        return value


class ProtocolsConfig(_FrozenConfig):
    structured_output: bool
    single_plan_tool: bool

    @model_validator(mode="after")
    def _validate_at_least_one(self) -> Self:
        if not self.structured_output and not self.single_plan_tool:
            raise ValueError("at least one protocol must be enabled")
        return self


class OutputConfig(_FrozenConfig):
    results_json: str
    report_markdown: str


class QualificationConfig(_FrozenConfig):
    minimum_samples: int
    qualified_semantic_success_rate: float
    conditional_semantic_success_rate: float
    maximum_invalid_drafts: int
    maximum_provider_failures: int
    maximum_unsafe_state_changes: int

    @field_validator("minimum_samples")
    @classmethod
    def _validate_minimum_samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("minimum_samples must be >= 1")
        return value

    @field_validator(
        "qualified_semantic_success_rate",
        "conditional_semantic_success_rate",
    )
    @classmethod
    def _validate_rate(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("rate must be within 0.0–1.0")
        return value

    @field_validator(
        "maximum_invalid_drafts",
        "maximum_provider_failures",
        "maximum_unsafe_state_changes",
    )
    @classmethod
    def _validate_counters(cls, value: int) -> int:
        if value < 0:
            raise ValueError("maximum counter must be >= 0")
        return value


class ModelConfig(_FrozenConfig):
    name: str
    enabled: bool
    role: str

    @field_validator("name", "role")
    @classmethod
    def _validate_nonblank_text(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("model field must be nonblank")
        if _CONTROL_CHARS.search(trimmed):
            raise ValueError("model field must not contain control characters")
        return trimmed


class LocalModelQualificationConfig(_FrozenConfig):
    schema_version: int
    ollama: OllamaConfig
    benchmark: BenchmarkConfig
    protocols: ProtocolsConfig
    output: OutputConfig
    qualification: QualificationConfig
    models: tuple[ModelConfig, ...]
    config_path: Path = Field(exclude=True)
    application_root: Path = Field(exclude=True)
    repository_root: Path = Field(exclude=True)
    compose_file_path: Path = Field(exclude=True)
    results_json_path: Path = Field(exclude=True)
    report_markdown_path: Path = Field(exclude=True)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != 2:
            raise ValueError("schema_version must be 2")
        return value

    @model_validator(mode="after")
    def _validate_models(self) -> Self:
        enabled = [model for model in self.models if model.enabled]
        if not enabled:
            raise ValueError("at least one model must be enabled")
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        return self


def _resolve_output_path(config_dir: Path, application_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError("output paths must be relative")
    resolved = (config_dir / relative_path).resolve()
    try:
        resolved.relative_to(application_root.resolve())
    except ValueError as exc:
        raise ValueError("output path escapes application directory") from exc
    return resolved


def _resolve_compose_path(config_dir: Path, repository_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute() or relative_path.startswith(("/", "\\")):
        raise ValueError("compose_file must be relative")
    resolved = (config_dir / relative_path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError("compose_file escapes repository root") from exc
    if not resolved.exists():
        raise ValueError("compose_file does not exist")
    return resolved


def load_config(path: Path | None = None) -> LocalModelQualificationConfig:
    """Load and validate benchmark configuration from TOML."""
    config_path = path or _DEFAULT_CONFIG_PATH
    config_dir = config_path.parent
    application_root = _APPLICATION_ROOT.resolve()
    repository_root = _REPO_ROOT.resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    allowed_keys = {
        "schema_version",
        "ollama",
        "benchmark",
        "protocols",
        "output",
        "qualification",
        "models",
    }
    unknown_keys = set(raw) - allowed_keys
    if unknown_keys:
        raise ValueError(f"unknown configuration keys: {sorted(unknown_keys)}")
    models_raw = raw.get("models", [])
    output = raw["output"]
    ollama_raw = raw["ollama"]
    compose_file_path = _resolve_compose_path(
        config_dir,
        repository_root,
        ollama_raw["compose_file"],
    )
    base_config = LocalModelQualificationConfig(
        schema_version=raw["schema_version"],
        ollama=OllamaConfig.model_validate(ollama_raw),
        benchmark=BenchmarkConfig.model_validate(raw["benchmark"]),
        protocols=ProtocolsConfig.model_validate(raw["protocols"]),
        output=OutputConfig.model_validate(output),
        qualification=QualificationConfig.model_validate(raw["qualification"]),
        models=tuple(ModelConfig.model_validate(item) for item in models_raw),
        config_path=config_path,
        application_root=application_root,
        repository_root=repository_root,
        compose_file_path=compose_file_path,
        results_json_path=application_root / "placeholder.json",
        report_markdown_path=application_root / "placeholder.md",
    )
    return base_config.model_copy(
        update={
            "results_json_path": _resolve_output_path(
                config_dir,
                application_root,
                output["results_json"],
            ),
            "report_markdown_path": _resolve_output_path(
                config_dir,
                application_root,
                output["report_markdown"],
            ),
        }
    )


def configuration_sha256(config: LocalModelQualificationConfig) -> str:
    digest = hashlib.sha256()
    digest.update(config.config_path.read_bytes())
    return digest.hexdigest()


def enabled_model_names(config: LocalModelQualificationConfig) -> tuple[str, ...]:
    return tuple(model.name for model in config.models if model.enabled)
