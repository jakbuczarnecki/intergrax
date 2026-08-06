# © Artur Czarnecki. All rights reserved.

"""Strict TOML loading for the universal Token Optimization proof harness."""

from __future__ import annotations

import math
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofAdapterConfig,
    ProofCaseInput,
    ProofConfigurationError,
    ProofOutputConfig,
    ProofPipelineConfig,
    ProofRouterConfig,
    SCHEMA_VERSION,
    UniversalTokenOptimizationProofConfig,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUN_MODES = frozenset({"offline_smoke", "live_adapter"})
_ADAPTER_TYPES = frozenset({"openai_compatible"})
# TOKEN-10F v1 transport profile, not global registry inventory.
_OPENAI_COMPATIBLE_PROVIDERS = frozenset(
    {
        LLMProvider.OPENAI.value,
        LLMProvider.VLLM.value,
        LLMProvider.GROQ.value,
        LLMProvider.TOGETHER.value,
        LLMProvider.FIREWORKS.value,
        LLMProvider.OPENROUTER.value,
        LLMProvider.DEEPSEEK.value,
        LLMProvider.XAI.value,
        LLMProvider.LLAMA_CPP.value,
        LLMProvider.COHERE.value,
        LLMProvider.AZURE_AI_INFERENCE.value,
    }
)
_PIPELINE_MODES = frozenset({"default", "replace"})
_FAILURE_POLICIES = frozenset({"continue", "fail_fast"})


def _fail(reason_code: str) -> ProofConfigurationError:
    return ProofConfigurationError(reason_code)


def _table(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"INVALID_{name.upper()}_TABLE")
    return value


def _strict_keys(table: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise _fail(f"UNKNOWN_{name.upper()}_FIELD")


def _required(table: Mapping[str, Any], name: str) -> Any:
    if name not in table:
        raise _fail(f"MISSING_{name.upper()}")
    return table[name]


def _strict_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or not _ID_RE.fullmatch(value):
        raise _fail(f"INVALID_{field_name.upper()}")
    return value


def _strict_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _fail(f"INVALID_{field_name.upper()}")
    return value


def _strict_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise _fail(f"INVALID_{field_name.upper()}")
    return value


def _strict_float(value: object, field_name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"INVALID_{field_name.upper()}")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise _fail(f"INVALID_{field_name.upper()}")
    return result


def _strict_non_negative_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"INVALID_{field_name.upper()}")
    return value


def _relative_path(value: object, field_name: str, *, base_dir: Path) -> Path:
    raw = _strict_string(value, field_name)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise _fail(f"UNSAFE_{field_name.upper()}_PATH")
    return (base_dir / path).resolve()


def _repository_root(source_path: Path) -> Path:
    """Resolve repository-relative output paths from the config source."""
    for candidate in (source_path.parent, *source_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / ".git").exists():
            return candidate
    return source_path.parent


def _canonical_provider(value: object) -> str:
    provider = _strict_string(value, "provider")
    try:
        canonical = LLMProvider(provider)
    except ValueError as exc:
        raise _fail("UNSUPPORTED_ADAPTER") from exc
    if (
        canonical.value not in LLMAdapterRegistry.registered_providers()
        or canonical.value not in _OPENAI_COMPATIBLE_PROVIDERS
    ):
        raise _fail("UNSUPPORTED_ADAPTER")
    return canonical.value


def _parse_adapter(table: Mapping[str, Any]) -> ProofAdapterConfig:
    _strict_keys(
        table,
        frozenset(
            {
                "adapter_id",
                "provider",
                "type",
                "model",
                "base_url",
                "api_key_env",
                "timeout_seconds",
                "max_output_tokens",
                "temperature",
            }
        ),
        "adapter",
    )
    provider = _canonical_provider(_required(table, "provider"))
    adapter_type = _strict_string(_required(table, "type"), "adapter_type")
    if adapter_type not in _ADAPTER_TYPES:
        raise _fail("UNSUPPORTED_ADAPTER")
    api_key_env = table.get("api_key_env")
    if api_key_env is not None:
        api_key_env = _strict_string(api_key_env, "api_key_env")
        if not _ENV_RE.fullmatch(api_key_env):
            raise _fail("INVALID_API_KEY_ENV")
    return ProofAdapterConfig(
        adapter_id=_strict_id(_required(table, "adapter_id"), "adapter_id"),
        provider=provider,
        adapter_type=adapter_type,
        model=_strict_string(_required(table, "model"), "model"),
        base_url=_strict_string(_required(table, "base_url"), "base_url"),
        api_key_env=api_key_env,
        timeout_seconds=_strict_float(
            _required(table, "timeout_seconds"),
            "timeout_seconds",
            minimum=0.001,
            maximum=86400.0,
        ),
        max_output_tokens=_strict_non_negative_int(
            _required(table, "max_output_tokens"),
            "max_output_tokens",
            minimum=1,
        ),
        temperature=_strict_float(
            _required(table, "temperature"),
            "temperature",
            minimum=0.0,
            maximum=2.0,
        ),
    )


def _parse_router(table: Mapping[str, Any]) -> ProofRouterConfig:
    _strict_keys(
        table,
        frozenset(
            {
                "enabled",
                "configuration_id",
                "minimum_confidence",
                "allow_structured_output_fallback",
                "require_review_for_protected_lossy_content",
            }
        ),
        "router",
    )
    return ProofRouterConfig(
        enabled=_strict_bool(_required(table, "enabled"), "router_enabled"),
        configuration_id=_strict_id(
            _required(table, "configuration_id"),
            "configuration_id",
        ),
        minimum_confidence=_strict_float(
            _required(table, "minimum_confidence"),
            "minimum_confidence",
            minimum=0.0,
            maximum=1.0,
        ),
        allow_structured_output_fallback=_strict_bool(
            _required(table, "allow_structured_output_fallback"),
            "allow_structured_output_fallback",
        ),
        require_review_for_protected_lossy_content=_strict_bool(
            _required(table, "require_review_for_protected_lossy_content"),
            "require_review_for_protected_lossy_content",
        ),
    )


def _parse_pipeline(table: Mapping[str, Any]) -> ProofPipelineConfig:
    _strict_keys(table, frozenset({"mode", "layer_ids", "failure_policy"}), "pipeline")
    mode = _strict_string(_required(table, "mode"), "pipeline_mode")
    failure_policy = _strict_string(
        _required(table, "failure_policy"),
        "failure_policy",
    )
    layer_ids = _required(table, "layer_ids")
    if (
        not isinstance(layer_ids, list)
        or any(not isinstance(layer_id, str) for layer_id in layer_ids)
    ):
        raise _fail("INVALID_LAYER_IDS")
    parsed_layer_ids = tuple(_strict_id(layer_id, "layer_id") for layer_id in layer_ids)
    if len(parsed_layer_ids) != len(set(parsed_layer_ids)):
        raise _fail("DUPLICATE_LAYER_IDS")
    if mode not in _PIPELINE_MODES or failure_policy not in _FAILURE_POLICIES:
        raise _fail("INVALID_PIPELINE_ENUM")
    return ProofPipelineConfig(
        mode=mode,
        layer_ids=parsed_layer_ids,
        failure_policy=failure_policy,
    )


def _parse_policy(table: Mapping[str, Any]) -> TokenOptimizationPolicy:
    _strict_keys(
        table,
        frozenset(
            {
                "enabled",
                "profile",
                "allow_lossy",
                "require_validation",
                "fallback_on_validation_failure",
            }
        ),
        "case_policy",
    )
    try:
        profile = TokenOptimizationProfile(
            _strict_string(_required(table, "profile"), "profile")
        )
    except ValueError as exc:
        raise _fail("INVALID_PROFILE") from exc
    return TokenOptimizationPolicy(
        enabled=_strict_bool(_required(table, "enabled"), "policy_enabled"),
        profile=profile,
        allow_lossy=_strict_bool(_required(table, "allow_lossy"), "allow_lossy"),
        require_validation=_strict_bool(
            _required(table, "require_validation"),
            "require_validation",
        ),
        fallback_on_validation_failure=_strict_bool(
            _required(table, "fallback_on_validation_failure"),
            "fallback_on_validation_failure",
        ),
    )


def _parse_protected_regions(value: object) -> tuple[ProtectedRegion, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _fail("INVALID_PROTECTED_REGIONS")
    regions: list[ProtectedRegion] = []
    for item in value:
        region = _table(item, "protected_region")
        _strict_keys(region, frozenset({"kind", "value"}), "protected_region")
        try:
            kind = ProtectedRegionKind(
                _strict_string(_required(region, "kind"), "protected_region_kind")
            )
        except ValueError as exc:
            raise _fail("INVALID_PROTECTED_REGION_KIND") from exc
        regions.append(
            ProtectedRegion(
                kind=kind,
                value=_strict_string(
                    _required(region, "value"),
                    "protected_region_value",
                ),
            )
        )
    return tuple(regions)


def _parse_case(table: Mapping[str, Any]) -> ProofCaseInput:
    _strict_keys(
        table,
        frozenset(
            {"case_id", "source_type", "content", "tags", "policy", "protected_regions"}
        ),
        "case",
    )
    try:
        source_type = TokenOptimizationSourceType(
            _strict_string(_required(table, "source_type"), "source_type")
        )
    except ValueError as exc:
        raise _fail("INVALID_SOURCE_TYPE") from exc
    tags = table.get("tags", [])
    if not isinstance(tags, list):
        raise _fail("INVALID_TAGS")
    parsed_tags = tuple(_strict_id(tag, "tag") for tag in tags)
    return ProofCaseInput(
        case_id=_strict_id(_required(table, "case_id"), "case_id"),
        request=TokenOptimizationRequest(
            content=_strict_string(_required(table, "content"), "content"),
            source_type=source_type,
            policy=_parse_policy(_table(_required(table, "policy"), "case_policy")),
            protected_regions=_parse_protected_regions(table.get("protected_regions")),
        ),
        tags=parsed_tags,
    )


def _parse_cases(value: object) -> tuple[ProofCaseInput, ...]:
    if not isinstance(value, list) or not value:
        raise _fail("INVALID_CASES")
    cases = tuple(_parse_case(_table(item, "case")) for item in value)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise _fail("DUPLICATE_CASE_IDS")
    return cases


def _parse_case_source(path: Path) -> tuple[ProofCaseInput, ...]:
    try:
        data = tomllib.loads(path.read_bytes().decode("utf-8"))
    except FileNotFoundError as exc:
        raise _fail("CASE_SOURCE_NOT_FOUND") from exc
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _fail("INVALID_CASE_SOURCE") from exc
    _strict_keys(data, frozenset({"cases"}), "case_source")
    return _parse_cases(data.get("cases"))


def load_universal_token_optimization_proof_config(
    path: str | Path,
) -> UniversalTokenOptimizationProofConfig:
    """Load and fully validate one strict, secret-free TOML configuration."""
    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise _fail("CONFIG_NOT_FOUND")
    source_path = source_path.resolve()
    try:
        data = tomllib.loads(source_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _fail("INVALID_TOML") from exc

    _strict_keys(
        data,
        frozenset(
            {
                "schema_version",
                "proof_id",
                "run_mode",
                "adapter",
                "router",
                "pipeline",
                "output",
                "cases",
                "case_source",
            }
        ),
        "root",
    )
    schema_version = _strict_string(_required(data, "schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise _fail("UNSUPPORTED_SCHEMA_VERSION")
    run_mode = _strict_string(_required(data, "run_mode"), "run_mode")
    if run_mode not in _RUN_MODES:
        raise _fail("INVALID_RUN_MODE")
    adapter = _parse_adapter(_table(_required(data, "adapter"), "adapter"))
    if run_mode == "live_adapter" and adapter.api_key_env and not os.environ.get(
        adapter.api_key_env
    ):
        raise _fail("MISSING_API_KEY_ENV")
    router = _parse_router(_table(_required(data, "router"), "router"))
    pipeline = _parse_pipeline(_table(_required(data, "pipeline"), "pipeline"))
    repository_root = _repository_root(source_path)

    output_table = _table(_required(data, "output"), "output")
    _strict_keys(output_table, frozenset({"directory", "fail_if_exists"}), "output")
    output = ProofOutputConfig(
        directory=_relative_path(
            _required(output_table, "directory"),
            "output_directory",
            base_dir=repository_root,
        ),
        fail_if_exists=_strict_bool(
            _required(output_table, "fail_if_exists"),
            "fail_if_exists",
        ),
    )

    has_cases = "cases" in data
    has_source = "case_source" in data
    if has_cases == has_source:
        raise _fail("CASES_OR_CASE_SOURCE_REQUIRED")
    case_source = (
        _relative_path(data["case_source"], "case_source", base_dir=source_path.parent)
        if has_source
        else None
    )
    cases = _parse_cases(data["cases"]) if has_cases else _parse_case_source(case_source)
    try:
        return UniversalTokenOptimizationProofConfig(
            schema_version=schema_version,
            proof_id=_strict_id(_required(data, "proof_id"), "proof_id"),
            run_mode=run_mode,
            adapter=adapter,
            router=router,
            pipeline=pipeline,
            output=output,
            cases=cases,
            case_source=case_source,
            source_path=source_path,
        )
    except ValueError as exc:
        raise _fail("INVALID_PROOF_CONFIG") from exc


__all__ = [
    "load_universal_token_optimization_proof_config",
]
