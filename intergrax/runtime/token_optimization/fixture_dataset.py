# © Artur Czarnecki. All rights reserved.

"""File-backed token regression fixture dataset loader (Phase TOKEN-OBS-2E-B)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from intergrax.memory.summary_compressor import (
    DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY,
    MemorySummaryCompressionConfig,
    optimize_memory_summary,
)
from intergrax.runtime.token_optimization.context_pack import (
    DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
    ContextFragment,
    ContextPackOptimizationConfig,
    optimize_context_pack,
)
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
)
from intergrax.runtime.token_optimization.regression import (
    RegressionRunner,
    TokenCounter,
    TokenRegressionExpectation,
    TokenRegressionFixture,
    TokenRegressionSourceType,
)
from intergrax.runtime.token_optimization.tool_schema import (
    DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
    ToolSchemaOptimizationConfig,
    optimize_tool_schema_catalog,
)

_PATH_SEGMENT_RE = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")

_REQUIRED_DATASET_KEYS = frozenset(
    {"schema_version", "dataset_id", "dataset_version", "case_discovery"}
)
_REQUIRED_FIXTURE_KEYS = frozenset(
    {
        "schema_version",
        "fixture_id",
        "source_type",
        "eval_case",
        "category",
        "expected_behavior",
        "description",
        "input",
        "optimizer",
        "protected_values",
        "expected",
    }
)
_ALLOWED_EVAL_CASES = frozenset({"compactable", "protected", "fallback"})
_ALLOWED_INPUT_FORMATS = frozenset(
    {
        "tool_schema_catalog",
        "context_pack_fragments",
        "plain_text",
    }
)
_ALLOWED_OPTIMIZER_KINDS = frozenset(
    {
        "tool_schema",
        "context_pack",
        "memory_summary",
    }
)
_ALLOWED_EXPECTED_RECEIPTS = frozenset({"required"})
_ALLOWED_EXPECTED_VALIDATIONS = frozenset({"pass_like", "failed"})
_ALLOWED_EXPECTED_FALLBACKS = frozenset({"forbidden", "required"})
_ALLOWED_SOURCE_TYPES = frozenset(member.value for member in TokenRegressionSourceType)


@dataclass(frozen=True, slots=True)
class TokenRegressionFixtureDataset:
    """Loaded file-backed regression fixture dataset."""

    dataset_id: str
    dataset_version: str
    fixtures: tuple[TokenRegressionFixture, ...]
    metadata: Mapping[str, Any]


def load_token_regression_fixture_dataset(dataset_path: str | Path) -> TokenRegressionFixtureDataset:
    """Load regression fixtures from a case-per-folder dataset directory."""
    root = Path(dataset_path).resolve()
    dataset_spec = _read_json(root / "dataset.json")
    _validate_dataset_contract(dataset_spec)

    dataset_id = str(dataset_spec["dataset_id"])
    dataset_version = str(dataset_spec["dataset_version"])
    cases_root = root / str(dataset_spec["case_discovery"]["root"])

    fixture_paths = sorted(cases_root.rglob("fixture.json"))
    if not fixture_paths:
        raise ValueError(f"No fixture.json files found under {cases_root}")

    fixtures: list[TokenRegressionFixture] = []
    seen_fixture_ids: set[str] = set()

    for fixture_path in fixture_paths:
        fixture_spec = _read_json(fixture_path)
        _validate_fixture_contract(fixture_spec)

        fixture_id = str(fixture_spec["fixture_id"])
        if fixture_id in seen_fixture_ids:
            raise ValueError(f"Duplicate fixture_id: {fixture_id}")
        seen_fixture_ids.add(fixture_id)

        case_dir = fixture_path.parent
        relative_fixture_path = fixture_path.relative_to(root).as_posix()
        protected_values = fixture_spec.get("protected_values", [])
        input_spec = fixture_spec["input"]
        optimizer_spec = fixture_spec["optimizer"]
        expected_spec = fixture_spec["expected"]

        fixtures.append(
            TokenRegressionFixture(
                fixture_id=fixture_id,
                source_type=TokenRegressionSourceType(str(fixture_spec["source_type"])),
                description=str(fixture_spec["description"]),
                expectation=_map_expectation(expected_spec),
                runner=_build_runner(case_dir, input_spec, optimizer_spec),
                metadata=_safe_fixture_metadata(
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    fixture_path=relative_fixture_path,
                    category=str(fixture_spec["category"]),
                    eval_case=str(fixture_spec["eval_case"]),
                    expected_behavior=str(fixture_spec["expected_behavior"]),
                    input_format=str(input_spec.get("format", "")),
                    optimizer_kind=str(optimizer_spec.get("kind", "")),
                    protected_value_count=len(protected_values),
                ),
            )
        )

    fixtures.sort(key=lambda fixture: fixture.fixture_id)
    return TokenRegressionFixtureDataset(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        fixtures=tuple(fixtures),
        metadata={
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "fixture_count": len(fixtures),
            "case_discovery_strategy": dataset_spec["case_discovery"]["strategy"],
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _validate_dataset_contract(dataset_spec: Mapping[str, Any]) -> None:
    missing = _REQUIRED_DATASET_KEYS - set(dataset_spec)
    if missing:
        raise ValueError(f"dataset.json missing required keys: {sorted(missing)}")
    if dataset_spec["schema_version"] != 1:
        raise ValueError(
            f"Unsupported dataset schema_version: {dataset_spec['schema_version']!r}"
        )
    discovery = dataset_spec["case_discovery"]
    if not isinstance(discovery, Mapping):
        raise ValueError("dataset.json case_discovery must be an object")
    if discovery.get("strategy") != "recursive_fixture_json":
        raise ValueError(
            "dataset.json case_discovery.strategy must be 'recursive_fixture_json'"
        )
    if discovery.get("root") != "cases":
        raise ValueError("dataset.json case_discovery.root must be 'cases'")


def _validate_fixture_contract(fixture_spec: Mapping[str, Any]) -> None:
    fixture_id = str(fixture_spec.get("fixture_id", "<unknown>"))
    missing = _REQUIRED_FIXTURE_KEYS - set(fixture_spec)
    if missing:
        raise ValueError(
            f"fixture {fixture_id} missing required keys: {sorted(missing)}"
        )
    if fixture_spec["schema_version"] != 1:
        raise ValueError(
            f"fixture {fixture_id} has unsupported schema_version: "
            f"{fixture_spec['schema_version']!r}"
        )
    source_type = fixture_spec["source_type"]
    if source_type not in _ALLOWED_SOURCE_TYPES:
        raise ValueError(
            f"fixture {fixture_id} has unsupported source_type: {source_type!r}"
        )
    eval_case = fixture_spec["eval_case"]
    if eval_case not in _ALLOWED_EVAL_CASES:
        raise ValueError(
            f"fixture {fixture_id} has unsupported eval_case: {eval_case!r}"
        )
    input_spec = fixture_spec["input"]
    if not isinstance(input_spec, Mapping):
        raise ValueError(f"fixture {fixture_id} input must be an object")
    input_format = input_spec.get("format")
    if input_format not in _ALLOWED_INPUT_FORMATS:
        raise ValueError(
            f"fixture {fixture_id} has unsupported input.format: {input_format!r}"
        )
    optimizer_spec = fixture_spec["optimizer"]
    if not isinstance(optimizer_spec, Mapping):
        raise ValueError(f"fixture {fixture_id} optimizer must be an object")
    optimizer_kind = optimizer_spec.get("kind")
    if optimizer_kind not in _ALLOWED_OPTIMIZER_KINDS:
        raise ValueError(
            f"fixture {fixture_id} has unsupported optimizer.kind: {optimizer_kind!r}"
        )
    expected_spec = fixture_spec["expected"]
    if not isinstance(expected_spec, Mapping):
        raise ValueError(f"fixture {fixture_id} expected must be an object")
    receipt = expected_spec.get("receipt")
    if receipt not in _ALLOWED_EXPECTED_RECEIPTS:
        raise ValueError(
            f"fixture {fixture_id} has unsupported expected.receipt: {receipt!r}"
        )
    validation = expected_spec.get("validation")
    if validation not in _ALLOWED_EXPECTED_VALIDATIONS:
        raise ValueError(
            f"fixture {fixture_id} has unsupported expected.validation: {validation!r}"
        )
    fallback = expected_spec.get("fallback")
    if fallback not in _ALLOWED_EXPECTED_FALLBACKS:
        raise ValueError(
            f"fixture {fixture_id} has unsupported expected.fallback: {fallback!r}"
        )
    protected_values = fixture_spec["protected_values"]
    if not isinstance(protected_values, list):
        raise ValueError(f"fixture {fixture_id} protected_values must be a list")


def _safe_fixture_metadata(
    *,
    dataset_id: str,
    dataset_version: str,
    fixture_path: str,
    category: str,
    eval_case: str,
    expected_behavior: str,
    input_format: str,
    optimizer_kind: str,
    protected_value_count: int,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "fixture_path": fixture_path,
        "category": category,
        "eval_case": eval_case,
        "expected_behavior": expected_behavior,
        "input_format": input_format,
        "optimizer_kind": optimizer_kind,
        "protected_value_count": protected_value_count,
    }


def _map_expectation(expected_spec: Mapping[str, Any]) -> TokenRegressionExpectation:
    receipt = expected_spec.get("receipt")
    validation = expected_spec.get("validation")
    fallback = expected_spec.get("fallback")

    require_receipt = receipt == "required"
    expect_validation_pass = validation != "failed"
    expected_validation_status: str | None = None
    if validation == "failed":
        expected_validation_status = "failed"

    allow_fallback = fallback == "required"
    expect_fallback: bool | None = None
    if fallback == "forbidden":
        expect_fallback = False
    elif fallback == "required":
        expect_fallback = True

    return TokenRegressionExpectation(
        expected_min_saved_tokens=expected_spec.get("min_saved_tokens"),
        expected_min_saved_ratio=expected_spec.get("min_saved_ratio"),
        expected_max_saved_tokens=expected_spec.get("max_saved_tokens"),
        expected_max_saved_ratio=expected_spec.get("max_saved_ratio"),
        expected_validation_status=expected_validation_status,
        expect_fallback=expect_fallback,
        require_receipt=require_receipt,
        expect_validation_pass=expect_validation_pass,
        allow_fallback=allow_fallback,
    )


def _build_runner(
    case_dir: Path,
    input_spec: Mapping[str, Any],
    optimizer_spec: Mapping[str, Any],
) -> RegressionRunner:
    input_format = input_spec.get("format")
    optimizer_kind = optimizer_spec.get("kind")

    if optimizer_kind == "tool_schema":
        if input_format != "tool_schema_catalog":
            raise ValueError(
                f"tool_schema optimizer requires tool_schema_catalog input, got {input_format!r}"
            )
        catalog_json = _load_tool_schema_catalog(case_dir, input_spec)
        config = _build_dataclass_config(
            ToolSchemaOptimizationConfig,
            optimizer_spec.get("config"),
        )
        return _make_tool_schema_runner(catalog_json, config)

    if optimizer_kind == "context_pack":
        if input_format != "context_pack_fragments":
            raise ValueError(
                f"context_pack optimizer requires context_pack_fragments input, "
                f"got {input_format!r}"
            )
        fragments = _load_context_pack_fragments(case_dir, input_spec)
        config = _build_dataclass_config(
            ContextPackOptimizationConfig,
            optimizer_spec.get("config"),
        )
        return _make_context_pack_runner(fragments, config)

    if optimizer_kind == "memory_summary":
        if input_format != "plain_text":
            raise ValueError(
                f"memory_summary optimizer requires plain_text input, got {input_format!r}"
            )
        summary_text = _load_plain_text_input(case_dir, input_spec)
        config = _build_dataclass_config(
            MemorySummaryCompressionConfig,
            optimizer_spec.get("config"),
        )
        policy_override = optimizer_spec.get("policy_override")
        token_policy = (
            _build_token_policy(policy_override)
            if isinstance(policy_override, Mapping)
            else DEFAULT_MEMORY_SUMMARY_TOKEN_POLICY
        )
        hook_name = optimizer_spec.get("semantic_validation_hook")
        semantic_hook = _resolve_semantic_validation_hook(hook_name)
        return _make_memory_summary_runner(
            summary_text,
            token_policy,
            config,
            semantic_hook,
        )

    raise ValueError(f"Unsupported optimizer kind: {optimizer_kind!r}")


def _load_tool_schema_catalog(case_dir: Path, input_spec: Mapping[str, Any]) -> str:
    schema_file = case_dir / str(input_spec["schema_file"])
    catalog = _read_json(schema_file)
    text_fields = input_spec.get("text_fields", {})
    if not isinstance(text_fields, Mapping):
        raise ValueError("tool_schema_catalog text_fields must be an object")
    for json_path, relative_text_file in text_fields.items():
        text_path = case_dir / str(relative_text_file)
        text_value = text_path.read_text(encoding="utf-8")
        _set_json_path_value(catalog, str(json_path), text_value)
    return json.dumps(catalog, indent=2)


def _load_context_pack_fragments(
    case_dir: Path,
    input_spec: Mapping[str, Any],
) -> tuple[ContextFragment, ...]:
    fragment_specs = input_spec.get("fragments")
    if not isinstance(fragment_specs, list):
        raise ValueError("context_pack_fragments input requires fragments list")
    fragments: list[ContextFragment] = []
    for fragment_spec in fragment_specs:
        if not isinstance(fragment_spec, Mapping):
            raise ValueError("Each context_pack fragment entry must be an object")
        fragment_id = str(fragment_spec["fragment_id"])
        content = (case_dir / str(fragment_spec["file"])).read_text(encoding="utf-8")
        metadata = fragment_spec.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Fragment {fragment_id} metadata must be an object")
        fragments.append(
            ContextFragment(
                fragment_id=fragment_id,
                content=content,
                required=bool(fragment_spec.get("required", False)),
                metadata=dict(metadata),
            )
        )
    return tuple(fragments)


def _load_plain_text_input(case_dir: Path, input_spec: Mapping[str, Any]) -> str:
    return (case_dir / str(input_spec["file"])).read_text(encoding="utf-8")


def _build_dataclass_config(
    config_cls: type[Any],
    config_data: object,
) -> Any | None:
    if not config_data:
        return None
    if not isinstance(config_data, Mapping):
        raise ValueError(f"{config_cls.__name__} config must be an object")
    allowed = {field.name for field in fields(config_cls)}
    unknown = set(config_data) - allowed
    if unknown:
        raise ValueError(
            f"{config_cls.__name__} config contains unsupported keys: {sorted(unknown)}"
        )
    return config_cls(**dict(config_data))


def _build_token_policy(policy_data: Mapping[str, Any]) -> TokenOptimizationPolicy:
    allowed = {field.name for field in fields(TokenOptimizationPolicy)}
    unknown = set(policy_data) - allowed
    if unknown:
        raise ValueError(
            "TokenOptimizationPolicy override contains unsupported keys: "
            f"{sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for key, value in policy_data.items():
        if key == "profile":
            kwargs[key] = TokenOptimizationProfile(str(value))
        elif key == "compression_level":
            kwargs[key] = CompressionLevel(str(value))
        else:
            kwargs[key] = value
    return TokenOptimizationPolicy(**kwargs)


def _resolve_semantic_validation_hook(hook_name: object) -> Any | None:
    if hook_name is None:
        return None
    if hook_name == "accept":
        return _accept_semantic_validation
    raise ValueError(f"Unsupported semantic_validation_hook: {hook_name!r}")


def _accept_semantic_validation(
    _original_content: str,
    _optimized_content: str,
    _metadata: object,
) -> bool:
    return True


def _make_tool_schema_runner(
    catalog_json: str,
    config: ToolSchemaOptimizationConfig | None,
) -> RegressionRunner:
    def runner(token_counter: TokenCounter) -> object:
        return optimize_tool_schema_catalog(
            catalog_json,
            token_policy=DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
            config=config,
            token_counter=token_counter,
        )

    return runner


def _make_context_pack_runner(
    fragments: tuple[ContextFragment, ...],
    config: ContextPackOptimizationConfig | None,
) -> RegressionRunner:
    def runner(token_counter: TokenCounter) -> object:
        return optimize_context_pack(
            fragments,
            token_policy=DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
            config=config,
            token_counter=token_counter,
        )

    return runner


def _make_memory_summary_runner(
    summary_text: str,
    token_policy: TokenOptimizationPolicy,
    config: MemorySummaryCompressionConfig | None,
    semantic_validation_hook: Any | None,
) -> RegressionRunner:
    def runner(token_counter: TokenCounter) -> object:
        return optimize_memory_summary(
            summary_text,
            token_policy=token_policy,
            config=config,
            semantic_validation_hook=semantic_validation_hook,
            token_counter=token_counter,
        )

    return runner


def _set_json_path_value(root: object, json_path: str, value: str) -> None:
    if not json_path.startswith("$."):
        raise ValueError(f"Unsupported JSONPath (must start with '$.'): {json_path!r}")
    segments = _parse_json_path_segments(json_path[2:])
    if not segments:
        raise ValueError(f"Empty JSONPath: {json_path!r}")
    target = root
    for segment in segments[:-1]:
        target = _traverse_json_path_segment(target, segment)
    _assign_json_path_segment(target, segments[-1], value)


def _parse_json_path_segments(path: str) -> list[str | int]:
    segments: list[str | int] = []
    for raw_segment in path.split("."):
        if not raw_segment:
            raise ValueError(f"Invalid JSONPath segment in {path!r}")
        match = _PATH_SEGMENT_RE.match(raw_segment)
        if match is None:
            raise ValueError(f"Unsupported JSONPath segment: {raw_segment!r}")
        key = match.group(1)
        index = match.group(2)
        segments.append(key)
        if index is not None:
            segments.append(int(index))
    return segments


def _traverse_json_path_segment(current: object, segment: str | int) -> object:
    if isinstance(segment, str):
        if not isinstance(current, Mapping):
            raise ValueError(f"Expected object at segment {segment!r}")
        if segment not in current:
            raise ValueError(f"Missing object key: {segment!r}")
        return current[segment]
    if not isinstance(current, list):
        raise ValueError(f"Expected list at index {segment}")
    if segment < 0 or segment >= len(current):
        raise ValueError(f"List index out of range: {segment}")
    return current[segment]


def _assign_json_path_segment(current: object, segment: str | int, value: str) -> None:
    if isinstance(segment, str):
        if not isinstance(current, dict):
            raise ValueError(f"Expected object for assignment at {segment!r}")
        current[segment] = value
        return
    if not isinstance(current, list):
        raise ValueError(f"Expected list for assignment at index {segment}")
    if segment < 0 or segment >= len(current):
        raise ValueError(f"List index out of range: {segment}")
    current[segment] = value
