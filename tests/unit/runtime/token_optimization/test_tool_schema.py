# © Artur Czarnecki. All rights reserved.

"""TOKEN-3: ToolSchemaOptimizer tests."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenSavingsClaimConfidence,
)
from intergrax.runtime.token_optimization.output_policy import (
    OutputPolicyResolutionStatus,
)
from intergrax.runtime.token_optimization.tool_schema import (
    DEFAULT_TOOL_CATALOG_TOKEN_POLICY,
    ToolSchemaOptimizationConfig,
    ToolSchemaOptimizationStatus,
    ToolSchemaOptimizer,
    optimize_tool_schema_catalog,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TOOL_SCHEMA_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "tool_schema.py"
)

def _sample_catalog_with_examples() -> dict[str, object]:
    return {
        "tools": [
            {
                "name": "search_files",
                "description": "Search files.",
                "examples": [{"query": "invoice"}],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "enum": ["name", "content", "path"],
                            "examples": ["invoice", "contract"],
                        }
                    },
                    "required": ["query"],
                },
            }
        ]
    }


def _sample_catalog(*, long_description: str | None = None) -> dict[str, object]:
    description = long_description or "  Search   the   workspace  "
    return {
        "tools": [
            {
                "name": "search_files",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "  Search   query  ",
                            "enum": ["name", "content", "path"],
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]
    }


def _enabled_policy() -> TokenOptimizationPolicy:
    return TokenOptimizationPolicy(
        enabled=True,
        profile=TokenOptimizationProfile.CONSERVATIVE,
        compression_level=CompressionLevel.LIGHT,
        allow_lossy=False,
        require_validation=True,
        fallback_on_validation_failure=True,
        emit_receipts=True,
    )


def test_optimize_compacts_json_whitespace_for_tool_catalog() -> None:
    catalog = _sample_catalog()
    pretty_input = json.dumps(catalog, indent=2)

    outcome = optimize_tool_schema_catalog(
        pretty_input,
        token_policy=_enabled_policy(),
    )

    assert "\n" not in outcome.optimized_content
    assert len(outcome.optimized_content) < len(pretty_input)
    assert outcome.changed is True
    assert outcome.status is ToolSchemaOptimizationStatus.APPLIED
    assert '"name":"search_files"' in outcome.optimized_content


def test_long_descriptions_are_shortened_deterministically() -> None:
    long_description = "alpha " * 120
    catalog = _sample_catalog(long_description=long_description)
    config = ToolSchemaOptimizationConfig(max_description_chars=80)

    outcome = optimize_tool_schema_catalog(
        catalog,
        token_policy=_enabled_policy(),
        config=config,
    )

    parsed = json.loads(outcome.optimized_content)
    shortened = parsed["tools"][0]["description"]
    assert len(shortened) <= 80
    assert shortened.endswith("…")
    assert outcome.result.metadata["description_fields_compacted"] >= 1


def test_short_descriptions_are_preserved_except_whitespace_normalization() -> None:
    catalog = _sample_catalog(long_description="  Keep   this   text  ")

    outcome = optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())

    parsed = json.loads(outcome.optimized_content)
    assert parsed["tools"][0]["description"] == "Keep this text"


def test_tool_name_parameter_names_required_fields_types_enums_properties_preserved() -> None:
    catalog = _sample_catalog()

    outcome = optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())
    parsed = json.loads(outcome.optimized_content)
    tool = parsed["tools"][0]
    params = tool["parameters"]

    assert tool["name"] == "search_files"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["query"]["enum"] == ["name", "content", "path"]
    assert params["properties"]["limit"]["type"] == "integer"


def test_input_mapping_and_list_are_not_mutated() -> None:
    catalog = _sample_catalog()
    catalog_snapshot = copy.deepcopy(catalog)
    tool_list = [catalog["tools"][0]]
    tool_list_snapshot = copy.deepcopy(tool_list)

    optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())
    optimize_tool_schema_catalog(tool_list, token_policy=_enabled_policy())

    assert catalog == catalog_snapshot
    assert tool_list == tool_list_snapshot


def test_default_config_preserves_examples() -> None:
    catalog = _sample_catalog_with_examples()

    outcome = optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())

    parsed = json.loads(outcome.optimized_content)
    tool = parsed["tools"][0]
    assert "examples" in tool
    assert tool["examples"] == [{"query": "invoice"}]
    assert "examples" in tool["parameters"]["properties"]["query"]
    assert tool["parameters"]["properties"]["query"]["examples"] == [
        "invoice",
        "contract",
    ]
    assert outcome.result.metadata.get("examples_removed", 0) == 0


def test_allow_example_removal_removes_examples() -> None:
    catalog = _sample_catalog_with_examples()
    config = ToolSchemaOptimizationConfig(allow_example_removal=True)

    outcome = optimize_tool_schema_catalog(
        catalog,
        token_policy=_enabled_policy(),
        config=config,
    )

    parsed = json.loads(outcome.optimized_content)
    tool = parsed["tools"][0]
    assert "examples" not in tool
    assert "examples" not in tool["parameters"]["properties"]["query"]
    assert outcome.result.metadata["examples_removed"] == 2


def test_allow_example_removal_preserves_schema_semantics() -> None:
    catalog = _sample_catalog_with_examples()
    config = ToolSchemaOptimizationConfig(allow_example_removal=True)

    outcome = optimize_tool_schema_catalog(
        catalog,
        token_policy=_enabled_policy(),
        config=config,
    )

    parsed = json.loads(outcome.optimized_content)
    tool = parsed["tools"][0]
    params = tool["parameters"]

    assert tool["name"] == "search_files"
    assert "query" in params["properties"]
    assert params["required"] == ["query"]
    assert params["type"] == "object"
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["query"]["enum"] == ["name", "content", "path"]


def test_input_not_mutated_when_examples_are_removed() -> None:
    catalog = _sample_catalog_with_examples()
    catalog_snapshot = copy.deepcopy(catalog)
    config = ToolSchemaOptimizationConfig(allow_example_removal=True)

    optimize_tool_schema_catalog(
        catalog,
        token_policy=_enabled_policy(),
        config=config,
    )

    assert catalog == catalog_snapshot
    assert "examples" in catalog["tools"][0]
    assert "examples" in catalog["tools"][0]["parameters"]["properties"]["query"]


def test_runtime_schema_snapshot_unchanged_after_optimization() -> None:
    original_tool_schema = _sample_catalog()
    original_tool_schema_snapshot = copy.deepcopy(original_tool_schema)

    optimize_tool_schema_catalog(original_tool_schema, token_policy=_enabled_policy())

    assert original_tool_schema == original_tool_schema_snapshot


def test_disabled_token_policy_bypasses_optimization() -> None:
    catalog = _sample_catalog()
    disabled = TokenOptimizationPolicy(enabled=False)

    outcome = optimize_tool_schema_catalog(catalog, token_policy=disabled)

    assert outcome.original_content == outcome.optimized_content
    assert outcome.changed is False
    assert outcome.status is ToolSchemaOptimizationStatus.BYPASSED
    assert outcome.result.decision is TokenOptimizationDecision.BYPASS
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_resolved_output_policy_is_present_in_outcome() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
    )

    assert outcome.resolved_output_policy is not None
    assert outcome.resolved_output_policy.enabled is True
    assert outcome.resolved_output_policy.status is OutputPolicyResolutionStatus.RESOLVED


def test_protected_regions_in_descriptions_are_preserved_at_field_level() -> None:
    prefix = "context " * 40
    protected_url = "https://example.com/protected/resource"
    description = f"{prefix}See {protected_url} for details and more usage notes."
    catalog = _sample_catalog(long_description=description)
    config = ToolSchemaOptimizationConfig(max_description_chars=120)

    outcome = optimize_tool_schema_catalog(
        catalog,
        token_policy=_enabled_policy(),
        config=config,
    )

    parsed = json.loads(outcome.optimized_content)
    result_description = parsed["tools"][0]["description"]
    assert protected_url in result_description
    assert outcome.result.metadata["description_fields_preserved_due_to_protected_regions"] >= 1


def test_global_protected_region_validation_passes_on_applied_optimization() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
    )

    assert outcome.protected_region_validation.status in {
        ProtectedRegionValidationStatus.PASSED,
        ProtectedRegionValidationStatus.NOT_APPLICABLE,
    }


def test_validation_failure_causes_fallback_to_original_content() -> None:
    failed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing protected region",),
    )
    catalog = _sample_catalog()

    with patch(
        "intergrax.runtime.token_optimization.tool_schema.validate_protected_regions",
        return_value=failed_validation,
    ):
        outcome = optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())

    assert outcome.optimized_content == outcome.original_content
    assert outcome.result.fallback_used is True
    assert outcome.result.decision is TokenOptimizationDecision.FALLBACK
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED
    assert outcome.status is ToolSchemaOptimizationStatus.FALLBACK


def test_receipt_created_for_applied_optimization() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
        config=ToolSchemaOptimizationConfig(include_receipt=True),
    )

    assert outcome.receipt is not None
    assert outcome.receipt.receipt_id.startswith("receipt_")


def test_receipt_created_for_bypass_when_include_receipt_true() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=TokenOptimizationPolicy(enabled=False),
        config=ToolSchemaOptimizationConfig(include_receipt=True),
    )

    assert outcome.receipt is not None
    assert outcome.result.decision is TokenOptimizationDecision.BYPASS


def test_receipt_created_for_fallback_when_include_receipt_true() -> None:
    failed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing protected region",),
    )

    with patch(
        "intergrax.runtime.token_optimization.tool_schema.validate_protected_regions",
        return_value=failed_validation,
    ):
        outcome = optimize_tool_schema_catalog(
            _sample_catalog(),
            token_policy=_enabled_policy(),
            config=ToolSchemaOptimizationConfig(include_receipt=True),
        )

    assert outcome.receipt is not None
    assert outcome.result.decision is TokenOptimizationDecision.FALLBACK


def test_receipt_ref_maps_to_receipt() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
    )

    assert outcome.receipt_ref is not None
    assert outcome.receipt is not None
    assert outcome.receipt_ref.receipt_id == outcome.receipt.receipt_id
    assert outcome.receipt_ref.original_hash == outcome.receipt.original_hash
    assert outcome.receipt_ref.optimized_hash == outcome.receipt.optimized_hash


def test_measurement_none_when_token_counter_not_provided() -> None:
    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
    )

    assert outcome.result.measurement is None


def test_measurement_created_when_token_counter_provided() -> None:
    def counter(text: str) -> int:
        return len(text)

    outcome = optimize_tool_schema_catalog(
        _sample_catalog(),
        token_policy=_enabled_policy(),
        token_counter=counter,
    )

    measurement = outcome.result.measurement
    assert measurement is not None
    assert measurement.confidence is TokenSavingsClaimConfidence.MEASURED
    assert measurement.saved_tokens == measurement.baseline_tokens - measurement.optimized_tokens


def test_default_policy_enables_safe_compaction_when_token_policy_is_none() -> None:
    outcome = optimize_tool_schema_catalog(_sample_catalog())

    assert outcome.changed is True
    assert outcome.request.policy == DEFAULT_TOOL_CATALOG_TOKEN_POLICY


def test_tool_schema_optimizer_class_wrapper_matches_helper() -> None:
    catalog = _sample_catalog()
    optimizer = ToolSchemaOptimizer()

    class_outcome = optimizer.optimize_catalog(catalog, token_policy=_enabled_policy())
    helper_outcome = optimize_tool_schema_catalog(catalog, token_policy=_enabled_policy())

    assert class_outcome.optimized_content == helper_outcome.optimized_content


def test_no_tokenizer_model_runtime_or_telemetry_imports_introduced() -> None:
    source = _TOOL_SCHEMA_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    forbidden_roots = {
        "openai",
        "anthropic",
        "tiktoken",
        "transformers",
        "httpx",
        "requests",
    }
    assert imported_roots.isdisjoint(forbidden_roots)
    assert "intergrax.runtime.nexus" not in source
    assert "intergrax.runtime.token_optimization.telemetry" not in source
