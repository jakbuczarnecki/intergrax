# © Artur Czarnecki. All rights reserved.

"""TOKEN-4: ContextPackOptimizer tests."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.runtime.token_optimization.context_pack import (
    DEFAULT_CONTEXT_PACK_TOKEN_POLICY,
    ContextFragment,
    ContextPackOptimizationConfig,
    ContextPackOptimizationStatus,
    ContextPackOptimizer,
    optimize_context_pack,
)
from intergrax.runtime.token_optimization.contracts import (
    CompressionLevel,
    ProtectedRegionValidationResult,
    ProtectedRegionValidationStatus,
    TokenOptimizationBypassReason,
    TokenOptimizationDecision,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
    TokenSavingsClaimConfidence,
)
from intergrax.runtime.token_optimization.output_policy import (
    OutputPolicyResolutionStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CONTEXT_PACK_MODULE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "token_optimization"
    / "context_pack.py"
)


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


def _sample_fragments(*, long_content: str | None = None) -> list[ContextFragment]:
    return [
        ContextFragment(
            fragment_id="evidence_1",
            content=long_content or "  Retrieved   evidence   fragment  ",
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            required=False,
            metadata={"source": "rag", "rank": 1},
        ),
        ContextFragment(
            fragment_id="policy_1",
            content="  Mandatory   policy   text  ",
            source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
            required=True,
            metadata={"source": "policy"},
        ),
    ]


def test_optimize_compacts_whitespace_in_non_required_fragments() -> None:
    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=_enabled_policy(),
    )

    optimized = outcome.optimized_fragments[0]
    assert optimized.content == "Retrieved evidence fragment"
    assert outcome.changed is True
    assert outcome.status is ContextPackOptimizationStatus.APPLIED


def test_long_non_required_fragments_are_truncated_deterministically() -> None:
    long_content = "alpha " * 400
    config = ContextPackOptimizationConfig(max_fragment_chars=80)

    outcome = optimize_context_pack(
        [ContextFragment(fragment_id="long_1", content=long_content)],
        token_policy=_enabled_policy(),
        config=config,
    )

    optimized = outcome.optimized_fragments[0]
    assert len(optimized.content) <= 80
    assert optimized.content.endswith("…")
    assert outcome.metadata["fragments_truncated"] >= 1


def test_required_fragments_are_preserved_exactly() -> None:
    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=_enabled_policy(),
    )

    required = outcome.optimized_fragments[1]
    assert required.content == "  Mandatory   policy   text  "
    assert outcome.metadata["fragments_preserved_required"] >= 1


def test_fragment_id_is_preserved() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert [f.fragment_id for f in outcome.optimized_fragments] == [
        "evidence_1",
        "policy_1",
    ]


def test_source_type_is_preserved() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.optimized_fragments[0].source_type is TokenOptimizationSourceType.RAG_CONTEXT_PACK
    assert outcome.optimized_fragments[1].source_type is TokenOptimizationSourceType.SYSTEM_POLICY


def test_metadata_provenance_is_preserved() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.optimized_fragments[0].metadata == {"source": "rag", "rank": 1}
    assert outcome.optimized_fragments[1].metadata == {"source": "policy"}


def test_fragment_order_is_preserved() -> None:
    fragments = [
        ContextFragment(fragment_id="first", content="one"),
        ContextFragment(fragment_id="second", content="two"),
        ContextFragment(fragment_id="third", content="three"),
    ]

    outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())

    assert [f.fragment_id for f in outcome.optimized_fragments] == [
        "first",
        "second",
        "third",
    ]


def test_input_fragments_are_not_mutated() -> None:
    fragments = _sample_fragments()
    snapshot = [
        ContextFragment(
            fragment_id=f.fragment_id,
            content=f.content,
            source_type=f.source_type,
            required=f.required,
            metadata=dict(f.metadata),
        )
        for f in fragments
    ]
    mapping = {
        "id": "map_1",
        "text": "  mapping   text  ",
        "metadata": {"path": "/workspace/a.py"},
    }
    mapping_snapshot = copy.deepcopy(mapping)

    optimize_context_pack(fragments, token_policy=_enabled_policy())
    optimize_context_pack([mapping], token_policy=_enabled_policy())

    assert fragments[0].content == snapshot[0].content
    assert fragments[1].content == snapshot[1].content
    assert mapping == mapping_snapshot


def test_raw_string_fragments_get_deterministic_fragment_ids() -> None:
    outcome = optimize_context_pack(
        ["  first  ", "  second  "],
        token_policy=_enabled_policy(),
    )

    assert [f.fragment_id for f in outcome.original_fragments] == [
        "fragment_0",
        "fragment_1",
    ]
    assert outcome.optimized_fragments[0].content == "first"
    assert outcome.optimized_fragments[1].content == "second"


def test_mapping_fragments_are_parsed_correctly() -> None:
    mapping = {
        "id": "chunk_a",
        "text": "  parsed   mapping  ",
        "source_type": "retrieved_evidence",
        "required": True,
        "metadata": {"doc_id": "doc-1"},
    }

    outcome = optimize_context_pack([mapping], token_policy=_enabled_policy())

    fragment = outcome.original_fragments[0]
    assert fragment.fragment_id == "chunk_a"
    assert fragment.content == "  parsed   mapping  "
    assert fragment.source_type is TokenOptimizationSourceType.RETRIEVED_EVIDENCE
    assert fragment.required is True
    assert fragment.metadata == {"doc_id": "doc-1"}


def test_disabled_token_policy_bypasses_optimization() -> None:
    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=TokenOptimizationPolicy(enabled=False),
    )

    assert outcome.original_content == outcome.optimized_content
    assert outcome.changed is False
    assert outcome.status is ContextPackOptimizationStatus.BYPASSED
    assert outcome.result.decision is TokenOptimizationDecision.BYPASS
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.DISABLED


def test_resolved_output_policy_is_present_in_outcome() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.resolved_output_policy is not None
    assert outcome.resolved_output_policy.enabled is True
    assert outcome.resolved_output_policy.status is OutputPolicyResolutionStatus.RESOLVED


def test_protected_regions_in_fragments_are_preserved() -> None:
    prefix = "context " * 40
    protected_url = "https://example.com/protected/resource"
    content = f"{prefix}See {protected_url} for details and more usage notes."
    config = ContextPackOptimizationConfig(max_fragment_chars=120)

    outcome = optimize_context_pack(
        [ContextFragment(fragment_id="url_frag", content=content)],
        token_policy=_enabled_policy(),
        config=config,
    )

    optimized = outcome.optimized_fragments[0]
    assert protected_url in optimized.content
    assert outcome.metadata["fragments_preserved_due_to_protected_regions"] >= 1


def test_fragment_level_protected_region_failure_preserves_original_fragment() -> None:
    prefix = "noise " * 50
    protected_path = "/workspace/src/module.py"
    content = f"{prefix}Open {protected_path} for edits."
    config = ContextPackOptimizationConfig(max_fragment_chars=60)

    outcome = optimize_context_pack(
        [ContextFragment(fragment_id="path_frag", content=content)],
        token_policy=_enabled_policy(),
        config=config,
    )

    assert outcome.optimized_fragments[0].content == content


def test_global_protected_region_validation_failure_causes_fallback() -> None:
    failed_validation = ProtectedRegionValidationResult(
        status=ProtectedRegionValidationStatus.FAILED,
        regions_checked=1,
        regions_preserved=0,
        regions_failed=1,
        failures=("missing protected region",),
    )

    with patch(
        "intergrax.runtime.token_optimization.context_pack.validate_protected_regions",
        return_value=failed_validation,
    ):
        outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.optimized_content == outcome.original_content
    assert outcome.optimized_fragments == outcome.original_fragments
    assert outcome.result.fallback_used is True
    assert outcome.result.decision is TokenOptimizationDecision.FALLBACK
    assert outcome.result.bypass_reason is TokenOptimizationBypassReason.VALIDATION_FAILED
    assert outcome.status is ContextPackOptimizationStatus.FALLBACK


def test_receipt_created_for_applied_optimization() -> None:
    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=_enabled_policy(),
        config=ContextPackOptimizationConfig(include_receipt=True),
    )

    assert outcome.receipt is not None
    assert outcome.receipt.receipt_id.startswith("receipt_")


def test_receipt_created_for_bypass_when_include_receipt_true() -> None:
    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=TokenOptimizationPolicy(enabled=False),
        config=ContextPackOptimizationConfig(include_receipt=True),
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
        "intergrax.runtime.token_optimization.context_pack.validate_protected_regions",
        return_value=failed_validation,
    ):
        outcome = optimize_context_pack(
            _sample_fragments(),
            token_policy=_enabled_policy(),
            config=ContextPackOptimizationConfig(include_receipt=True),
        )

    assert outcome.receipt is not None
    assert outcome.result.decision is TokenOptimizationDecision.FALLBACK


def test_receipt_ref_maps_to_receipt() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.receipt_ref is not None
    assert outcome.receipt is not None
    assert outcome.receipt_ref.receipt_id == outcome.receipt.receipt_id
    assert outcome.receipt_ref.original_hash == outcome.receipt.original_hash
    assert outcome.receipt_ref.optimized_hash == outcome.receipt.optimized_hash


def test_measurement_none_when_token_counter_not_provided() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert outcome.result.measurement is None


def test_measurement_created_when_token_counter_provided() -> None:
    def counter(text: str) -> int:
        return len(text)

    outcome = optimize_context_pack(
        _sample_fragments(),
        token_policy=_enabled_policy(),
        token_counter=counter,
    )

    measurement = outcome.result.measurement
    assert measurement is not None
    assert measurement.confidence is TokenSavingsClaimConfidence.MEASURED
    assert measurement.saved_tokens == measurement.baseline_tokens - measurement.optimized_tokens


def test_metadata_contains_safe_counters_and_not_raw_full_context() -> None:
    outcome = optimize_context_pack(_sample_fragments(), token_policy=_enabled_policy())

    assert "mode" in outcome.metadata
    assert "fragment_count" in outcome.metadata
    assert "changed" in outcome.metadata
    assert outcome.metadata["fragment_count"] == 2
    for key in ("original_content", "optimized_content", "raw_context"):
        assert key not in outcome.metadata


def test_optimizer_does_not_reorder_fragments() -> None:
    fragments = [
        ContextFragment(fragment_id=f"frag_{index}", content=f"content {index}")
        for index in range(5)
    ]

    outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())

    assert len(outcome.optimized_fragments) == len(fragments)
    assert [f.fragment_id for f in outcome.optimized_fragments] == [
        f.fragment_id for f in fragments
    ]


def test_optimizer_does_not_merge_fragments() -> None:
    fragments = [
        ContextFragment(fragment_id="a", content="alpha"),
        ContextFragment(fragment_id="b", content="beta"),
    ]

    outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())
    parsed = json.loads(outcome.optimized_content)

    assert len(parsed["fragments"]) == 2


def test_optimizer_does_not_remove_fragments() -> None:
    fragments = _sample_fragments()

    outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())

    assert len(outcome.optimized_fragments) == len(fragments)
    parsed = json.loads(outcome.optimized_content)
    assert len(parsed["fragments"]) == len(fragments)


def test_default_policy_enables_safe_compaction_when_token_policy_is_none() -> None:
    outcome = optimize_context_pack(_sample_fragments())

    assert outcome.changed is True
    assert outcome.request.policy == DEFAULT_CONTEXT_PACK_TOKEN_POLICY


def test_context_pack_optimizer_class_wrapper_matches_helper() -> None:
    fragments = _sample_fragments()
    optimizer = ContextPackOptimizer()

    class_outcome = optimizer.optimize_pack(fragments, token_policy=_enabled_policy())
    helper_outcome = optimize_context_pack(fragments, token_policy=_enabled_policy())

    assert class_outcome.optimized_content == helper_outcome.optimized_content


def test_no_tokenizer_model_runtime_or_telemetry_imports_introduced() -> None:
    source = _CONTEXT_PACK_MODULE.read_text(encoding="utf-8")
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
