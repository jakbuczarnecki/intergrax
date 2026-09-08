"""Unit tests for bounded semantic representation layer."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog import (
    build_source_record_ref,
    derive_bounded_search_representation,
    derive_search_representation,
    derive_search_representation_with_policy,
)
from platform_proofs.scenarios.verified_product_identification.application.domain import (
    ProductOfferId,
    SourceRecordRef,
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation import (
    CharacterRatioTokenEstimator,
    RepresentationPolicyProfile,
    SemanticRepresentationBuilder,
    bounded_policy_compact,
    bounded_policy_v1,
    build_representation_reduction_metrics,
    parse_policy_json,
    parse_reduction_metrics_json,
    resolve_semantic_representation_policy,
    run_all_representation_policy_experiments,
    serialize_policy_json,
    serialize_reduction_metrics_json,
    serialize_result_json,
    token_limit_policy_v1,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SEMANTIC_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/semantic_representation"
)
_CATALOG_DERIVE = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/application/catalog/derive_search_representation.py"
)
_CATALOG_ID = "catalog-semantic"
_SOURCE_REVISION = "rev-semantic"


class _DeterministicTokenEstimator:
    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 10) if text else 0


def _source_ref(offer_id: str) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id=_CATALOG_ID,
        source_revision=_SOURCE_REVISION,
    )


def _offer_from_record(record: dict[str, object]):
    source_offer = parse_wdc_source_offer_json(json.dumps(record, ensure_ascii=False))
    source_ref = build_source_record_ref(
        source_offer,
        catalog_id=_CATALOG_ID,
        source_revision=_SOURCE_REVISION,
    )
    return source_offer, source_ref


def _electronics_record() -> dict[str, object]:
    return {
        "id": 5001,
        "title": "Neutral NVMe Storage Device 2TB",
        "description": "High-speed internal storage for general-purpose systems.",
        "brand": "NeutralBrand",
        "identifiers": [
            {"/gtin13": "[8806095123456]"},
            {"/mpn": "[MZ-V9P2T0BW]"},
        ],
        "keyValuePairs": {
            "Capacity": "2TB",
            "Interface": "NVMe",
            "Color": "Black",
        },
        "specTableContent": "Capacity 2TB\nInterface NVMe",
    }


def test_identity_fields_always_preserved() -> None:
    record = _electronics_record()
    source_offer, source_ref = _offer_from_record(record)
    derived = derive_bounded_search_representation(
        source_offer,
        source_ref=source_ref,
        policy=bounded_policy_compact(),
    )
    semantic_text = derived.semantic.semantic_text
    assert "NeutralBrand" in semantic_text
    assert "MZ-V9P2T0BW" in semantic_text
    assert "8806095123456" in semantic_text
    assert "Neutral NVMe Storage Device 2TB" in semantic_text


def test_large_description_truncation() -> None:
    long_description = "Sentence start. " + ("detail " * 5000)
    record = {
        "id": "desc-heavy",
        "title": "Compact Widget",
        "brand": "WidgetCo",
        "description": long_description,
    }
    source_offer, source_ref = _offer_from_record(record)
    legacy = derive_search_representation(source_offer, source_ref=source_ref)
    bounded = derive_bounded_search_representation(
        source_offer,
        source_ref=source_ref,
        policy=bounded_policy_compact(),
    )
    assert len(bounded.semantic.semantic_text) < len(legacy.semantic.semantic_text)
    assert bounded.semantic.semantic_text.startswith("brand: WidgetCo")
    assert "Compact Widget" in bounded.semantic.semantic_text


def test_large_spec_table_truncation() -> None:
    long_specs = "Width: 10\n" * 4000
    record = {
        "id": "spec-heavy",
        "title": "Industrial Panel",
        "brand": "PanelCorp",
        "specTableContent": long_specs,
    }
    source_offer, source_ref = _offer_from_record(record)
    legacy = derive_search_representation(source_offer, source_ref=source_ref)
    bounded = derive_bounded_search_representation(
        source_offer,
        source_ref=source_ref,
        policy=bounded_policy_compact(),
    )
    assert len(bounded.semantic.semantic_text) < len(legacy.semantic.semantic_text)
    assert "Industrial Panel" in bounded.semantic.semantic_text
    assert "PanelCorp" in bounded.semantic.semantic_text


def test_unicode_safety() -> None:
    record = {
        "id": "unicode-offer",
        "title": "Żółć gęślą jaźń — produkt testowy",
        "brand": "Märke™",
        "description": "Opis z emoji 🚀 i znakiem ∑.",
        "keyValuePairs": {"Kolor": "czarny"},
    }
    source_offer, source_ref = _offer_from_record(record)
    bounded = derive_bounded_search_representation(source_offer, source_ref=source_ref)
    text = bounded.semantic.semantic_text
    assert "Żółć" in text
    assert "Märke" in text
    assert "🚀" in text


def test_empty_fields() -> None:
    record = {"id": "empty-offer", "title": "Only Title"}
    source_offer, source_ref = _offer_from_record(record)
    bounded = derive_bounded_search_representation(source_offer, source_ref=source_ref)
    assert bounded.semantic.semantic_text == "Only Title"


def test_null_fields() -> None:
    record = {
        "id": "null-offer",
        "title": "Nullable Offer",
        "description": None,
        "brand": None,
        "specTableContent": None,
        "keyValuePairs": {},
    }
    source_offer, source_ref = _offer_from_record(record)
    bounded = derive_bounded_search_representation(source_offer, source_ref=source_ref)
    assert bounded.semantic.semantic_text == "Nullable Offer"


def test_deterministic_output() -> None:
    record = _electronics_record()
    source_offer, source_ref = _offer_from_record(record)
    first = derive_bounded_search_representation(source_offer, source_ref=source_ref)
    second = derive_bounded_search_representation(source_offer, source_ref=source_ref)
    assert first.semantic.semantic_text == second.semantic.semantic_text
    assert first == second


def test_token_budget_enforcement() -> None:
    record = {
        "id": "token-budget",
        "title": "Token Budget Device",
        "brand": "BudgetBrand",
        "description": "word " * 2000,
        "specTableContent": "spec " * 2000,
    }
    source_offer, source_ref = _offer_from_record(record)
    build_output = SemanticRepresentationBuilder(
        token_limit_policy_v1(),
        token_estimator=_DeterministicTokenEstimator(),
    ).build(source_offer, source_ref=source_ref)
    assert build_output.result.estimated_tokens <= 2000


def test_policy_switching() -> None:
    record = _electronics_record()
    source_offer, source_ref = _offer_from_record(record)
    legacy = derive_search_representation_with_policy(
        source_offer,
        source_ref=source_ref,
        representation_policy_profile=RepresentationPolicyProfile.FULL_V1,
    )
    bounded = derive_search_representation_with_policy(
        source_offer,
        source_ref=source_ref,
        representation_policy_profile=RepresentationPolicyProfile.BOUND_8000,
    )
    assert legacy.semantic.semantic_text == derive_search_representation(
        source_offer,
        source_ref=source_ref,
    ).semantic.semantic_text
    assert bounded.semantic.semantic_text
    assert legacy.exact == bounded.exact
    assert legacy.lexical == bounded.lexical
    assert legacy.structured == bounded.structured


def test_architecture_conformance_no_forbidden_contract_patterns() -> None:
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "Mapping[str, object]",
    )
    core_modules = (
        _SEMANTIC_ROOT / "contracts.py",
        _SEMANTIC_ROOT / "policy.py",
        _SEMANTIC_ROOT / "builder.py",
        _SEMANTIC_ROOT / "metrics.py",
        _SEMANTIC_ROOT / "ports.py",
        _SEMANTIC_ROOT / "serialization.py",
    )
    for module_path in core_modules:
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {module_path.name}"


def test_no_forbidden_imports_in_core_modules() -> None:
    forbidden_roots = (
        "psycopg",
        "qdrant",
        "docker",
        "sentence_transformers",
        "transformers",
    )
    core_modules = sorted(_SEMANTIC_ROOT.glob("*.py"))
    for module_path in core_modules:
        if module_path.name == "__init__.py":
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name)
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        violations = sorted(
            name for name in imported if any(root in name for root in forbidden_roots)
        )
        assert violations == [], f"{module_path.name} imports forbidden roots: {violations}"


def test_serialization_roundtrip() -> None:
    policy = bounded_policy_v1()
    policy_roundtrip = parse_policy_json(serialize_policy_json(policy))
    assert policy_roundtrip == policy

    record = _electronics_record()
    source_offer, source_ref = _offer_from_record(record)
    build_output = SemanticRepresentationBuilder(policy).build(source_offer, source_ref=source_ref)
    serialized_result = serialize_result_json(build_output.result)
    assert "representation_text" in serialized_result

    metrics = build_representation_reduction_metrics(
        before_chars=1000,
        after_chars=250,
        before_tokens=250,
        after_tokens=63,
    )
    metrics_roundtrip = parse_reduction_metrics_json(serialize_reduction_metrics_json(metrics))
    assert metrics_roundtrip == metrics


def test_experiment_framework_emits_reduction_metrics() -> None:
    record = {
        "id": "experiment-heavy",
        "title": "Heavy Experiment Device",
        "brand": "HeavyBrand",
        "description": "detail " * 3000,
        "specTableContent": "spec-line\n" * 1500,
        "keyValuePairs": {f"Attr{i}": f"value-{i}" for i in range(80)},
    }
    source_offer, source_ref = _offer_from_record(record)
    legacy = derive_search_representation(source_offer, source_ref=source_ref)
    experiments = run_all_representation_policy_experiments(
        source_offer,
        source_ref=source_ref,
        token_estimator=CharacterRatioTokenEstimator(),
        legacy_semantic_text=legacy.semantic.semantic_text,
        profiles=(
            RepresentationPolicyProfile.FULL_V1,
            RepresentationPolicyProfile.BOUND_4000,
        ),
    )
    assert len(experiments) == 2
    bounded = next(
        item for item in experiments if item.profile == RepresentationPolicyProfile.BOUND_4000
    )
    assert bounded.reduction.after_chars < bounded.reduction.before_chars
    assert bounded.reduction.reduction_ratio > 0.0


def test_resolve_semantic_representation_policy_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unsupported semantic representation policy"):
        resolve_semantic_representation_policy("unknown-profile")


def test_legacy_derivation_unchanged_when_full_v1_selected() -> None:
    record = _electronics_record()
    source_offer, source_ref = _offer_from_record(record)
    assert resolve_semantic_representation_policy(RepresentationPolicyProfile.FULL_V1) is None
    selected = derive_search_representation_with_policy(
        source_offer,
        source_ref=source_ref,
        representation_policy_profile=RepresentationPolicyProfile.FULL_V1,
    )
    baseline = derive_search_representation(source_offer, source_ref=source_ref)
    assert selected == baseline


def test_catalog_derive_module_has_no_forbidden_provider_imports() -> None:
    forbidden_tokens = ("postgres", "qdrant", "sentence_transformers", "transformers")
    tree = ast.parse(_CATALOG_DERIVE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(token in alias.name.casefold() for token in forbidden_tokens)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not any(token in node.module.casefold() for token in forbidden_tokens)
