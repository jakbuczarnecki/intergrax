from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from intergrax.runtime.token_optimization.proofs.config import (
    load_universal_token_optimization_proof_config,
)
from intergrax.runtime.token_optimization.proofs.corpus import load_proof_corpus
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    EvaluationConfigurationError,
    expand_proof_config_with_corpus,
)

_ROOT = Path(__file__).resolve().parents[5]
_CORPUS = _ROOT / "configs/token_optimization/corpus/universal_proof_cases.toml"


def test_checked_in_corpus_is_strict_and_covers_required_categories() -> None:
    corpus = load_proof_corpus(_CORPUS)

    assert corpus.schema_version == "token-optimization-proof-corpus.v1"
    assert len(corpus.cases) == 16
    assert {
        "short_clean_prompt",
        "exact_duplicate_content",
        "noisy_tool_output",
        "terminal_log_output",
        "rag_context_pack",
        "protected_url_path_hash_error",
        "code_heavy_content",
        "high_risk_lossy_content",
        "policy_disabled",
        "measure_only",
        "prefix_stable_repeat",
        "changed_dynamic_tail",
        "tool_payload_order",
        "canonical_inner_payload_order",
        "warm_cache_evidence",
        "changed_prefix_negative_control",
    } == {case.category for case in corpus.cases}
    assert all(
        case.router and case.pipeline and case.protected for case in corpus.cases
    )
    assert all(
        "SYNTHETIC_SECRET_MARKER" not in case.description for case in corpus.cases
    )


def test_corpus_inputs_are_representative_and_expectations_are_assertive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "test-only")
    corpus = load_proof_corpus(_CORPUS)
    proof_config = load_universal_token_optimization_proof_config(
        _ROOT / "configs/token_optimization/proof_vllm.toml"
    )
    expanded = expand_proof_config_with_corpus(proof_config, corpus)
    cases = {case.case_id: case for case in expanded.cases}
    expectations = {case.case_id: case for case in corpus.cases}

    assert len({case.input_case_id for case in corpus.cases}) == 16
    assert all(case.input_case_id != "smoke-exact-dedup" for case in corpus.cases)
    assert cases["case-short-clean-prompt"].request.content == (
        "Return current deployment status."
    )
    assert cases["case-exact-duplicate-content"].request.content.splitlines().count(
        "line-a"
    ) == 2
    assert len(cases["case-noisy-tool-output"].request.content.splitlines()) >= 50
    assert "[031] execute error=E_SYNTHETIC_TERMINAL" in cases[
        "case-terminal-log-output"
    ].request.content
    assert cases["case-noisy-tool-output"].request.source_type.value == "tool_output"
    assert (
        cases["case-terminal-log-output"].request.source_type.value
        == "terminal_output"
    )
    assert cases["case-rag-context-pack"].request.source_type.value == "rag_context_pack"
    assert cases["case-code-heavy-content"].request.source_type.value == "structured_data"
    assert cases["case-reordered-tools"].request.source_type.value == "tool_catalog"
    assert cases["case-rag-context-pack"].request.content.count("[evidence-001]") == 2
    assert "def verify_artifact" in cases["case-code-heavy-content"].request.content
    assert "sha256" in cases["case-code-heavy-content"].request.content
    assert len(expectations["case-protected-values"].protected_regions) == 4
    protected_case = cases["case-protected-values"]
    protected_values = tuple(
        region.value for region in expectations["case-protected-values"].protected_regions
    )
    assert len(protected_values) == 4
    assert set(protected_values) == {
        "https://synthetic.invalid/proof/run",
        "synthetic/artifacts/fixed-run.json",
        "SYNTHETIC_RUN_001",
        "E_SYNTHETIC_PROTECTED",
    }
    assert all(protected_case.request.content.count(value) == 1 for value in protected_values)
    assert sum(
        protected_case.request.content.count(value) for value in protected_values
    ) == 4
    assert len(expectations["case-high-risk-lossy-content"].protected_regions) == 2
    assert cases["case-policy-disabled"].request.policy.enabled is False
    assert cases["case-measure-only"].request.policy.profile.value == "measure_only"

    assert sum(
        bool(
            case.router.allowed_configuration_ids
            and case.router.allowed_reason_codes
        )
        for case in corpus.cases
    ) >= 6
    assert sum(
        bool(
            case.pipeline.required_layer_ids
            or case.pipeline.forbidden_layer_ids
            or case.pipeline.expected_completion is not None
        )
        for case in corpus.cases
    ) >= 5
    assert sum(
        case.measurement.baseline.value == "required"
        and case.measurement.optimized.value == "required"
        and case.measurement.ordering_required
        for case in corpus.cases
    ) >= 3
    assert sum(case.prefix.identity_required for case in corpus.cases) >= 2
    assert sum(case.cache.mode.value != "not_applicable" for case in corpus.cases) >= 2
    assert all(
        case.router.allowed_statuses != frozenset({"routed", "blocked"})
        for case in corpus.cases
    )


def test_full_run_corpus_leaves_identity_controls_to_evaluate_only_fixtures() -> None:
    corpus = load_proof_corpus(_CORPUS)
    cases = {case.case_id: case for case in corpus.cases}

    assert cases["case-prefix-stable-repeat"].prefix.same_as_case_id == (
        "case-exact-duplicate-content"
    )
    assert cases["case-prefix-changed"].category == "changed_dynamic_tail"
    assert cases["case-prefix-changed"].prefix.identity_required is False

    for case_id in (
        "case-reordered-tools",
        "case-inner-dictionary-order",
        "case-warm-cache",
        "case-changed-prefix-cache-control",
    ):
        assert cases[case_id].prefix.identity_required is False
        assert cases[case_id].prefix.tool_schema_identity is None


def test_corpus_rejects_unknown_fields_duplicate_ids_and_unsafe_ids(
    tmp_path: Path,
) -> None:
    base = _CORPUS.read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(
        base.replace("\n[[cases]]", "\nunknown = true\n\n[[cases]]", 1),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationConfigurationError, match="UNKNOWN_CORPUS_FIELD"):
        load_proof_corpus(unknown)

    duplicate = tmp_path / "duplicate.toml"
    duplicate.write_text(
        base.replace(
            'case_id = "case-exact-duplicate-content"',
            'case_id = "case-short-clean-prompt"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationConfigurationError, match="INVALID_CORPUS"):
        load_proof_corpus(duplicate)

    unsafe = tmp_path / "unsafe.toml"
    unsafe.write_text(
        base.replace(
            'case_id = "case-short-clean-prompt"',
            'case_id = "../unsafe"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationConfigurationError, match="INVALID_CASE_ID"):
        load_proof_corpus(unsafe)


def test_corpus_contracts_are_immutable() -> None:
    case = load_proof_corpus(_CORPUS).cases[0]
    with pytest.raises(FrozenInstanceError):
        case.case_id = "changed"
