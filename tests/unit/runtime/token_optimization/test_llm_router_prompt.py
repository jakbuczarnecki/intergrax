from __future__ import annotations

from intergrax.runtime.token_optimization.llm_router import (
    _RISK_INVARIANTS,
    _RISK_LEVELS,
    _SYSTEM_PROMPT,
    _SYSTEM_PROMPT_PREFIX,
    _SYSTEM_PROMPT_SUFFIX,
    _render_risk_semantics,
)

_EXPECTED_RISK_LEVELS = ("low", "medium", "high")
_EXPECTED_RISK_INVARIANTS = (
    "classification_before_final_policy_enforcement",
    "risk_is_information_loss",
    "high_requires_review",
    "lossy_is_not_always_high",
    "protected_values_alone_are_not_high",
    "lossless_exact_preservation_is_low",
    "ordinary_lossy_extractive_filtering_is_medium",
    "source_type_does_not_reduce_lossy_risk",
)


def test_decision_prompt_freezes_structured_risk_contract() -> None:
    level_names = tuple(level.name for level in _RISK_LEVELS)
    invariant_keys = tuple(invariant.key for invariant in _RISK_INVARIANTS)
    rendered_risk = _render_risk_semantics()

    assert level_names == _EXPECTED_RISK_LEVELS
    assert len(level_names) == len(set(level_names))
    assert all(level.definition.strip() for level in _RISK_LEVELS)
    assert invariant_keys == _EXPECTED_RISK_INVARIANTS
    assert len(invariant_keys) == len(set(invariant_keys))
    assert all(invariant.statement.strip() for invariant in _RISK_INVARIANTS)

    for level in _RISK_LEVELS:
        assert rendered_risk.count(f"- {level.name}:") == 1
        assert f"- {level.name}: {level.definition}" in rendered_risk
    for invariant in _RISK_INVARIANTS:
        assert rendered_risk.count(f"- {invariant.statement}") == 1

    assert _render_risk_semantics() == rendered_risk
    assert _SYSTEM_PROMPT.count(rendered_risk) == 1
    assert _SYSTEM_PROMPT == (
        _SYSTEM_PROMPT_PREFIX + rendered_risk + _SYSTEM_PROMPT_SUFFIX
    )

    for field in (
        "configuration_id",
        "reason_code",
        "risk",
        "review_required",
        "confidence",
    ):
        assert field in _SYSTEM_PROMPT
