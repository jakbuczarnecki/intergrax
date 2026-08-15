from __future__ import annotations

from intergrax.runtime.token_optimization.llm_router import (
    _RISK_CONDITION_PROSE,
    _RISK_INVARIANTS,
    _RISK_LEVELS,
    _RiskCondition,
    _RiskInvariantId,
    _RiskLevelId,
    _RISK_INVARIANT_PROSE,
    _SYSTEM_PROMPT,
    _SYSTEM_PROMPT_PREFIX,
    _SYSTEM_PROMPT_SUFFIX,
    _render_risk_semantics,
)

_EXPECTED_RISK_LEVELS = (
    _RiskLevelId.LOW,
    _RiskLevelId.MEDIUM,
    _RiskLevelId.HIGH,
)
_EXPECTED_RISK_CONDITIONS = {
    _RiskLevelId.LOW: (
        _RiskCondition.LOSSLESS_OR_NO_OPTIMIZATION,
        _RiskCondition.NO_INDEPENDENT_CRITICAL_SIGNAL,
        _RiskCondition.NO_MATERIAL_INFORMATION_LOSS_RISK,
    ),
    _RiskLevelId.MEDIUM: (
        _RiskCondition.LOSSY_MAY_REMOVE_OMIT_OR_COMPRESS,
        _RiskCondition.NO_INDEPENDENT_CRITICAL_SIGNAL,
        _RiskCondition.AUTOMATIC_WITHOUT_MANDATORY_REVIEW,
    ),
    _RiskLevelId.HIGH: (
        _RiskCondition.EXPLICIT_CRITICAL_SIGNAL,
        _RiskCondition.LOSSY_MAY_TOUCH_PROTECTED_OR_CRITICAL_INFORMATION,
        _RiskCondition.LOSS_MAY_CHANGE_CRITICAL_MEANING,
        _RiskCondition.HUMAN_REVIEW_INDEPENDENTLY_REQUIRED,
    ),
}
_EXPECTED_RISK_INVARIANTS = (
    _RiskInvariantId.CLASSIFY_BEFORE_FINAL_POLICY_ENFORCEMENT,
    _RiskInvariantId.RISK_MEANS_INFORMATION_LOSS_DISTORTION_OR_OMISSION,
    _RiskInvariantId.HIGH_REQUIRES_REVIEW,
    _RiskInvariantId.LOSSY_NOT_ALWAYS_HIGH,
    _RiskInvariantId.PROTECTED_VALUES_NOT_AUTOMATICALLY_HIGH,
    _RiskInvariantId.LOSSLESS_EXACT_PRESERVATION_MAY_BE_LOW,
    _RiskInvariantId.ORDINARY_NONCRITICAL_LOSSY_EXTRACTION_IS_MEDIUM,
    _RiskInvariantId.SOURCE_TYPE_ALONE_DOES_NOT_LOWER_LOSSY_RISK,
)


def test_decision_prompt_freezes_structured_risk_contract() -> None:
    level_names = tuple(level.name for level in _RISK_LEVELS)
    conditions_by_level = {level.name: level.conditions for level in _RISK_LEVELS}
    invariant_keys = tuple(invariant.key for invariant in _RISK_INVARIANTS)
    rendered_risk = _render_risk_semantics()

    assert level_names == _EXPECTED_RISK_LEVELS
    assert len(level_names) == len(set(level_names))
    assert conditions_by_level == _EXPECTED_RISK_CONDITIONS
    assert invariant_keys == _EXPECTED_RISK_INVARIANTS
    assert len(invariant_keys) == len(set(invariant_keys))
    assert len(_RiskCondition.__members__) == len(
        {condition.value for condition in _RiskCondition.__members__.values()}
    )
    assert len(_RiskInvariantId.__members__) == len(
        {invariant.value for invariant in _RiskInvariantId.__members__.values()}
    )

    contract_conditions = frozenset(
        condition for level in _RISK_LEVELS for condition in level.conditions
    )
    assert contract_conditions == frozenset(_RiskCondition)
    assert frozenset(_RISK_INVARIANT_PROSE) == frozenset(_RiskInvariantId)
    assert frozenset(_RISK_CONDITION_PROSE) == frozenset(_RiskCondition)
    for condition in contract_conditions:
        assert _RISK_CONDITION_PROSE[condition].strip()
    for invariant in _RISK_INVARIANTS:
        assert _RISK_INVARIANT_PROSE[invariant.key].strip()

    assert rendered_risk.count("Risk levels:\n") == 1
    assert rendered_risk.count("Frozen risk invariants:\n") == 1

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
