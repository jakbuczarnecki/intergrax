# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.critic.contracts import CriticLayer, LayerVerdict
from intergrax.runtime.critic.guardrail_l0 import merge_guardrail_l0

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_merge_guardrail_l0_fails_when_scan_blocked() -> None:
    verdict = LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=True, score=1.0)
    merged = merge_guardrail_l0(
        verdict,
        context={"guardrail_scan": {"allowed": False, "detail": "output blocked"}},
    )
    assert merged.passed is False
    assert any("guardrail_l0" in error for error in merged.errors)


def test_merge_guardrail_l0_adds_category_warnings() -> None:
    verdict = LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=True, score=1.0)
    merged = merge_guardrail_l0(
        verdict,
        context={"guardrail_scan": {"allowed": True, "categories": ["pii_email"]}},
    )
    assert merged.passed is True
    assert merged.warnings
