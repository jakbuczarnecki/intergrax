from __future__ import annotations

from intergrax.runtime.token_optimization.llm_router import _SYSTEM_PROMPT


def test_decision_prompt_freezes_production_risk_semantics() -> None:
    prompt = " ".join(_SYSTEM_PROMPT.split())

    assert "before final deterministic policy enforcement" in prompt
    assert "- low:" in prompt
    assert "- medium:" in prompt
    assert "- high:" in prompt
    assert "High requires review_required=true." in prompt
    assert "Protected values alone do not make risk high" in prompt
    assert (
        "ordinary lossy extractive filtering is medium regardless of source_type"
        in prompt
    )
    assert "Not every lossy operation is high." in prompt
