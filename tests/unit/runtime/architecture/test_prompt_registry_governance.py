from __future__ import annotations

from intergrax.runtime.architecture.prompt_registry_governance import (
    PromptRegistryEntry,
    PromptRiskTier,
    evaluate_prompt_registry,
    validate_prompt_registry_entry,
)


def test_prompt_registry_requires_change_ticket_for_high_risk() -> None:
    result = validate_prompt_registry_entry(
        PromptRegistryEntry(
            prompt_id="prompt.legal",
            version="1.0.0",
            owner_team="legal",
            owner_contact="owner@intergrax",
            risk_tier=PromptRiskTier.HIGH,
        )
    )
    assert result.valid is False
    assert any("change ticket" in reason.lower() for reason in result.reasons)


def test_prompt_registry_passes_complete_entry() -> None:
    report = evaluate_prompt_registry(
        [
            PromptRegistryEntry(
                prompt_id="prompt.echo",
                version="1.0.0",
                owner_team="platform",
                owner_contact="owner@intergrax",
                risk_tier=PromptRiskTier.LOW,
            )
        ]
    )
    assert report.validations[0].valid is True
