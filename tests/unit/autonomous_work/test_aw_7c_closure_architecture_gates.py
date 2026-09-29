# © Artur Czarnecki. All rights reserved.

"""AW-7C-CLOSURE static architecture gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_AW7C_QUAL = _REPO / "intergrax/integrations/qualification"
_FORBIDDEN_IN_QUAL = (
    "mint_run_id",
    "mint_attempt_id",
    "mint_execution_id",
    "expected_operation",
    "ScopedAdaptiveIntegrationReferenceExecutionIntake",
    "Callable[[ExecutionId], CredentialUseGrant]",
)


@pytest.mark.parametrize("name", sorted(p.name for p in _AW7C_QUAL.glob("scoped_adaptive*.py")))
def test_closure_qualification_modules_forbidden_patterns(name: str) -> None:
    text = (_AW7C_QUAL / name).read_text(encoding="utf-8")
    for token in _FORBIDDEN_IN_QUAL:
        assert token not in text, f"{name} must not contain {token}"


def test_closure_intake_is_composition_not_alternate_intake_owner() -> None:
    text = (_AW7C_QUAL / "scoped_adaptive_integration_execution_intake.py").read_text(
        encoding="utf-8",
    )
    assert "class ScopedAdaptiveIntegrationReferenceExecutionIntake" not in text
    assert "build_scoped_adaptive_integration_canonical_execution_intake" in text
    assert "CanonicalExecutionRuntimeAdapter" in text
