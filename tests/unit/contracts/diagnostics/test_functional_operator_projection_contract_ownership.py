# © Artur Czarnecki. All rights reserved.

"""Unit gates for functional operator projection contract ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection,
    FunctionalOperatorOutcomeStatus,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection as RuntimeProjection,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorOutcomeStatus as RuntimeOutcomeStatus,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalOperatorProjector,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT = (
    _REPO_ROOT
    / "intergrax"
    / "contracts"
    / "diagnostics"
    / "functional_operator_projection.py"
)


def test_canonical_operator_projection_lives_in_contracts() -> None:
    assert (
        FunctionalDiagnosticOperatorProjection.__module__
        == "intergrax.contracts.diagnostics.functional_operator_projection"
    )
    assert (
        FunctionalOperatorOutcomeStatus.__module__
        == "intergrax.contracts.diagnostics.functional_operator_projection"
    )


def test_runtime_path_is_compatibility_alias_not_second_definition() -> None:
    assert RuntimeProjection is FunctionalDiagnosticOperatorProjection
    assert RuntimeOutcomeStatus is FunctionalOperatorOutcomeStatus


def test_contract_module_has_no_projector_implementation() -> None:
    tree = ast.parse(_CONTRACT.read_text(encoding="utf-8"), filename=str(_CONTRACT))
    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    assert "FunctionalOperatorProjector" not in class_names
    assert "FunctionalDiagnosticOperatorProjection" in class_names


def test_projector_is_runtime_only() -> None:
    assert FunctionalOperatorProjector.__module__ == (
        "intergrax.runtime.diagnostics.functional_operator_projection"
    )
