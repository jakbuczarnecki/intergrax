# © Artur Czarnecki. All rights reserved.

"""Deterministic orchestrator expansion cycle diagnostics."""

from __future__ import annotations

import pytest

from testing_support.execution_qualification.catalog.expansion import (
    expand_mandatory_subprocesses,
)
from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationDependencyCycleError,
)

_PATH_A = "tests/unit/cycle/a.py"
_PATH_B = "tests/unit/cycle/b.py"
_PATH_C = "tests/unit/cycle/c.py"


def test_three_node_expansion_cycle_path_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.execution_qualification.catalog import (
        expansion as expansion_mod,
    )
    from testing_support.execution_qualification.catalog import (
        orchestrators as orch_mod,
    )

    cycle_source: FrozenPytestSuiteSource = (("root", [_PATH_A]),)
    mapping = {
        _PATH_A: (("to_b", [_PATH_B]),),
        _PATH_B: (("to_c", [_PATH_C]),),
        _PATH_C: (("to_a", [_PATH_A]),),
    }
    monkeypatch.setattr(
        orch_mod,
        "CANONICAL_ORCHESTRATOR_PATHS",
        frozenset({_PATH_A, _PATH_B, _PATH_C}),
    )
    monkeypatch.setattr(
        expansion_mod,
        "CANONICAL_ORCHESTRATOR_PATHS",
        frozenset({_PATH_A, _PATH_B, _PATH_C}),
    )

    def _fake_orchestrator_expansion(targets: list[str]):
        if len(targets) != 1:
            return None
        return mapping.get(targets[0])

    monkeypatch.setattr(
        expansion_mod, "_orchestrator_expansion", _fake_orchestrator_expansion
    )

    with pytest.raises(QualificationDependencyCycleError) as exc:
        expand_mandatory_subprocesses(cycle_source)
    assert exc.value.cycle_path == (_PATH_A, _PATH_B, _PATH_C, _PATH_A)
