# © Artur Czarnecki. All rights reserved.

import pytest

from testing_support.agent_contract_test_inventory import (
    find_concrete_agent_subclasses_missing_run_runtime,
    find_harness_reference_subclasses_missing_uaep_ast,
    find_direct_agent_subclasses_missing_run_ast,
)

pytestmark = pytest.mark.unit


def test_no_direct_agent_subclass_without_run_in_test_support_ast() -> None:
    violations = find_direct_agent_subclasses_missing_run_ast()
    assert not violations, (
        "Direct Agent subclasses in tests/support must implement run() "
        "or inherit HarnessReferenceAgent/IntergraxAgent. Found: "
        + ", ".join(f"{path}:{name}" for path, name in violations)
    )


def test_no_instantiable_agent_missing_run_at_runtime() -> None:
    violations = find_concrete_agent_subclasses_missing_run_runtime()
    assert not violations, (
        "Concrete Agent subclasses must satisfy Agent.run contract. Found: "
        + ", ".join(violations)
    )


def test_no_harness_reference_subclass_missing_uaep_methods_ast() -> None:
    violations = find_harness_reference_subclasses_missing_uaep_ast()
    assert not violations, (
        "HarnessReferenceAgent test doubles must implement get_steps/run_step. Found: "
        + ", ".join(f"{path}:{name}" for path, name in violations)
    )
