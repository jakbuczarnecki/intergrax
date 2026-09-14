# © Artur Czarnecki. All rights reserved.

"""Orchestrator test-module paths and nested mandatory sources (catalog SSOT)."""

from __future__ import annotations

from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)

NPSC5E_FINAL_ORCHESTRATOR_PATH = "tests/unit/runtime/architecture/test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
NPSC5E_R3_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
)
NPSC5E_R2_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"
)

# Legacy parity expansion recognizes only these three orchestrator single-file targets.
LEGACY_ORCHESTRATOR_EXPANSION_PATHS: frozenset[str] = frozenset(
    {
        NPSC5E_FINAL_ORCHESTRATOR_PATH,
        NPSC5E_R3_FINAL_ORCHESTRATOR_PATH,
        NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
    },
)

# Paths that must not appear as compiled DAG leaves (broader guard set).
NESTED_PYTEST_ORCHESTRATOR_LEAF_PATHS: frozenset[str] = frozenset(
    LEGACY_ORCHESTRATOR_EXPANSION_PATHS
)


def orchestrator_mandatory_lookup() -> dict[str, FrozenPytestSuiteSource]:
    """Lazy import of mandatory sources to avoid circular imports at module load."""
    from testing_support.execution_qualification.catalog.mandatory_sources import (
        NPSC5E_FINAL_MANDATORY,
        NPSC5E_R2_FINAL_MANDATORY,
        NPSC5E_R3_FINAL_MANDATORY,
    )

    return {
        NPSC5E_FINAL_ORCHESTRATOR_PATH: NPSC5E_FINAL_MANDATORY,
        NPSC5E_R3_FINAL_ORCHESTRATOR_PATH: NPSC5E_R3_FINAL_MANDATORY,
        NPSC5E_R2_FINAL_ORCHESTRATOR_PATH: NPSC5E_R2_FINAL_MANDATORY,
    }
