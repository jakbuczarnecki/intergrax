# © Artur Czarnecki. All rights reserved.

"""Canonical qualification orchestrator paths and expansion mapping (catalog SSOT)."""

from __future__ import annotations

from collections.abc import Mapping

from testing_support.execution_qualification.frozen_pytest_adapter import (
    FrozenPytestSuiteSource,
)

NPSC5E_R2_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r2_final_checkpoint_durable_resume_qualification.py"
)
NPSC5E_R3_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5e_r3_final_child_fanout_partial_recovery_qualification.py"
)
NPSC5E_R3_IMPLEMENTATION_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/test_npsc5e_r3_child_fanout_partial_recovery.py"
)
NPSC5E_R2_H2_Q1_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py"
)
NPSC5E_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5e_final_recovery_plane_qualification_and_freeze.py"
)

NPSC5F_R1_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity.py"
)
NPSC5F_R2_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/"
    "test_npsc5f_r2_final_journal_completeness_ordering.py"
)
NPSC5F_R3_FINAL_ORCHESTRATOR_PATH = (
    "tests/unit/runtime/architecture/test_npsc5f_r3_final_governed_evidence_export.py"
)

CANONICAL_ORCHESTRATOR_PATHS: frozenset[str] = frozenset(
    {
        NPSC5E_R2_FINAL_ORCHESTRATOR_PATH,
        NPSC5E_R3_FINAL_ORCHESTRATOR_PATH,
        NPSC5E_FINAL_ORCHESTRATOR_PATH,
        NPSC5F_R1_FINAL_ORCHESTRATOR_PATH,
        NPSC5F_R2_FINAL_ORCHESTRATOR_PATH,
        NPSC5F_R3_FINAL_ORCHESTRATOR_PATH,
    },
)

# Alias for leaf guards (same set as expansion orchestrators).
NESTED_PYTEST_ORCHESTRATOR_LEAF_PATHS: frozenset[str] = CANONICAL_ORCHESTRATOR_PATHS

LEGACY_ORCHESTRATOR_EXPANSION_PATHS: frozenset[str] = CANONICAL_ORCHESTRATOR_PATHS


def orchestrator_expansion_mapping() -> Mapping[str, FrozenPytestSuiteSource]:
    """Orchestrator module path → mandatory source to expand (explicit, no reflection)."""
    from testing_support.execution_qualification.catalog.mandatory_sources import (
        NPSC5E_FINAL_MANDATORY,
        NPSC5E_R2_FINAL_MANDATORY,
        NPSC5E_R3_FINAL_MANDATORY,
        NPSC5F_R1_FINAL_MANDATORY,
        NPSC5F_R2_FINAL_MANDATORY,
        NPSC5F_R3_FINAL_MANDATORY,
    )

    return {
        NPSC5E_R2_FINAL_ORCHESTRATOR_PATH: NPSC5E_R2_FINAL_MANDATORY,
        NPSC5E_R3_FINAL_ORCHESTRATOR_PATH: NPSC5E_R3_FINAL_MANDATORY,
        NPSC5E_FINAL_ORCHESTRATOR_PATH: NPSC5E_FINAL_MANDATORY,
        NPSC5F_R1_FINAL_ORCHESTRATOR_PATH: NPSC5F_R1_FINAL_MANDATORY,
        NPSC5F_R2_FINAL_ORCHESTRATOR_PATH: NPSC5F_R2_FINAL_MANDATORY,
        NPSC5F_R3_FINAL_ORCHESTRATOR_PATH: NPSC5F_R3_FINAL_MANDATORY,
    }


def orchestrator_mandatory_lookup() -> dict[str, FrozenPytestSuiteSource]:
    """Backward-compatible name for expansion mapping."""
    return dict(orchestrator_expansion_mapping())
