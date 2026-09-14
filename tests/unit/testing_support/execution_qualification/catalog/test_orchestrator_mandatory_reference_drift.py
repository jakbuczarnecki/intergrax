# © Artur Czarnecki. All rights reserved.

"""Reference-only legacy _MANDATORY_SUITES drift guards vs canonical catalog."""

from __future__ import annotations

from testing_support.execution_qualification.catalog.mandatory_sources import (
    NPSC5F_R1_FINAL_MANDATORY,
    NPSC5F_R2_FINAL_MANDATORY,
    NPSC5F_R3_FINAL_MANDATORY,
)


def test_npsc5f_r1_final_orchestrator_reference_matches_canonical() -> None:
    from tests.unit.runtime.architecture import (
        test_npsc5f_r1_final_durable_evidence_commit_tenant_integrity as mod,
    )

    assert mod._MANDATORY_SUITES == NPSC5F_R1_FINAL_MANDATORY


def test_npsc5f_r2_final_orchestrator_reference_matches_canonical() -> None:
    from tests.unit.runtime.architecture import (
        test_npsc5f_r2_final_journal_completeness_ordering as mod,
    )

    assert mod._MANDATORY_SUITES == NPSC5F_R2_FINAL_MANDATORY


def test_npsc5f_r3_final_orchestrator_reference_matches_canonical() -> None:
    from tests.unit.runtime.architecture import (
        test_npsc5f_r3_final_governed_evidence_export as mod,
    )

    assert mod._MANDATORY_SUITES == NPSC5F_R3_FINAL_MANDATORY
