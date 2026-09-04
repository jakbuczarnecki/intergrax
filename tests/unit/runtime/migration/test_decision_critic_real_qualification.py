# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.migration.critic_retirement_qualification import (
    CriticRetirementEvidenceProvenance,
    proven_critic_retirement_qualification,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    ParityHostScope,
    ParityVerificationCapability,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATH = Path("intergrax/runtime/migration/critic_retirement_qualification.py")
_FORBIDDEN_FRAGMENTS = (
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect.",
    "exec(",
    "eval(",
    "dict[str, Any]",
)


def test_retirement_qualification_certificate_is_ready_with_provenance() -> None:
    qualification = proven_critic_retirement_qualification()
    assert qualification.readiness is CriticRetirementReadiness.READY
    assert qualification.provenance is (
        CriticRetirementEvidenceProvenance.HISTORICAL_PRE_RETIREMENT_QUALIFICATION
    )
    assert qualification.parity_qualification_commit
    assert qualification.ds_mig_03_hitl_transition_commit
    assert qualification.final_regression_gate_commit
    assert qualification.report.blocking_mismatch_count == 0
    assert qualification.report.shadow_error_count == 0
    assert ParityHostScope.GRAPH_FINAL in qualification.qualified_scopes
    assert ParityHostScope.UAEP_STEP in qualification.qualified_scopes
    assert ParityVerificationCapability.SEMANTIC in qualification.qualified_capabilities
    assert ParityVerificationCapability.HUMAN_HITL in qualification.qualified_capabilities


def test_forbidden_audit_retirement_qualification_module() -> None:
    source = _MODULE_PATH.read_text(encoding="utf-8")
    for fragment in _FORBIDDEN_FRAGMENTS:
        assert fragment not in source
