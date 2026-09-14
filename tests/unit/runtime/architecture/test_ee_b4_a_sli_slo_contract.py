# © Artur Czarnecki. All rights reserved.

"""EE-B4-A — SLI catalog and SLO contract ownership."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_operational_readiness.sli_catalog import (
    EXECUTION_ENGINE_SLI_CATALOG,
    SloTargetOwner,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_OPERATIONAL_READINESS_SLO_HEALTH_MODEL.md"
)

_REQUIRED_SLI_IDS = {
    "execution_success_rate",
    "execution_failure_rate",
    "admission_reject_defer_rate",
    "execution_latency",
    "admission_wait_latency",
    "capacity_utilization",
    "worker_failure_rate",
    "dependency_failure_rate",
    "mandatory_evidence_persistence_failure_rate",
    "recovery_success_rate",
    "shutdown_drain_duration",
}


def test_ee_b4_a_sli_catalog_complete() -> None:
    catalog_ids = {item.sli_id for item in EXECUTION_ENGINE_SLI_CATALOG}
    assert _REQUIRED_SLI_IDS <= catalog_ids
    for item in EXECUTION_ENGINE_SLI_CATALOG:
        assert item.numerator
        assert item.denominator
        assert item.measurement_window
        assert item.fact_owner is SloTargetOwner.EXECUTION_ENGINE_FACTS


def test_ee_b4_a_slo_targets_not_hardcoded_in_model_doc() -> None:
    text = _MODEL.read_text(encoding="utf-8")
    assert "99.999" not in text
    assert (
        "deployment/operator configurable" in text.lower()
        or "deployment_operator" in text
    )
