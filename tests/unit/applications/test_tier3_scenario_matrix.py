# © Artur Czarnecki. All rights reserved.

"""APP-CON-7 — Tier-3 scenario matrix gate and UC-A* minimum per reference host."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.tier3_scenario_matrix_wiring import (
    REFERENCE_HOST_SCENARIO_MATRIX,
    SCENARIO_CATALOG,
    Tier3ScenarioId,
    check_reference_host_scenario_matrix,
    check_scenario_catalog_complete,
    iter_reference_host_profiles,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.tier3_scenario]

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("scenario_id", list(Tier3ScenarioId))
def test_scenario_catalog_has_evidence_files(scenario_id: Tier3ScenarioId) -> None:
    definition = SCENARIO_CATALOG[scenario_id]
    for rel_path in definition.evidence_paths:
        assert (REPO / rel_path).is_file(), f"{scenario_id}: missing evidence {rel_path}"


@pytest.mark.parametrize("package", sorted(REFERENCE_HOST_SCENARIO_MATRIX))
def test_reference_host_package_exists(package: str) -> None:
    root = REPO / "applications" / package
    assert root.is_dir(), f"missing applications/{package}"
    assert (root / "manifest.py").is_file()


@pytest.mark.parametrize("package", sorted(REFERENCE_HOST_SCENARIO_MATRIX))
def test_reference_host_required_scenarios_registered(package: str) -> None:
    profile = REFERENCE_HOST_SCENARIO_MATRIX[package]
    assert profile.required_scenarios, f"{package}: must declare required scenarios"
    for scenario_id in profile.required_scenarios:
        assert scenario_id in SCENARIO_CATALOG


def test_scenario_catalog_complete() -> None:
    assert check_scenario_catalog_complete() == []


def test_reference_host_matrix_gate_passes() -> None:
    assert check_reference_host_scenario_matrix(REPO) == []


def test_all_reference_hosts_enumerated() -> None:
    expected = {
        "poc_template_application",
        "lab_application",
        "legal_application",
        "research_application",
        "local_workspace_application",
        "dispute_sim_application",
        "intergrax_assistant_application",
    }
    assert {profile.package for profile in iter_reference_host_profiles()} == expected


@pytest.mark.parametrize(
    ("scenario_id", "evidence_suffix"),
    [
        (Tier3ScenarioId.EVAL_CI_HARNESS, "test_lab_strict_harness.py"),
        (Tier3ScenarioId.YAML_DRIVEN_LAB, "test_lab_manifest_wiring.py"),
    ],
)
def test_lab_eval_scenarios_have_evidence(
    scenario_id: Tier3ScenarioId,
    evidence_suffix: str,
) -> None:
    lab_profile = REFERENCE_HOST_SCENARIO_MATRIX["lab_application"]
    assert scenario_id in lab_profile.required_scenarios
    for rel_path in SCENARIO_CATALOG[scenario_id].evidence_paths:
        assert rel_path.endswith(evidence_suffix)
        assert (REPO / rel_path).is_file()
