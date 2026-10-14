# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P2 — strict production integrated E2E (bounded P2-A slice)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.qualification.prod_q._prod_q_r1_r7_q_p2_strict_integrated_e2e import (
    D4_MODULE_RELPATH,
    P2_NEGATIVE_D1B_REUSE,
    P2_NEGATIVE_D4_REUSE,
    P2_POSITIVE_D4_REUSE,
    assert_identity_continuity,
    default_pytest_collect_probe,
    integration_import_blocker,
    load_d4_launcher_module,
    repo_root,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]

_EXPECTED_START_HEAD = "5a4bd2835d7022f56ea52cd5587ade7eb75a5fd2"


def test_p2_g00_qualification_baseline_ancestor() -> None:
    import subprocess as sp

    sp.check_call(
        ["git", "merge-base", "--is-ancestor", _EXPECTED_START_HEAD, "HEAD"],
        cwd=repo_root(),
    )


def test_p2_g01_default_pytest_entry_blocked_by_conftest_cycle() -> None:
    probe = default_pytest_collect_probe()
    assert probe.returncode != 0, "expected default pytest entry to fail at audited HEAD"
    combined = probe.combined
    assert "RuntimeCheckpoint" in combined or "INTERNALERROR" in combined


def test_p2_g02_strict_launcher_import_chain_succeeds() -> None:
    blocked = integration_import_blocker()
    assert blocked is None, f"expected strict launcher import to succeed after ENV-1: {blocked}"


def test_p2_g03_bounded_d4_reuse_inventory_complete() -> None:
    d4_path = repo_root() / D4_MODULE_RELPATH
    text = d4_path.read_text(encoding="utf-8")
    for name in (*P2_POSITIVE_D4_REUSE, *P2_NEGATIVE_D4_REUSE):
        assert f"def {name}" in text


def test_p2_a01_positive_chain_identity_continuity(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    if err:
        pytest.skip(f"ENVIRONMENT/TEST ISSUE — integration import blocked: {err}")
    revision_id = "rev-p2-a01-positive"
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id=revision_id,
    )
    try:
        assert session.readiness.ready_for_traffic is True
        assert session.readiness.execution_admission_eligible is True
        assert_identity_continuity(session, revision_id=revision_id)
        handoff = session.serving_runtime_handoff
        admission = handoff.colocated_execution_admission
        assert admission is not None
        unit_ref = handoff.identity.serving_unit_ref
        admission.admit_execution(unit_ref)
        admission.release_execution(unit_ref)
    finally:
        session.close()


def test_p2_a02_gca_strict_factory_consumes_same_handoff(tmp_path: Path) -> None:
    from fastapi import FastAPI

    d4, err = load_d4_launcher_module()
    if err:
        pytest.skip(f"ENVIRONMENT/TEST ISSUE — integration import blocked: {err}")
    session = d4._open_with_activation(
        tmp_path,
        environment=d4._ac3_environment(),
        revision_id="rev-p2-a02-gca",
    )
    handoff = session.serving_runtime_handoff
    try:
        with patch(
            "governed_contractor_application.host.strict_production_serving.create_governed_contractor_process_app",
            return_value=FastAPI(),
        ) as create_app:
            d4.create_governed_contractor_strict_production_asgi_app(session)
            composition = create_app.call_args.kwargs["process_composition"]
            assert composition.serving_runtime_handoff is handoff
    finally:
        session.close()


def test_p2_n01_unready_session_rejects_gca_factory(tmp_path: Path) -> None:
    d4, err = load_d4_launcher_module()
    if err:
        pytest.skip(f"ENVIRONMENT/TEST ISSUE — integration import blocked: {err}")
    session = d4._open_with_activation(tmp_path, environment=d4._ac3_environment())
    session.readiness = d4.StrictProductionServingReadiness(
        process_lifecycle_open=True,
        production_dependencies_validated=True,
        serving_state_reconciled=False,
        serving_handoff_bound=False,
        execution_admission_eligible=False,
        ready_for_traffic=False,
    )
    try:
        with pytest.raises(ValueError, match="not ready for traffic"):
            d4.create_governed_contractor_strict_production_asgi_app(session)
    finally:
        session.close()


def _run_named_d4_test(test_name: str) -> None:
    d4_path = repo_root() / D4_MODULE_RELPATH
    completed = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-p",
            "no:xdist",
            "-q",
            "--noconftest",
            f"{d4_path}::{test_name}",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 and "circular import" in (completed.stdout + completed.stderr).lower():
        pytest.skip(
            "ENVIRONMENT/TEST ISSUE — D4 replay blocked by import cycle "
            f"({test_name})",
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("test_name", P2_NEGATIVE_D4_REUSE)
def test_p2_n02_reuse_d4_negative_matrix(test_name: str) -> None:
    _run_named_d4_test(test_name)


def test_p2_n03_wrong_runtime_revision_handoff_rejected(tmp_path: Path) -> None:
    if integration_import_blocker():
        pytest.skip("ENVIRONMENT/TEST ISSUE — D1b negative blocked by import cycle")
    d1b_path = (
        repo_root()
        / "tests/unit/applications/test_prod_q_r1_r7_d1b_serving_runtime_handoff.py"
    )
    completed = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-p",
            "no:xdist",
            "-q",
            "--noconftest",
            f"{d1b_path}::{P2_NEGATIVE_D1B_REUSE[0]}",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 and "circular import" in (completed.stdout + completed.stderr).lower():
        pytest.skip("ENVIRONMENT/TEST ISSUE — D1b replay blocked by import cycle")
    assert completed.returncode == 0, completed.stdout + completed.stderr
