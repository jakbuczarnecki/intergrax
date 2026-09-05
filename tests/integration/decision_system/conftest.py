# © Artur Czarnecki. All rights reserved.

"""Shared fixtures for Decision System DS-E2E qualification."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from testing_support.decision_e2e.composition import (
    QualificationComposition,
    build_qualification_composition,
    build_sqlite_persistence,
)
from testing_support.decision_e2e.environment import (
    qualification_required,
    qualification_strict_required,
    resolve_qualification_environment,
)
from testing_support.decision_e2e.reporting import QualificationReportCollector

_OUTPUT_DIR = Path(".tmp/decision_e2e_qualification")


@pytest.fixture(scope="session")
def decision_e2e_report_collector() -> QualificationReportCollector:
    return QualificationReportCollector()


@pytest.fixture(scope="session")
def decision_e2e_environment():
    environment, block_reason = resolve_qualification_environment()
    return environment, block_reason


@pytest.fixture
def require_decision_e2e_environment(decision_e2e_environment):
    environment, block_reason = decision_e2e_environment
    if environment is None:
        if qualification_strict_required():
            pytest.fail(block_reason or "decision e2e environment blocked")
        pytest.skip(block_reason or "decision e2e environment unavailable")
    return environment


@pytest.fixture
def decision_e2e_composition(
    require_decision_e2e_environment,
) -> QualificationComposition:
    return build_qualification_composition(require_decision_e2e_environment)


@pytest.fixture
def decision_e2e_sqlite_composition(
    require_decision_e2e_environment,
    tmp_path: Path,
) -> QualificationComposition:
    persistence = build_sqlite_persistence(tmp_path / "durable")
    return build_qualification_composition(
        require_decision_e2e_environment,
        persistence=persistence,
    )


@pytest.fixture(scope="session", autouse=True)
def _write_decision_e2e_report(
    decision_e2e_report_collector: QualificationReportCollector,
) -> Generator[None, None, None]:
    yield
    if not qualification_required():
        return
    from testing_support.decision_e2e.reporting import write_qualification_artifacts

    profile = "decision_e2e_qualification"
    report = decision_e2e_report_collector.build_report(environment_profile=profile)
    write_qualification_artifacts(report, output_dir=_OUTPUT_DIR)


def record_qualification_result(
    collector: QualificationReportCollector,
    result,
) -> None:
    collector.record(result)
