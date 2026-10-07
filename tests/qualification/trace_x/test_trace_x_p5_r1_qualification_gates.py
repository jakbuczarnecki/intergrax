# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1 mechanical qualification gates."""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from tests.qualification.trace_x._trace_x_p5_r1_support import (
    TRACE_X_P5_R1_START_HEAD,
    discover_profile_aware_environment_host_roots,
    reconstruction_profile_fallback_sentinels,
    reconstruction_runtime_import_violations,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp5r1_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P5_R1_START_HEAD, "HEAD"],
    )


def test_txp5r1_q02_reconstruction_no_application_profile_imports() -> None:
    violations = reconstruction_runtime_import_violations()
    assert not violations, f"forbidden reconstruction imports: {violations}"


def test_txp5r1_q03_reconstruction_no_profile_fallback_sentinels() -> None:
    hits = reconstruction_profile_fallback_sentinels()
    assert not hits, f"forbidden profile fallback patterns: {hits}"


def test_txp5r1_q04_environment_host_revision_admission_mandatory() -> None:
    signature = inspect.signature(build_environment_host_task_execution)
    param = signature.parameters["revision_admission"]
    assert param.default is inspect.Parameter.empty


def test_txp5r1_q05_profile_aware_roots_wire_revision_admission() -> None:
    missing = discover_profile_aware_environment_host_roots()
    assert not missing, f"profile-aware roots missing revision_admission: {missing}"
