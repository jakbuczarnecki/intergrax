# © Artur Czarnecki. All rights reserved.

"""Host environment evidence for performance certification (no private identifiers)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

from testing_support.execution_qualification.performance.models import (
    QualificationPerformanceEnvironmentEvidence,
)


def _uv_version() -> str | None:
    uv_path = shutil.which("uv")
    if uv_path is None:
        return None
    completed = subprocess.run(
        [uv_path, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def collect_performance_environment_evidence(
    *,
    max_parallel: int,
) -> QualificationPerformanceEnvironmentEvidence:
    logical = os.cpu_count()
    if logical is None or logical < 1:
        logical = 1
    return QualificationPerformanceEnvironmentEvidence(
        python_version=sys.version.split()[0],
        uv_version=_uv_version(),
        platform_system=platform.system(),
        cpu_logical_count=logical,
        max_parallel=max_parallel,
    )
