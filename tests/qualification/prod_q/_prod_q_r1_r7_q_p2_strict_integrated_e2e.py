# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-R7-Q-P2 — bounded strict production integrated E2E helpers."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_REPO_ROOT = Path(__file__).resolve().parents[3]

STRICT_LAUNCHER_IMPORT_PROBE: Final[str] = (
    "from intergrax.applications._shared.strict_production_serving_launcher "
    "import open_strict_production_serving_session"
)

D4_MODULE_RELPATH: Final[str] = (
    "tests/unit/applications/governed_contractor/"
    "test_prod_q_r1_r7_d4_strict_production_gca_launcher.py"
)

P2_POSITIVE_D4_REUSE: Final[tuple[str, ...]] = (
    "test_d4_active_state_served_after_reconciliation",
    "test_d4_gca_factory_consumes_validated_handoff",
)

P2_NEGATIVE_D4_REUSE: Final[tuple[str, ...]] = (
    "test_d4_asgi_startup_only_after_preconditions",
    "test_d4_stale_physical_unit_after_serving",
    "test_d4_governance_deny_blocks_activation",
    "test_d4_fake_adapter_rejected",
    "test_d4_missing_serving_pointer_after_open",
    "test_d4_21_activation_intent_scope_mismatch_before_lifecycle",
)

P2_NEGATIVE_D1B_REUSE: Final[tuple[str, ...]] = (
    "test_handoff_rejects_resolver_instance_mismatch",
)


def repo_root() -> Path:
    return _REPO_ROOT


@dataclass(frozen=True, slots=True)
class SubprocessProbeResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}".strip()


def run_python_snippet(snippet: str, *, cwd: Path | None = None) -> SubprocessProbeResult:
    completed = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=cwd or repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return SubprocessProbeResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def strict_launcher_import_probe() -> SubprocessProbeResult:
    return run_python_snippet(STRICT_LAUNCHER_IMPORT_PROBE)


_INTEGRATION_BLOCKER: str | None = None


def integration_import_blocker() -> str | None:
    """Subprocess probe — avoids in-process partial import cycles during pytest."""
    global _INTEGRATION_BLOCKER
    if _INTEGRATION_BLOCKER is None:
        probe = strict_launcher_import_probe()
        _INTEGRATION_BLOCKER = probe.combined if probe.returncode != 0 else ""
    return _INTEGRATION_BLOCKER or None


def default_pytest_collect_probe() -> SubprocessProbeResult:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:xdist",
            "tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_strict_production_integrated_e2e.py",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return SubprocessProbeResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def load_d4_launcher_module():
    """Import D4 fixture module; return (module, error_text)."""
    blocked = integration_import_blocker()
    if blocked:
        return None, blocked
    from tests.unit.applications.governed_contractor import (
        test_prod_q_r1_r7_d4_strict_production_gca_launcher as d4_mod,
    )

    return d4_mod, None


def assert_identity_continuity(session, *, revision_id: str) -> None:
    """Application / environment / revision / serving unit / resolver continuity."""
    env = session.environment
    handoff = session.serving_runtime_handoff
    assert session.composition.serving_runtime_handoff is handoff
    identity = handoff.identity
    assert identity.application_id == env.application_id
    assert identity.application_environment_id == env.application_environment_id
    assert identity.runtime_revision_id == revision_id
    assert identity.serving_unit_ref
    assert handoff.runtime_agent_factory_resolver is not None
