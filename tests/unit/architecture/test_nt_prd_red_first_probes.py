"""Red-first NT-PRD probes naming the FRZ guard in assertion text (no ENT semantic gates)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def test_nt_prd_05_entrypoint_uses_hosting_lifecycle_engine() -> None:
    """NT-PRD-05: Tier-3 serve command must route through hosted strict production application."""
    cmd = (
        _REPO
        / "applications"
        / "governed_contractor_application"
        / "host"
        / "package_b_strict_production_serving_command.py"
    )
    text = cmd.read_text(encoding="utf-8")
    assert "run_governed_contractor_hosted_strict_production_application" in text, (
        "NT-PRD-05 missing hosting lifecycle entry: run_governed_contractor_hosted_strict_production_application"
    )
