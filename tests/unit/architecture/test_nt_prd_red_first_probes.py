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


def test_nt_prd_03_unsupported_queue_backend_must_not_fall_back_to_eager_celery() -> None:
    """NT-PRD-03 / FRZ-PRD-03: unsupported production configuration fails closed."""
    text = (_REPO / "intergrax/applications/_shared/queue_worker_wiring.py").read_text(encoding="utf-8")
    assert "fall back to eager Celery" not in text, (
        "NT-PRD-03 unsupported production configuration fails closed: "
        "Kafka/RabbitMQ queue backends must not be rewritten to in-process eager Celery"
    )
    assert "UnsupportedQueueBackendError" in text, (
        "NT-PRD-03 unsupported production configuration fails closed: "
        "queue worker wiring must raise UnsupportedQueueBackendError for unsupported broker backends"
    )
