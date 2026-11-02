from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import AuditMode
from intergrax.platform.enterprise_audit.modes import default_limitations


def test_limitations_list_prod_q_live_open() -> None:
    limits = default_limitations(AuditMode.FULL)
    assert any("FRZ-PRD" in item or "Path B" in item for item in limits)
