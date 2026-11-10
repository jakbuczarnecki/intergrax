"""FRZ-AUD-03 Annex B negative: unmapped check fails fail-closed FRZ coverage rule."""

from __future__ import annotations

import pytest
from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditMode,
    AuditSeverity,
    ManifestCheck,
)


def _require_frz_or_sys_inv_mapping(check: ManifestCheck) -> None:
    """Same invariant as ``test_every_check_maps_to_frz_or_sys_inv``, exposed for negative tests."""
    if check.pg_id or check.check_id.startswith(("TEN-", "VBN-")):
        return
    raise ValueError(f"unmapped FRZ/invariant coverage for check_id={check.check_id!r}")


def test_ent_audit_unmapped_frz_neg() -> None:
    synthetic = ManifestCheck(
        check_id="ENT-A-SYNTH-UNMAPPED-FRZ",
        entrypoint_kind="pytest_nodeid",
        entrypoint="tests/qualification/ent_audit_x/unit/test_frz_aud_03_unmapped_neg.py::test_ent_audit_unmapped_frz_neg",
        category=AuditCategory.ARCHITECTURE,
        severity_default=AuditSeverity.ERROR,
        modes=frozenset({AuditMode.FULL, AuditMode.FREEZE}),
        pg_id="",
    )
    with pytest.raises(ValueError, match="unmapped FRZ"):
        _require_frz_or_sys_inv_mapping(synthetic)
