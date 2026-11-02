from __future__ import annotations

import pytest
from intergrax.platform.enterprise_audit.contracts import AuditCategory, AuditMode, AuditSeverity, ManifestCheck
from intergrax.platform.enterprise_audit.schedule import topological_order


def _minimal(check_id: str, *, depends_on: frozenset[str] = frozenset()) -> ManifestCheck:
    return ManifestCheck(
        check_id=check_id,
        entrypoint_kind="pytest_nodeid",
        entrypoint="tests/qualification/ent_audit_x/unit/test_manifest_and_exits.py::test_manifest_has_eighty_five_checks",
        modes=frozenset({AuditMode.FAST}),
        severity_default=AuditSeverity.ERROR,
        category=AuditCategory.ARCHITECTURE,
        depends_on=depends_on,
    )


def test_manifest_dag_rejects_unknown_depends_on() -> None:
    a = _minimal("A", depends_on=frozenset({"MISSING-ID"}))
    with pytest.raises(ValueError, match="unknown check_id"):
        topological_order((a,))
