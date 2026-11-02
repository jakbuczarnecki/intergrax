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


def test_manifest_dag_has_no_cycle_on_default_manifest() -> None:
    from intergrax.platform.enterprise_audit.manifest import load_default_manifest

    topological_order(load_default_manifest())


def test_manifest_dag_rejects_cycle() -> None:
    a = _minimal("A", depends_on=frozenset({"B"}))
    b = _minimal("B", depends_on=frozenset({"A"}))
    with pytest.raises(ValueError, match="cycle"):
        topological_order((a, b))
