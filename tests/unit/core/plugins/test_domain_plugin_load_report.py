# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import EntryPointLoadResult, EntryPointSpec
from intergrax.core.plugins.errors import PluginLoadError

pytestmark = pytest.mark.unit


def _spec(name: str) -> EntryPointSpec:
    return EntryPointSpec(
        name=name,
        group="intergrax.security_defenses",
        value=f"pkg:{name}",
        distribution=None,
    )


def test_domain_plugin_load_report_is_immutable() -> None:
    report = DomainPluginLoadReport.empty("intergrax.security_defenses")
    with pytest.raises(FrozenInstanceError):
        report.registered_count = 1  # type: ignore[misc]


def test_domain_plugin_load_report_deterministic_and_counts() -> None:
    accepted = (_spec("b"), _spec("a"))
    rejected = (
        PluginAdmissionRejection(
            spec=_spec("c"),
            reason_code=PluginAdmissionReasonCode.PLUGIN_ID_COLLISION,
            reason="collision",
            plugin_id="dup",
            fail_closed=True,
        ),
    )
    failed = (
        EntryPointLoadResult(spec=_spec("d"), error=PluginLoadError("boom")),
    )
    report = DomainPluginLoadReport(
        group="intergrax.security_defenses",
        accepted=accepted,
        rejected=rejected,
        failed=failed,
        registered_count=2,
    )
    assert report.registered_count == 2
    assert [item.name for item in report.accepted] == ["b", "a"]
    assert report.rejected[0].reason_code is PluginAdmissionReasonCode.PLUGIN_ID_COLLISION
    assert report.failed[0].error is not None
    assert report.critical_bootstrap_acceptable is False


def test_domain_plugin_load_report_audit_has_no_target_objects() -> None:
    report = DomainPluginLoadReport(
        group="intergrax.security_defenses",
        accepted=(_spec("ok"),),
        rejected=(
            PluginAdmissionRejection(
                spec=_spec("bad-type"),
                reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                reason="not a defense",
                fail_closed=True,
            ),
        ),
        failed=(EntryPointLoadResult(spec=_spec("broken"), error=PluginLoadError("load")),),
        registered_count=1,
    )
    payload = report.to_audit_dict()
    assert payload["group"] == "intergrax.security_defenses"
    assert payload["registered_count"] == 1
    accepted_rows = payload["accepted"]
    rejected_rows = payload["rejected"]
    failed_rows = payload["failed"]
    assert isinstance(accepted_rows, tuple)
    assert isinstance(rejected_rows, tuple)
    assert isinstance(failed_rows, tuple)
    for row in (*accepted_rows, *rejected_rows, *failed_rows):
        assert isinstance(row, dict)
        assert "target" not in row
    assert report.accepted[0].name == "ok"
