# © Artur Czarnecki. All rights reserved.

"""CONFIG-X historical reconciliation vs EBH-6 current HEAD (extends EBH-5 mapping)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_5._config_x_reconciliation import (
    CONFIG_X_CONCERN_COUNT,
    CONFIG_X_HISTORICAL_PASS_SHA,
    ConfigXReconciliationRow,
    assert_config_x_coverage_complete,
    config_x_reconciliation_rows,
)
from tests.qualification.ebh_6._evidence_manifest import evidence_git_head_sha

EBH6_CONFIG_X_RECONCILIATION_NOTE: Final[str] = (
    "Historical FRZ-CFG PASS preserved @ fac9a43f…; EBH-6 records current-HEAD structural "
    "obligations via EBH-5 VBN inventory + CORE_HARDCODING_FINDINGS_REGISTER — not PASS upgrade."
)


@dataclass(frozen=True, slots=True)
class ConfigXHeadDriftSummary:
    historical_pass_sha: str
    current_head_sha: str
    concern_count: int
    mapped_concern_count: int


def config_x_head_drift_summary() -> ConfigXHeadDriftSummary:
    rows = config_x_reconciliation_rows()
    return ConfigXHeadDriftSummary(
        historical_pass_sha=CONFIG_X_HISTORICAL_PASS_SHA,
        current_head_sha=evidence_git_head_sha(),
        concern_count=CONFIG_X_CONCERN_COUNT,
        mapped_concern_count=len(rows),
    )


def assert_ebh6_config_x_fully_reconciled() -> None:
    summary = config_x_head_drift_summary()
    assert summary.concern_count == CONFIG_X_CONCERN_COUNT
    assert summary.mapped_concern_count == CONFIG_X_CONCERN_COUNT
    from tests.qualification.ebh_5._vendor_backend_inventory import VENDOR_BACKEND_DEPENDENCY_INVENTORY

    lineage = frozenset(
        row.concern_id for row in VENDOR_BACKEND_DEPENDENCY_INVENTORY if row.config_x_concern_id is not None
    )
    assert_config_x_coverage_complete(lineage)


__all__ = [
    "CONFIG_X_CONCERN_COUNT",
    "CONFIG_X_HISTORICAL_PASS_SHA",
    "ConfigXHeadDriftSummary",
    "ConfigXReconciliationRow",
    "EBH6_CONFIG_X_RECONCILIATION_NOTE",
    "assert_config_x_coverage_complete",
    "assert_ebh6_config_x_fully_reconciled",
    "config_x_head_drift_summary",
    "config_x_reconciliation_rows",
]
