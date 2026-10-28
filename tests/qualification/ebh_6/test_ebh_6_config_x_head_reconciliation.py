# © Artur Czarnecki. All rights reserved.

"""CONFIG-X 54/54 reconciliation @ current HEAD for EBH-6."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.config_x._config_x_concern_inventory import CONFIG_X_CONCERN_INVENTORY
from tests.qualification.ebh_5._config_x_reconciliation import CONFIG_X_CONCERN_COUNT, CONFIG_X_HISTORICAL_PASS_SHA
from tests.qualification.ebh_6._config_x_reconciliation import (
    EBH6_CONFIG_X_RECONCILIATION_NOTE,
    assert_ebh6_config_x_fully_reconciled,
    config_x_head_drift_summary,
    config_x_reconciliation_rows,
)
from tests.qualification.ebh_6._evidence_manifest import EBH6_PUBLICATION_SHA, evidence_git_head_sha

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUDIT_MD = _REPO_ROOT / "docs/project/maintainers/qualification/EBH_6_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md"


def test_ebh6_config_x_all_concerns_mapped_at_head() -> None:
    assert_ebh6_config_x_fully_reconciled()


def test_ebh6_config_x_row_count_54() -> None:
    assert len(config_x_reconciliation_rows()) == CONFIG_X_CONCERN_COUNT
    assert len(CONFIG_X_CONCERN_INVENTORY) == CONFIG_X_CONCERN_COUNT


def test_ebh6_config_x_historical_pass_preserved() -> None:
    summary = config_x_head_drift_summary()
    assert summary.historical_pass_sha == CONFIG_X_HISTORICAL_PASS_SHA
    assert summary.mapped_concern_count == 54


def test_ebh6_config_x_audit_doc_records_reconciliation() -> None:
    text = _AUDIT_MD.read_text(encoding="utf-8")
    assert CONFIG_X_HISTORICAL_PASS_SHA[:12] in text
    head_sha = evidence_git_head_sha()
    assert head_sha in text or EBH6_PUBLICATION_SHA in text
    assert EBH6_CONFIG_X_RECONCILIATION_NOTE.split()[0] in text
