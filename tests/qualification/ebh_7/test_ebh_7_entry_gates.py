# © Artur Czarnecki. All rights reserved.

"""EBH-7 W0 entry gates."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.ebh_7._constants import EBH7_PLANNING_BASELINE_SHA
from tests.qualification.ebh_7._task_register import assert_ebh7_task_register_integrity
from tests.qualification.ebh_7._traceability_matrix import assert_ebh7_traceability_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO = Path(__file__).resolve().parents[3]

_W0_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/EBH_7_C0_BASELINE_AUDIT_RECORD.md",
    "docs/project/maintainers/qualification/EBH_7_SHA_LADDER.md",
    "docs/project/maintainers/qualification/EBH_7_OPERATOR_DECISION_REGISTER.md",
)


def test_ebh7_entry_traceability_466() -> None:
    assert_ebh7_traceability_integrity()


def test_ebh7_entry_task_register_92() -> None:
    assert_ebh7_task_register_integrity()


def test_ebh7_entry_w0_docs_present() -> None:
    for rel in _W0_DOCS:
        assert (_REPO / rel).is_file(), rel


def test_ebh7_entry_parent_replay_closed_docs() -> None:
    assert (
        _REPO / "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_OD_REPLAYX_08_CLOSURE_SIGNATURE_RECORD.md"
    ).is_file()


def test_ebh7_entry_planning_baseline_documented() -> None:
    text = (_REPO / "docs/project/maintainers/qualification/EBH_7_C0_BASELINE_AUDIT_RECORD.md").read_text(
        encoding="utf-8"
    )
    assert EBH7_PLANNING_BASELINE_SHA in text
