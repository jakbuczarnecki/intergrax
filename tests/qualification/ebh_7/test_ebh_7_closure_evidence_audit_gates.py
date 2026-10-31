# © Artur Czarnecki. All rights reserved.

"""G01–G09 closure evidence audit gates — 0 OPEN_FINDING before independent B6 handoff."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._ebh7_closure_audit_markdown_sync import assert_closure_audit_markdown_matches_ssot
from tests.qualification.ebh_7._ebh7_closure_evidence_audit import (
    assert_ebh7_closure_evidence_audit_ready,
    closure_open_findings,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_closure_evidence_audit_zero_open_findings() -> None:
    assert_ebh7_closure_evidence_audit_ready()


def test_ebh7_closure_evidence_audit_markdown_matches_ssot() -> None:
    assert_closure_audit_markdown_matches_ssot()


def test_ebh7_closure_evidence_audit_open_register_empty() -> None:
    assert closure_open_findings() == ()
