# © Artur Czarnecki. All rights reserved.

"""Unit tests for functional qualification verdict aggregation."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_qualification_verdict import (
    QualificationVerdict,
    aggregate_qualification_verdicts,
    aggregate_plugin_verdict,
)

pytestmark = pytest.mark.unit


def test_all_pass_yields_pass() -> None:
    assert aggregate_qualification_verdicts((QualificationVerdict.PASS, QualificationVerdict.PASS)) is QualificationVerdict.PASS


def test_one_failed_yields_failed() -> None:
    assert aggregate_qualification_verdicts((QualificationVerdict.PASS, QualificationVerdict.FAILED)) is QualificationVerdict.FAILED


def test_blocked_without_failed_yields_blocked() -> None:
    assert aggregate_qualification_verdicts((QualificationVerdict.PASS, QualificationVerdict.BLOCKED)) is QualificationVerdict.BLOCKED


def test_failed_precedence_over_blocked() -> None:
    assert aggregate_qualification_verdicts(
        (QualificationVerdict.FAILED, QualificationVerdict.BLOCKED),
    ) is QualificationVerdict.FAILED


def test_aggregate_plugin_verdict_blocked() -> None:
    assert aggregate_plugin_verdict(domain_verdict=QualificationVerdict.BLOCKED, gate_failures=0) is QualificationVerdict.BLOCKED


def test_aggregate_plugin_verdict_gate_failure() -> None:
    assert aggregate_plugin_verdict(domain_verdict=QualificationVerdict.PASS, gate_failures=1) is QualificationVerdict.FAILED
