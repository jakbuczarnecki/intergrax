# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.core.qualification import QualificationStatus, qualification_status_satisfies

pytestmark = pytest.mark.unit


def test_qualification_status_ordering() -> None:
    assert qualification_status_satisfies(
        QualificationStatus.PRODUCTION_QUALIFIED,
        QualificationStatus.QUALIFIED,
    )
    assert qualification_status_satisfies(
        QualificationStatus.QUALIFIED,
        QualificationStatus.QUALIFIED,
    )
    assert not qualification_status_satisfies(
        QualificationStatus.NOT_QUALIFIED,
        QualificationStatus.QUALIFIED,
    )


def test_rejected_does_not_satisfy_qualified() -> None:
    assert not qualification_status_satisfies(
        QualificationStatus.REJECTED,
        QualificationStatus.QUALIFIED,
    )
    assert not qualification_status_satisfies(
        QualificationStatus.REJECTED,
        QualificationStatus.PRODUCTION_QUALIFIED,
    )
