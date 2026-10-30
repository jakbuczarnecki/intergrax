# © Artur Czarnecki. All rights reserved.

"""EBH-7 corrective audit — findings register and traceability behavioral integrity."""

from __future__ import annotations

import pytest
from tests.qualification.ebh_7._gate_findings_register import assert_ebh7_corrective_findings_registered
from tests.qualification.ebh_7._traceability_matrix import assert_ebh7_traceability_behavioral_integrity

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_ebh7_corrective_findings_register() -> None:
    assert_ebh7_corrective_findings_registered()


def test_ebh7_traceability_behavioral_integrity() -> None:
    assert_ebh7_traceability_behavioral_integrity()
