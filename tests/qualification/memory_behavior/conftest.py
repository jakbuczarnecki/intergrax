# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorViolationLedger


@pytest.fixture
def violation_ledger() -> BehaviorViolationLedger:
    return BehaviorViolationLedger()
