# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext, BehaviorViolationLedger


@pytest.fixture
def violation_ledger() -> BehaviorViolationLedger:
    return BehaviorViolationLedger()


@pytest.fixture
def behavior_eval_context() -> BehaviorEvalContext:
    return BehaviorEvalContext()
