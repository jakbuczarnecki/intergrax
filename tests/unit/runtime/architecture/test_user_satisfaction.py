# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.architecture.online_evaluation_registry import InMemoryOnlineEvaluationRegistry
from intergrax.runtime.architecture.user_satisfaction import (
    SatisfactionSignal,
    UserSatisfactionEvent,
    record_user_satisfaction,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_record_user_satisfaction_writes_online_registry() -> None:
    registry = InMemoryOnlineEvaluationRegistry()
    event = UserSatisfactionEvent(
        tenant_id="t1",
        task_id="task1",
        run_id="run1",
        signal=SatisfactionSignal.THUMBS_UP,
        score=10.0,
    )
    observation = record_user_satisfaction(event, registry=registry)
    assert observation.passed is True
    assert len(registry.list_observations()) == 1
