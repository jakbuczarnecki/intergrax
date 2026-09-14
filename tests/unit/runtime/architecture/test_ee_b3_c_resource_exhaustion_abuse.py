# © Artur Czarnecki. All rights reserved.

"""EE-B3-C — AC-12 resource exhaustion abuse (bounded controls)."""

from __future__ import annotations

from typing import cast

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    FanOutId,
    FanOutRequest,
    InvalidFanOutError,
    MAX_FAN_OUT_CONCURRENCY,
    MAX_FAN_OUT_ITEMS,
    validate_fan_out_request,
)
from intergrax.contracts.execution_identity import mint_task_id
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _fan_out_item,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_b3_c_fan_out_rejects_item_count_above_platform_limit() -> None:
    task_scope = mint_task_id()
    items = tuple(
        _fan_out_item(
            item_id=f"item-{index}",
            task_scope=task_scope,
            coordination_id=f"coord-{index}",
            delegation_id=f"delegation-{index}",
            lease_id=f"lease-{index}",
        )
        for index in range(MAX_FAN_OUT_ITEMS + 1)
    )
    request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-b3c-overflow"),
        items=items,
        max_concurrency=1,
    )
    with pytest.raises(InvalidFanOutError, match=str(MAX_FAN_OUT_ITEMS)):
        validate_fan_out_request(cast(FanOutRequest[object], request))


def test_ee_b3_c_fan_out_rejects_concurrency_above_platform_limit() -> None:
    task_scope = mint_task_id()
    request = FanOutRequest(
        fan_out_id=FanOutId("fan-out-b3c-concurrency"),
        items=(
            _fan_out_item(
                item_id="item-0",
                task_scope=task_scope,
                coordination_id="coord-0",
                delegation_id="delegation-0",
                lease_id="lease-0",
            ),
        ),
        max_concurrency=MAX_FAN_OUT_CONCURRENCY + 1,
    )
    with pytest.raises(InvalidFanOutError, match=str(MAX_FAN_OUT_CONCURRENCY)):
        validate_fan_out_request(cast(FanOutRequest[object], request))
