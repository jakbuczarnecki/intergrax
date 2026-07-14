# © Artur Czarnecki. All rights reserved.

from __future__ import annotations


import pytest
from pydantic import ValidationError

from intergrax.hosting import (
    HostedApplicationHook,
    HostedApplicationHookMode,
    HostedApplicationHookPoint,
    HostedApplicationHooks,
)
from tests.unit.hosting._helpers import flush_state_handler, warm_cache_handler

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "point",
    list(HostedApplicationHookPoint),
)
def test_all_hook_points_supported(point: HostedApplicationHookPoint) -> None:
    hooks = HostedApplicationHooks(**{point.value: ()})
    assert hooks.hooks_for_point(point) == ()


def test_fixed_blocking_observer_semantics() -> None:
    assert HostedApplicationHookPoint.BEFORE_READY.value == "before_ready"
    hooks = HostedApplicationHooks(
        before_ready=(HostedApplicationHook(hook_id="warm", handler=warm_cache_handler),),
        after_ready=(HostedApplicationHook(hook_id="observe", handler=warm_cache_handler),),
    )
    descriptors = hooks.flattened_public_descriptors()
    blocking = next(item for item in descriptors if item.hook_id == "warm")
    observer = next(item for item in descriptors if item.hook_id == "observe")
    assert blocking.mode is HostedApplicationHookMode.BLOCKING
    assert observer.mode is HostedApplicationHookMode.OBSERVER


def test_duplicate_hook_ids_rejected_across_points() -> None:
    with pytest.raises(ValidationError, match="duplicate hook_id"):
        HostedApplicationHooks(
            before_ready=(HostedApplicationHook(hook_id="same", handler=warm_cache_handler),),
            after_stop=(HostedApplicationHook(hook_id="same", handler=flush_state_handler),),
        )


def test_deterministic_descriptor_ordering() -> None:
    hooks = HostedApplicationHooks(
        before_ready=(
            HostedApplicationHook(hook_id="b", handler=warm_cache_handler, priority=1, source_id="b"),
            HostedApplicationHook(hook_id="a", handler=flush_state_handler, priority=0, source_id="a"),
        )
    )
    ordered = [item.hook_id for item in hooks.flattened_public_descriptors()]
    assert ordered == ["a", "b"]


def test_runtime_handler_absent_from_dump_schema_repr() -> None:
    hook = HostedApplicationHook(hook_id="warm", handler=warm_cache_handler)
    assert "handler" not in hook.model_dump()
    assert "handler" not in hook.model_json_schema().get("properties", {})
    assert "handler=" not in repr(hook)


def test_unstable_callable_requires_explicit_handler_id() -> None:
    with pytest.raises(ValidationError, match="handler_id"):
        HostedApplicationHook(hook_id="lambda_hook", handler=lambda context: None)
