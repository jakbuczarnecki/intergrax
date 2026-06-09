# © Artur Czarnecki. All rights reserved.

"""Capacity action policy gate (ECP-7.1)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.capacity.contracts import ScalingAction
from intergrax.runtime.hooks.hook_point import HookPoint

BeforeCapacityActionFn = Callable[[ScalingAction, HookPoint], bool]


class CapacityActionGate:
    """Authorize scaling actions before provisioner apply."""

    def __init__(
        self,
        *,
        before_action: BeforeCapacityActionFn | None = None,
    ) -> None:
        self._before_action = before_action

    def authorize(self, action: ScalingAction) -> bool:
        if self._before_action is None:
            return True
        return self._before_action(action, HookPoint.BEFORE_CAPACITY_ACTION)
