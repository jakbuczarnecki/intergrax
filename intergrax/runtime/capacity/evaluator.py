# © Artur Czarnecki. All rights reserved.

"""Scaling evaluator (ECP-3.*)."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from intergrax.runtime.capacity.contracts import (
    CapacitySignal,
    ScalingAction,
    ScalingActionPlan,
    ScalingPolicy,
    ScalingRule,
)
from intergrax.runtime.capacity.events import PublishFn, publish_scale_evaluated


class ScalingEvaluator:
    """Rule matching with cooldown, hysteresis, and anti-flap guard (ECP-3, ECP-7.3)."""

    def __init__(
        self,
        policy: ScalingPolicy,
        *,
        publish: PublishFn | None = None,
    ) -> None:
        self._policy = policy
        self._publish = publish
        self._last_action_at: dict[str, datetime] = {}
        self._action_timestamps: deque[datetime] = deque()

    def evaluate(self, signals: Sequence[CapacitySignal]) -> ScalingActionPlan:
        if not self._policy.enabled:
            plan = ScalingActionPlan(evaluation_status="noop")
            self._emit_evaluated(plan)
            return plan

        if self._actions_in_last_hour() >= self._policy.max_actions_per_hour:
            plan = ScalingActionPlan(evaluation_status="denied")
            self._emit_evaluated(plan)
            return plan

        by_metric = {s.metric_name: s for s in signals}
        actions: list[ScalingAction] = []
        for rule in self._policy.rules:
            signal = by_metric.get(rule.metric_name)
            if signal is None:
                continue
            if self._in_cooldown(rule):
                continue
            if signal.value >= rule.scale_up_threshold:
                actions.append(
                    ScalingAction(
                        kind=rule.action_kind,
                        target=rule.target,
                        delta=rule.delta,
                        reason=f"{rule.metric_name}={signal.value} >= {rule.scale_up_threshold}",
                    )
                )
            elif signal.value <= rule.scale_down_threshold:
                actions.append(
                    ScalingAction(
                        kind=rule.action_kind,
                        target=rule.target,
                        delta=-rule.delta,
                        reason=f"{rule.metric_name}={signal.value} <= {rule.scale_down_threshold}",
                    )
                )

        if not actions:
            plan = ScalingActionPlan(evaluation_status="noop")
            self._emit_evaluated(plan)
            return plan

        if self._policy.require_hitl_for_scale_up and any(a.delta > 0 for a in actions):
            plan = ScalingActionPlan(
                actions=tuple(actions),
                evaluation_status="hitl_required",
            )
            self._emit_evaluated(plan)
            return plan

        for action in actions:
            self._record_action(action.action_id)

        plan = ScalingActionPlan(actions=tuple(actions), evaluation_status="planned")
        self._emit_evaluated(plan)
        return plan

    def _emit_evaluated(self, plan: ScalingActionPlan) -> None:
        if self._publish is not None:
            publish_scale_evaluated(self._publish, plan)

    def _in_cooldown(self, rule: ScalingRule) -> bool:
        last = self._last_action_at.get(rule.rule_id)
        if last is None:
            return False
        return datetime.now(timezone.utc) - last < timedelta(seconds=rule.cooldown_seconds)

    def _record_action(self, action_id: str) -> None:
        now = datetime.now(timezone.utc)
        self._last_action_at[action_id] = now
        self._action_timestamps.append(now)

    def _actions_in_last_hour(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        while self._action_timestamps and self._action_timestamps[0] < cutoff:
            self._action_timestamps.popleft()
        return len(self._action_timestamps)
