# © Artur Czarnecki. All rights reserved.

"""Quota hard-stop vs warn enforcement (IDEAL-24.5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuotaAction(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    HARD_STOP = "hard_stop"


@dataclass(frozen=True, slots=True)
class TenantQuota:
    max_cost_usd: float
    warn_ratio: float = 0.8


class QuotaExceededError(RuntimeError):
    """Raised when a tenant exceeds a hard-stop quota."""


def evaluate_quota(
    *,
    spent_usd: float,
    quota: TenantQuota,
) -> QuotaAction:
    if spent_usd >= quota.max_cost_usd:
        return QuotaAction.HARD_STOP
    if spent_usd >= quota.max_cost_usd * quota.warn_ratio:
        return QuotaAction.WARN
    return QuotaAction.ALLOW


def assert_quota_allows(*, spent_usd: float, quota: TenantQuota) -> QuotaAction:
    action = evaluate_quota(spent_usd=spent_usd, quota=quota)
    if action is QuotaAction.HARD_STOP:
        raise QuotaExceededError(
            f"tenant quota exceeded: spent={spent_usd} max={quota.max_cost_usd}"
        )
    return action
