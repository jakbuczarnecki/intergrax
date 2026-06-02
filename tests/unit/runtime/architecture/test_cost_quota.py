from __future__ import annotations

from intergrax.runtime.architecture.cost_quota import (
    QuotaEnforcementAction,
    QuotaResourceType,
    QuotaUsageRequest,
    ResourceQuota,
    evaluate_quota_enforcement,
)


def test_quota_enforcement_denies_when_limit_exceeded() -> None:
    report = evaluate_quota_enforcement(
        quotas=[
            ResourceQuota(
                resource_type=QuotaResourceType.TOKENS,
                scope_id="agent:research",
                limit=1000,
                used=990,
            )
        ],
        requests=[
            QuotaUsageRequest(
                resource_type=QuotaResourceType.TOKENS,
                scope_id="agent:research",
                requested_units=20,
            )
        ],
    )
    assert report.decisions[0].action == QuotaEnforcementAction.DENY


def test_quota_enforcement_degrades_near_limit() -> None:
    report = evaluate_quota_enforcement(
        quotas=[
            ResourceQuota(
                resource_type=QuotaResourceType.TOOL_CALLS,
                scope_id="agent:research",
                limit=100,
                used=92,
            )
        ],
        requests=[
            QuotaUsageRequest(
                resource_type=QuotaResourceType.TOOL_CALLS,
                scope_id="agent:research",
                requested_units=1,
            )
        ],
        degrade_threshold_ratio=0.90,
    )
    assert report.decisions[0].action == QuotaEnforcementAction.DEGRADE
