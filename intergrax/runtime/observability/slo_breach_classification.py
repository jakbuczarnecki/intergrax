# © Artur Czarnecki. All rights reserved.

"""SLO breach incident classification (IDEAL-30.1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SloBreachSeverity(str, Enum):
    WARNING = "warning"
    PAGE = "page"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class SloBreachIncident:
    slo_id: str
    severity: SloBreachSeverity
    breach_ratio: float
    message: str


def classify_slo_breach(*, slo_id: str, observed: float, target: float) -> SloBreachIncident:
    if target <= 0:
        return SloBreachIncident(
            slo_id=slo_id,
            severity=SloBreachSeverity.WARNING,
            breach_ratio=0.0,
            message="invalid slo target",
        )
    ratio = observed / target
    if ratio >= 2.0:
        severity = SloBreachSeverity.CRITICAL
    elif ratio >= 1.25:
        severity = SloBreachSeverity.PAGE
    else:
        severity = SloBreachSeverity.WARNING
    return SloBreachIncident(
        slo_id=slo_id,
        severity=severity,
        breach_ratio=ratio,
        message=f"observed={observed} target={target}",
    )
