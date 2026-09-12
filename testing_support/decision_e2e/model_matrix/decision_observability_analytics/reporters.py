# © Artur Czarnecki. All rights reserved.

"""Report provider plugins (DS-E2E-15J-L8)."""

from __future__ import annotations

from datetime import datetime

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionAnalyticsReport,
    DecisionAnalyticsResult,
)


class AuditTrailReportProvider:
    report_provider_id = "audit_trail"
    report_provider_version = "1"

    def build_report(
        self,
        results: tuple[DecisionAnalyticsResult, ...],
        *,
        generated_at: datetime,
    ) -> DecisionAnalyticsReport:
        lines: list[str] = []
        for result in results:
            audit = result.audit
            period = "unknown"
            if audit.period_start is not None and audit.period_end is not None:
                period = (
                    f"{audit.period_start.isoformat()}..{audit.period_end.isoformat()}"
                )
            lines.append(
                f"analyzer={audit.analyzer_id} version={audit.analyzer_version} "
                f"status={result.status.value} decisions={len(audit.decision_ids)} "
                f"period={period}"
            )
        if not lines:
            lines.append("status=no_analytics_results")
        return DecisionAnalyticsReport(
            report_provider_id=self.report_provider_id,
            report_provider_version=self.report_provider_version,
            generated_at=generated_at,
            analyzer_ids=tuple(result.audit.analyzer_id for result in results),
            summary_lines=tuple(lines),
        )


def default_report_providers() -> tuple[AuditTrailReportProvider, ...]:
    return (AuditTrailReportProvider(),)


__all__ = [
    "AuditTrailReportProvider",
    "default_report_providers",
]
