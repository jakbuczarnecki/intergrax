#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-31.2 — evaluation evidence required before production promotion."""

from __future__ import annotations

import sys

from intergrax.runtime.architecture.agent_certification import AgentCertificationEvaluation
from intergrax.runtime.architecture.agent_promotion import (
    PromotionEvidenceBundle,
    PromotionStage,
    evaluate_agent_promotion,
)


def main() -> int:
    bundle = PromotionEvidenceBundle(
        agent_id="echo",
        agent_version="1.0.0",
        source_stage=PromotionStage.STAGING,
        target_stage=PromotionStage.PRODUCTION,
        certification=AgentCertificationEvaluation(
            agent_id="echo",
            agent_version="1.0.0",
            eligible=True,
            reasons=[],
        ),
        evaluation_report_refs=[],
        rollback_plan_ref="runbook/rollback-echo",
        change_ticket_ref="CHG-001",
    )
    decision = evaluate_agent_promotion(bundle)
    if decision.approved:
        print("promotion must be rejected without evaluation_report_refs", file=sys.stderr)
        return 1

    bundle_ok = bundle.model_copy(
        update={"evaluation_report_refs": ["eval/echo/baseline-2026.json"]},
    )
    approved = evaluate_agent_promotion(bundle_ok)
    if not approved.approved:
        print(f"promotion should pass with eval evidence: {approved.reasons}", file=sys.stderr)
        return 1

    print("OK: agent promotion eval gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
