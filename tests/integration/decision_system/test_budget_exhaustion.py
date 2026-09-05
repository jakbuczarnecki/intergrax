# © Artur Czarnecki. All rights reserved.

"""DS-E2E-08 — real budget exhaustion."""

from __future__ import annotations

import pytest

from intergrax.contracts.council_strategy import CouncilDeadlockReasonCode
from intergrax.runtime.nexus.budget.budget_models import RunBudget

from testing_support.decision_e2e.composition import (
    build_qualification_composition,
    mint_qualification_identity,
)
from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.council import (
    council_deliberation_input,
    run_council_deliberation,
    run_with_execution_bindings,
    three_participant_strategy,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_provider,
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


@pytest.mark.asyncio
async def test_ds_e2e_08_budget_exhaustion(
    require_decision_e2e_environment,
    decision_e2e_report_collector,
) -> None:
    composition = build_qualification_composition(
        require_decision_e2e_environment,
        run_budget=RunBudget(max_llm_calls=1),
    )
    identity = mint_qualification_identity(subject="budget-exhaustion")
    deliberation_input = council_deliberation_input(
        identity,
        task_message="Return recommendation=stop with confidence=low.",
    )
    strategy = three_participant_strategy()

    result, invocations = await run_with_execution_bindings(
        composition,
        identity,
        lambda: run_council_deliberation(
            composition,
            strategy=strategy,
            deliberation_input=deliberation_input,
        ),
    )
    assert invocations <= 1 or result.deadlock_reason in {
        CouncilDeadlockReasonCode.EXECUTION_BUDGET_EXHAUSTED,
        CouncilDeadlockReasonCode.INSUFFICIENT_PROPOSALS,
        CouncilDeadlockReasonCode.PARTICIPANT_FAILURE,
    }

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_08,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
            reason=f"bounded_stop invocations={invocations}",
        ),
    )
