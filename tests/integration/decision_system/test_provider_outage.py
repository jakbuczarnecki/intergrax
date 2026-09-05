# © Artur Czarnecki. All rights reserved.

"""DS-E2E-09 — provider outage fail-closed."""

from __future__ import annotations

import pytest

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile

from testing_support.decision_e2e.composition import (
    build_qualification_composition,
    mint_qualification_identity,
    run_single_model_producer,
)
from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.environment import QualificationEnvironment

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.external_provider,
    pytest.mark.network,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


def _outage_environment(base: QualificationEnvironment) -> QualificationEnvironment:
    outage_profile = LLMProfile(
        provider=LLMProvider.OLLAMA,
        model=base.producer_profile.model,
        options={"base_url": "http://127.0.0.1:19"},
    )
    outage_adapter = outage_profile.create_adapter()
    return QualificationEnvironment(
        producer_profile=outage_profile,
        producer_adapter=outage_adapter,
        verifier_profile=base.verifier_profile,
        verifier_adapter=base.verifier_adapter,
        council_profile_b=base.council_profile_b,
        council_adapter_b=base.council_adapter_b,
        council_profile_c=base.council_profile_c,
        council_adapter_c=base.council_adapter_c,
        producer_evidence=base.producer_evidence,
        verifier_evidence=base.verifier_evidence,
        council_b_evidence=base.council_b_evidence,
        council_c_evidence=base.council_c_evidence,
        independence_level=base.independence_level,
    )


@pytest.mark.asyncio
async def test_ds_e2e_09_provider_outage_fail_closed(
    require_decision_e2e_environment,
    decision_e2e_report_collector,
) -> None:
    outage_env = _outage_environment(require_decision_e2e_environment)
    composition = build_qualification_composition(outage_env)
    identity = mint_qualification_identity(subject="provider-outage")
    with pytest.raises(Exception):
        await run_single_model_producer(
            composition,
            identity=identity,
            task_message="This must fail through canonical adapter boundary.",
        )

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_09,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
            reason="unreachable provider endpoint fail-closed",
        ),
    )
