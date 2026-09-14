from __future__ import annotations

import pytest

from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import build_safe_read_fixture
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.contracts import ProofVerdict
from platform_proofs.scenarios.indirect_prompt_injection.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import execute_fixture_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.qualification_facts import (
    extract_write_tool_qualification_facts,
    shipping_address_unchanged,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
    InProcessOrderProviderClient,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.llm_doubles import (
    SummaryOnlyOrderLLM,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.qualification_runtime import (
    execution_engine_importable,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not execution_engine_importable(),
        reason="HostTaskExecution import blocked by overlapping execution-layer WIP",
    ),
]


@pytest.mark.asyncio
async def test_safe_read_control_no_write_proposal_or_provider_mutation(
    patch_scenario_llm: object,
) -> None:
    fixture = build_safe_read_fixture()
    provider = InProcessOrderProviderClient()
    patch_scenario_llm(SummaryOnlyOrderLLM())  # type: ignore[operator]
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=SummaryOnlyOrderLLM(),
    )
    result = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    facts = extract_write_tool_qualification_facts(result, fixture=fixture)

    assert not facts.write_proposed
    assert result.provider_write_count == 0
    assert shipping_address_unchanged(result)

    evaluation = evaluate_scenario_run(result, fixture)
    assert evaluation.verdict is ProofVerdict.PASS
