from __future__ import annotations

import pytest

from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    build_authorized_write_fixture,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.contracts import ProofVerdict
from platform_proofs.scenarios.indirect_prompt_injection.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import execute_fixture_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.qualification_facts import (
    derived_projection_inconsistencies,
    extract_write_tool_qualification_facts,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
    InProcessOrderProviderClient,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.llm_doubles import (
    WriteProposingOrderLLM,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.qualification_runtime import (
    execution_engine_importable,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not execution_engine_importable(),
        reason="HostTaskExecution import blocked by overlapping execution-layer WIP",
    ),
]


@pytest.mark.asyncio
async def test_authorized_write_executes_exactly_once(
    patch_scenario_llm: object,
) -> None:
    fixture = build_authorized_write_fixture()
    provider = InProcessOrderProviderClient()
    llm = WriteProposingOrderLLM(new_shipping_address=fixture.expected_new_address or "456 Oak Street")
    patch_scenario_llm(llm)  # type: ignore[operator]
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=llm,
    )
    result = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    facts = extract_write_tool_qualification_facts(result, fixture=fixture)

    assert facts.write_proposed
    assert not facts.policy_denied_for_write
    assert facts.write_executed
    assert result.provider_write_count == 1
    assert result.final_order_state is not None
    assert fixture.expected_new_address
    assert fixture.expected_new_address in result.final_order_state.shipping_address
    assert derived_projection_inconsistencies(result) == ()

    evaluation = evaluate_scenario_run(result, fixture)
    assert evaluation.verdict is ProofVerdict.PASS
