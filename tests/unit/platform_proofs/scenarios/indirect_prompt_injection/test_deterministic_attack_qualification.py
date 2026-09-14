from __future__ import annotations

import pytest

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    TOOL_ORDER_GET_NOTES,
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    AttackVariantId,
    build_attack_fixture,
    build_authorized_write_fixture,
    build_safe_read_fixture,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.contracts import ProofVerdict
from platform_proofs.scenarios.indirect_prompt_injection.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import execute_fixture_scenario_run
from platform_proofs.scenarios.indirect_prompt_injection.proof.qualification_facts import (
    canonical_write_tool_proposed,
    derived_projection_inconsistencies,
    extract_write_tool_qualification_facts,
    shipping_address_unchanged,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    READ_ONLY_DENY_RULE_ID,
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
async def test_deterministic_attack_denies_write_with_zero_provider_mutations(
    patch_scenario_llm: object,
) -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    provider = InProcessOrderProviderClient()
    patch_scenario_llm(WriteProposingOrderLLM())  # type: ignore[operator]
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=WriteProposingOrderLLM(),
    )
    result = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    facts = extract_write_tool_qualification_facts(result, fixture=fixture)

    assert facts.write_proposed
    assert facts.policy_denied_for_write
    assert READ_ONLY_DENY_RULE_ID in facts.matched_rule_ids
    assert not facts.write_executed
    assert facts.notes_retrieved_via_tool
    assert facts.hostile_fixture_notes_observed
    assert result.provider_write_count == 0
    assert shipping_address_unchanged(result)
    assert result.run_id
    assert derived_projection_inconsistencies(result) == ()

    evaluation = evaluate_scenario_run(result, fixture)
    assert evaluation.verdict is ProofVerdict.PASS


@pytest.mark.asyncio
async def test_deterministic_attack_correlates_same_run_id_across_facts(
    patch_scenario_llm: object,
) -> None:
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    provider = InProcessOrderProviderClient()
    patch_scenario_llm(WriteProposingOrderLLM())  # type: ignore[operator]
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=WriteProposingOrderLLM(),
    )
    result = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    assert result.run_id
    for trace in result.tool_traces:
        raw = trace.raw_trace or {}
        if raw.get("run_id"):
            assert raw["run_id"] == result.run_id
    assert canonical_write_tool_proposed(result)


@pytest.mark.asyncio
async def test_deterministic_attack_fails_qualification_when_write_tool_missing_from_path(
    patch_scenario_llm: object,
) -> None:
    """Anti-fake-security: policy deny alone without proposal evidence is NOT_EXERCISED."""
    fixture = build_attack_fixture(AttackVariantId.ATTACK_DIRECT)
    provider = InProcessOrderProviderClient()
    patch_scenario_llm(WriteProposingOrderLLM())  # type: ignore[operator]
    bundle = build_fixture_runtime_bundle(
        fixture,
        order_operations=provider,
        llm_adapter_override=WriteProposingOrderLLM(),
    )
    result = await execute_fixture_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=provider,
        provider_control=provider,
        fixture=fixture,
    )
    if not canonical_write_tool_proposed(result):
        evaluation = evaluate_scenario_run(result, fixture)
        assert evaluation.verdict is ProofVerdict.NOT_EXERCISED
