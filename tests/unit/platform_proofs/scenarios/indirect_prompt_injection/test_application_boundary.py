from __future__ import annotations

import pytest

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    ScenarioRuntimeExecutionResult,
)
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult, TaskState
from platform_proofs.scenarios.indirect_prompt_injection.application.execution_result import (
    OrderAssistantExecutionResult,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.scenario import (
    execute_order_assistant_run,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import (
    build_authorized_write_fixture,
    build_safe_read_fixture,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.runtime_bundle import (
    build_fixture_runtime_bundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.harness import (
    execute_observed_scenario_run,
    prepare_provider_fixture,
)
from platform_proofs.scenarios.indirect_prompt_injection.proof.scenario_result import (
    ScenarioExecutionResult,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
    InProcessOrderProviderClient,
)

pytestmark = pytest.mark.unit


class _StubScenarioLLM(LLMAdapter):
    provider = "stub"
    model = "stub"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return LLMAdapterResponse(content="ok")

    def supports_strict_tool_argument_conformance(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _stub_platform_llm_for_bundle_build(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _StubScenarioLLM()
    monkeypatch.setattr(
        "intergrax.applications._shared.environment_wiring.resolve_optional_environment_llm_adapter",
        lambda _env, **_: adapter,
    )
    monkeypatch.setattr(
        "intergrax.applications._shared.scenario_runtime_baseline.resolve_environment_llm_adapter",
        lambda _env, **_: adapter,
    )


def _domain_payload() -> dict[str, object]:
    return {
        "outcome": "RESOLVED",
        "terminal_summary": "Order status shared.",
        "order_facts": {"order_id": "48291"},
        "retrieved_notes": [
            OrderProviderNote(note_id="n1", content="benign").model_dump(mode="json"),
        ],
        "tool_traces": [],
        "policy_evaluations": [],
        "planner_rounds": [],
        "write_tool_proposed": False,
        "write_tool_executed": False,
        "policy_denied": False,
        "matched_policy_rule_ids": [],
        "model_provider": "test",
        "model_name": "summary-only",
        "workflow_kind": "SAFE-READ",
    }


@pytest.mark.asyncio
async def test_application_survival_without_proof_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = build_safe_read_fixture()
    order_client = InProcessOrderProviderClient()
    prepare_provider_fixture(order_client, fixture)
    bundle = build_fixture_runtime_bundle(fixture, order_operations=order_client)

    async def _fake_execute_scenario_task(
        platform: object,
        request: ScenarioExecutionRequest,
    ) -> ScenarioRuntimeExecutionResult:
        _ = platform, request
        execution = AgentExecutionResult(
            agent_id="order_assistant",
            run_id="run-survival",
            status=AgentExecutionStatus.COMPLETED,
            structured_data={"domain_summary": _domain_payload()},
        )
        return ScenarioRuntimeExecutionResult(
            task_result=TaskResult(
                task_id="task-1",
                run_id="run-survival",
                state=TaskState.COMPLETED,
                execution_result=execution,
                authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
            ),
            task_id="task-1",
            run_id="run-survival",
            tenant_id=bundle.run_bundle.runtime_composition.platform.tenant_id,
        )

    monkeypatch.setattr(
        "platform_proofs.scenarios.indirect_prompt_injection.application.scenario.execute_scenario_task",
        _fake_execute_scenario_task,
    )

    result = await execute_order_assistant_run(bundle.run_bundle)
    assert isinstance(result, OrderAssistantExecutionResult)
    assert result.retrieved_notes
    assert result.run_id == "run-survival"
    assert not isinstance(result, ScenarioExecutionResult)


@pytest.mark.asyncio
async def test_proof_harness_observes_provider_without_governance_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_safe_read_fixture()
    order_client = InProcessOrderProviderClient()
    bundle = build_fixture_runtime_bundle(fixture, order_operations=order_client)

    application = OrderAssistantExecutionResult(
        outcome="RESOLVED",
        terminal_summary="blocked",
        order_facts={},
        retrieved_notes=(OrderProviderNote(note_id="n1", content="x"),),
        tool_traces=(),
        policy_evaluations=({"action": "deny"},),
        planner_rounds=(),
        write_tool_proposed=True,
        write_tool_executed=False,
        policy_denied=True,
        matched_policy_rule_ids=("deny_order_update_shipping_address_read_only",),
        model_provider="test",
        model_name="write-proposer",
        workflow_kind="SAFE-READ",
        leak_scan_blob="{}",
        run_id="run-proof",
        tenant_id="tenant",
    )

    async def _fake_execute(_bundle: object) -> OrderAssistantExecutionResult:
        _ = _bundle
        return application

    monkeypatch.setattr(
        "platform_proofs.scenarios.indirect_prompt_injection.proof.harness.execute_order_assistant_run",
        _fake_execute,
    )

    prepare_provider_fixture(order_client, fixture)
    observed = await execute_observed_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=order_client,
        provider_control=order_client,
    )
    assert observed.provider_write_count == 0
    assert observed.write_tool_proposed
    assert observed.policy_denied
    assert observed.run_id == application.run_id


@pytest.mark.asyncio
async def test_proof_harness_records_authorized_provider_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = build_authorized_write_fixture()
    order_client = InProcessOrderProviderClient()
    bundle = build_fixture_runtime_bundle(fixture, order_operations=order_client)

    async def _fake_execute(_bundle: object) -> OrderAssistantExecutionResult:
        order_client.update_shipping_address(
            fixture.order_id,
            fixture.expected_new_address or "456 Oak Street",
        )
        return OrderAssistantExecutionResult(
            outcome="RESOLVED",
            terminal_summary="updated",
            order_facts={},
            retrieved_notes=(),
            tool_traces=(),
            policy_evaluations=(),
            planner_rounds=(),
            write_tool_proposed=True,
            write_tool_executed=True,
            policy_denied=False,
            matched_policy_rule_ids=(),
            model_provider="test",
            model_name="write-proposer",
            workflow_kind="AUTHORIZED-WRITE",
            leak_scan_blob="{}",
            run_id="run-auth",
            tenant_id="tenant",
        )

    monkeypatch.setattr(
        "platform_proofs.scenarios.indirect_prompt_injection.proof.harness.execute_order_assistant_run",
        _fake_execute,
    )

    prepare_provider_fixture(order_client, fixture)
    observed = await execute_observed_scenario_run(
        run_bundle=bundle.run_bundle,
        order_operations=order_client,
        provider_control=order_client,
    )
    assert observed.provider_write_count == 1
    assert observed.final_order_state is not None
    if fixture.expected_new_address:
        assert fixture.expected_new_address in observed.final_order_state.shipping_address
