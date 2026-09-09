# © Artur Czarnecki. All rights reserved.

"""NPSC-5C — coordination intent executor tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutItemFailure,
    FanOutItemId,
    FanOutItemOutcome,
    FanOutItemStatus,
    FanOutOrchestrationPort,
    FanOutRequest,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentContractError,
    CoordinationIntentId,
)
from intergrax.agent_distribution.coordination_binding_materialization import (
    CoordinationCollaborativeApplicabilityClassification,
)
from intergrax.agent_distribution.coordination_intent_executor import (
    CoordinationContributionBinding,
    CoordinationIntentBinding,
    CoordinationIntentExecutor,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationFailureCode,
    CoordinationId,
    CoordinationRequest,
    CoordinationResult,
)
from intergrax.agent_distribution.task_capability_resolution import (
    build_task_capability_resolution_request,
    unresolved_agent_distribution_capability_need,
)
from intergrax.agent_distribution.task_scoped_agents import TaskScopedAgentLeaseId
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    mint_task_id,
    require_active_execution_id,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from tests.unit.agent_distribution.test_bounded_multi_agent_fanout import (
    _build_fan_out_service,
    build_fan_out_harness,
)
from tests.unit.agent_distribution.test_coordination_intent import (
    _fan_out_intent,
    _single_intent,
)
from tests.unit.agent_distribution.test_delegated_subtasks import (
    OcrRequest,
    OcrResult,
    _APP,
    _ENV,
    _OCR_PACKAGE,
    _discovery_candidate,
    admin_test_principal,
    build_delegated_harness,
)
from tests.unit.agent_distribution.test_multi_agent_coordination import (
    _build_coordination_service,
    _root_identity,
)
from testing_support.agent_distribution.coordination_governance import (
    allowing_coordination_governance,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())


def _binding(
    task_scope,
    *,
    pairs: tuple[tuple[str, str], ...],
    workspace_id: str | None = None,
) -> CoordinationIntentBinding:
    collaborative_applicability = (
        CoordinationCollaborativeApplicabilityClassification.required(
            workspace_id=workspace_id,
        )
        if workspace_id is not None
        else CoordinationCollaborativeApplicabilityClassification.not_applicable()
    )
    return CoordinationIntentBinding(
        task_scope_id=task_scope,
        application_id=_APP,
        application_environment_id=_ENV,
        contribution_bindings=tuple(
            CoordinationContributionBinding(
                contribution_id=CoordinationContributionId(contribution_id),
                lease_id=TaskScopedAgentLeaseId(lease_id),
            )
            for contribution_id, lease_id in pairs
        ),
        collaborative_applicability=collaborative_applicability,
    )


class _TrackingCoordinationService:
    def __init__(self) -> None:
        self.calls = 0
        self.last_request: CoordinationRequest | None = None

    async def coordinate(self, request, *, delegation, principal):
        del principal
        self.calls += 1
        self.last_request = request
        return CoordinationResult(
            coordination_id=request.coordination_id,
            delegated=SimpleNamespace(
                result=OcrResult(text=f"ocr:{delegation.payload.document_ref}"),
            ),
        )


class _TrackingFanOutService(BoundedMultiAgentFanOutService[OcrRequest, OcrResult]):
    def __init__(self, inner: BoundedMultiAgentFanOutService[OcrRequest, OcrResult]) -> None:
        super().__init__(orchestration=inner._orchestration)
        self._inner = inner
        self.calls = 0

    async def fan_out(self, request, *, principal):
        self.calls += 1
        return await self._inner.fan_out(request, principal=principal)


class _StaticOrchestrationPort(FanOutOrchestrationPort[OcrRequest, OcrResult]):
    def __init__(self, outcomes: tuple[FanOutItemOutcome[OcrResult], ...]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    async def orchestrate_fan_out(
        self,
        request: FanOutRequest[OcrRequest],
        *,
        principal,
    ) -> tuple[FanOutItemOutcome[OcrResult], ...]:
        del principal
        self.calls += 1
        return self._outcomes


def _success_outcome(item_id: str, text: str) -> FanOutItemOutcome[OcrResult]:
    return FanOutItemOutcome(
        item_id=FanOutItemId(item_id),
        status=FanOutItemStatus.SUCCESS,
        result=CoordinationResult(
            coordination_id=CoordinationId(item_id),
            delegated=SimpleNamespace(result=OcrResult(text=text)),
        ),
    )


def _failure_outcome(item_id: str) -> FanOutItemOutcome[OcrResult]:
    return FanOutItemOutcome(
        item_id=FanOutItemId(item_id),
        status=FanOutItemStatus.FAILURE,
        failure=FanOutItemFailure(
            failure_code=CoordinationFailureCode.CHILD_EXECUTION_FAILED,
            message="child failed",
        ),
    )


@pytest.mark.asyncio
async def test_single_execution_routes_through_multi_agent_coordination_service() -> None:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _single_intent("contrib-a")
    binding = _binding(task_scope, pairs=(("contrib-a", "lease-a"),))

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert coordination.calls == 1
    assert coordination.last_request is not None
    assert coordination.last_request.coordination_id == CoordinationId("contrib-a")
    assert fan_out.calls == 0
    assert result.mode is CoordinationExecutionMode.SINGLE
    assert result.single is not None
    assert result.single.contribution_id == CoordinationContributionId("contrib-a")
    assert result.single.coordination.result.text == "ocr:doc-1"


@pytest.mark.asyncio
async def test_fan_out_execution_routes_through_bounded_fan_out_service() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordination = _build_coordination_service(harness)
    orchestration = _StaticOrchestrationPort(
        (
            _success_outcome("contrib-c", "c"),
            _success_outcome("contrib-a", "a"),
            _success_outcome("contrib-b", "b"),
        ),
    )
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(orchestration=orchestration),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(
        ("contrib-a", "contrib-b", "contrib-c"),
        requested_max_concurrency=2,
    )
    binding = _binding(
        task_scope,
        pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
    )

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert fan_out.calls == 1
    assert orchestration.calls == 1
    assert result.mode is CoordinationExecutionMode.FAN_OUT
    assert result.fan_out is not None
    assert [item.item_id for item in result.fan_out.fan_out.items] == [
        FanOutItemId("contrib-a"),
        FanOutItemId("contrib-b"),
        FanOutItemId("contrib-c"),
    ]


@pytest.mark.asyncio
async def test_fan_out_partial_failure_preserves_other_results() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordination = _build_coordination_service(harness)
    orchestration = _StaticOrchestrationPort(
        (
            _success_outcome("contrib-a", "a"),
            _failure_outcome("contrib-b"),
            _success_outcome("contrib-c", "c"),
        ),
    )
    fan_out = BoundedMultiAgentFanOutService(orchestration=orchestration)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b", "contrib-c"))
    binding = _binding(
        task_scope,
        pairs=(
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
            ("contrib-c", "lease-c"),
        ),
    )

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert result.fan_out is not None
    items = result.fan_out.fan_out.items
    assert items[0].status is FanOutItemStatus.SUCCESS
    assert items[1].status is FanOutItemStatus.FAILURE
    assert items[1].failure is not None
    assert (
        items[1].failure.failure_code
        is CoordinationFailureCode.CHILD_EXECUTION_FAILED
    )
    assert items[2].status is FanOutItemStatus.SUCCESS


@pytest.mark.asyncio
async def test_programming_error_propagates_from_coordination_service() -> None:
    class _BrokenCoordination:
        async def coordinate(self, request, *, delegation, principal):
            del request, delegation, principal
            raise TypeError("programming error")

    executor = CoordinationIntentExecutor(
        coordination=_BrokenCoordination(),
        fan_out=BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    with pytest.raises(TypeError, match="programming error"):
        await executor.execute(
            _single_intent(),
            binding=_binding(task_scope, pairs=(("contrib-a", "lease-a"),)),
            principal=admin_test_principal(),
        )


@pytest.mark.asyncio
async def test_single_end_to_end_through_coordination_intent_executor() -> None:
    harness = build_delegated_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordination = _build_coordination_service(harness)
    fan_out = _build_fan_out_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            result = await executor.execute(
                CoordinationIntent(
                    intent_id=CoordinationIntentId("intent-single-e2e"),
                    mode=CoordinationExecutionMode.SINGLE,
                    contributions=(
                        CoordinationContribution(
                            contribution_id=CoordinationContributionId("contrib-e2e"),
                            payload=request,
                            capability_need=unresolved_agent_distribution_capability_need(
                                build_task_capability_resolution_request(
                                    task_kind="document.ocr",
                                ),
                            ),
                        ),
                    ),
                ),
                binding=_binding(task_scope, pairs=(("contrib-e2e", "lease-e2e"),)),
                principal=admin_test_principal(),
            )
            captured.append(result)
            assert result.single is not None
            return result.single.coordination.result

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="doc-e2e"))
    assert captured
    assert captured[0].single is not None
    assert captured[0].single.coordination.result.text == "ocr:doc-e2e"


@pytest.mark.asyncio
async def test_fan_out_end_to_end_without_decision_through_nexus() -> None:
    harness = build_fan_out_harness(
        candidates=(
            _discovery_candidate(_OCR_PACKAGE, capability_ids=("document.ocr",)),
        ),
    )
    coordination = _build_coordination_service(harness)
    fan_out = _build_fan_out_service(harness)
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    harness.task_scope_authority.task_scope_id = task_scope
    root = _root_identity()
    captured = []

    class RootDelegate:
        async def execute(self, request: OcrRequest) -> OcrResult:
            del request
            budget_token = bind_root_execution_budget(
                execution_id=require_active_execution_id(),
                ledger=_UNLIMITED_LEDGER,
            )
            try:
                result = await executor.execute(
                    _fan_out_intent(
                        ("contrib-a", "contrib-b", "contrib-c"),
                        requested_max_concurrency=2,
                    ),
                    binding=_binding(
                        task_scope,
                        pairs=(
                            ("contrib-a", "lease-a"),
                            ("contrib-b", "lease-b"),
                            ("contrib-c", "lease-c"),
                        ),
                    ),
                    principal=admin_test_principal(),
                )
            finally:
                reset_active_execution_budget(budget_token)
            captured.append(result)
            assert result.fan_out is not None
            texts = tuple(
                item.result.result.text
                for item in result.fan_out.fan_out.items
                if item.status is FanOutItemStatus.SUCCESS and item.result is not None
            )
            return OcrResult(text="|".join(texts))

    await ExecutionBoundary[OcrRequest, OcrResult](
        RootDelegate(),
        identity=root,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(OcrRequest(document_ref="ignored"))
    assert captured
    result = captured[0]
    assert result.fan_out is not None
    assert [item.item_id for item in result.fan_out.fan_out.items] == [
        FanOutItemId("contrib-a"),
        FanOutItemId("contrib-b"),
        FanOutItemId("contrib-c"),
    ]
    assert all(
        item.status is FanOutItemStatus.SUCCESS
        for item in result.fan_out.fan_out.items
    )


def _executor_with_tracking() -> tuple[
    CoordinationIntentExecutor[OcrRequest, OcrResult],
    _TrackingCoordinationService,
    _TrackingFanOutService,
]:
    coordination = _TrackingCoordinationService()
    fan_out = _TrackingFanOutService(
        BoundedMultiAgentFanOutService(
            orchestration=_StaticOrchestrationPort(()),
        ),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    return executor, coordination, fan_out


@pytest.mark.asyncio
async def test_single_binding_wrong_contribution_id_fails_closed() -> None:
    executor, coordination, fan_out = _executor_with_tracking()
    task_scope = mint_task_id()

    with pytest.raises(
        CoordinationIntentContractError,
        match="missing contribution binding ids: \\[contrib-a\\]",
    ):
        await executor.execute(
            _single_intent("contrib-a"),
            binding=_binding(task_scope, pairs=(("contrib-b", "lease-b"),)),
            principal=admin_test_principal(),
        )

    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_fan_out_reordered_bindings_resolve_by_contribution_id() -> None:
    coordination = _TrackingCoordinationService()
    captured_leases: list[str] = []

    class _LeaseCapturingOrchestration(FanOutOrchestrationPort[OcrRequest, OcrResult]):
        async def orchestrate_fan_out(
            self,
            request: FanOutRequest[OcrRequest],
            *,
            principal,
        ) -> tuple[FanOutItemOutcome[OcrResult], ...]:
            del principal
            for item in request.items:
                captured_leases.append(str(item.request.lease_id))
            return tuple(
                _success_outcome(str(item.item_id), str(item.item_id))
                for item in request.items
            )

    fan_out = BoundedMultiAgentFanOutService(
        orchestration=_LeaseCapturingOrchestration(),
    )
    executor = CoordinationIntentExecutor(
        coordination=coordination,
        fan_out=fan_out,
        governance=allowing_coordination_governance(),
    )
    task_scope = mint_task_id()
    intent = _fan_out_intent(("contrib-a", "contrib-b", "contrib-c"))
    binding = _binding(
        task_scope,
        pairs=(
            ("contrib-c", "lease-c"),
            ("contrib-a", "lease-a"),
            ("contrib-b", "lease-b"),
        ),
    )

    result = await executor.execute(
        intent,
        binding=binding,
        principal=admin_test_principal(),
    )

    assert captured_leases == ["lease-a", "lease-b", "lease-c"]
    assert result.fan_out is not None
    assert [item.item_id for item in result.fan_out.fan_out.items] == [
        FanOutItemId("contrib-a"),
        FanOutItemId("contrib-b"),
        FanOutItemId("contrib-c"),
    ]


@pytest.mark.asyncio
async def test_binding_missing_contribution_id_fails_closed() -> None:
    executor, coordination, fan_out = _executor_with_tracking()
    task_scope = mint_task_id()

    with pytest.raises(
        CoordinationIntentContractError,
        match="missing contribution binding ids: \\[contrib-c\\]",
    ):
        await executor.execute(
            _fan_out_intent(("contrib-a", "contrib-b", "contrib-c")),
            binding=_binding(
                task_scope,
                pairs=(
                    ("contrib-a", "lease-a"),
                    ("contrib-b", "lease-b"),
                ),
            ),
            principal=admin_test_principal(),
        )

    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_binding_extra_contribution_id_fails_closed() -> None:
    executor, coordination, fan_out = _executor_with_tracking()
    task_scope = mint_task_id()

    with pytest.raises(
        CoordinationIntentContractError,
        match="extra contribution binding ids: \\[contrib-c\\]",
    ):
        await executor.execute(
            _fan_out_intent(("contrib-a", "contrib-b")),
            binding=_binding(
                task_scope,
                pairs=(
                    ("contrib-a", "lease-a"),
                    ("contrib-b", "lease-b"),
                    ("contrib-c", "lease-c"),
                ),
            ),
            principal=admin_test_principal(),
        )

    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_binding_duplicate_contribution_id_fails_closed() -> None:
    executor, coordination, fan_out = _executor_with_tracking()
    task_scope = mint_task_id()

    with pytest.raises(
        CoordinationIntentContractError,
        match="duplicate contribution binding id: contrib-a",
    ):
        await executor.execute(
            _fan_out_intent(("contrib-a", "contrib-b")),
            binding=_binding(
                task_scope,
                pairs=(
                    ("contrib-a", "lease-a"),
                    ("contrib-a", "lease-a-duplicate"),
                ),
            ),
            principal=admin_test_principal(),
        )

    assert coordination.calls == 0
    assert fan_out.calls == 0


@pytest.mark.asyncio
async def test_binding_unknown_same_count_fails_closed() -> None:
    executor, coordination, fan_out = _executor_with_tracking()
    task_scope = mint_task_id()

    with pytest.raises(
        CoordinationIntentContractError,
        match="missing contribution binding ids: \\[contrib-b\\]",
    ):
        await executor.execute(
            _fan_out_intent(("contrib-a", "contrib-b")),
            binding=_binding(
                task_scope,
                pairs=(
                    ("contrib-a", "lease-a"),
                    ("contrib-c", "lease-c"),
                ),
            ),
            principal=admin_test_principal(),
        )

    assert coordination.calls == 0
    assert fan_out.calls == 0
