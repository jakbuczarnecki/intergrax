# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Coordination intent execution router (NPSC-5C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.agent_distribution.coordination_binding_materialization import (
    CoordinationCollaborativeApplicabilityClassification,
    reconcile_coordination_collaborative_applicability,
)
from intergrax.agent_distribution.coordination_governance_adapter import (
    build_multi_agent_coordination_governance_request,
)
from intergrax.agent_distribution.bounded_multi_agent_fanout import (
    BoundedMultiAgentFanOutService,
    FanOutId,
    FanOutItem,
    FanOutItemId,
    FanOutRequest,
    FanOutResult,
)
from intergrax.agent_distribution.coordination_intent import (
    CoordinationContribution,
    CoordinationContributionId,
    CoordinationExecutionMode,
    CoordinationIntent,
    CoordinationIntentContractError,
    CoordinationIntentId,
    effective_fan_out_max_concurrency,
    validate_coordination_intent,
)
from intergrax.agent_distribution.multi_agent_coordination import (
    CoordinationDelegation,
    CoordinationId,
    CoordinationRequest,
    CoordinationResult,
    MultiAgentCoordinationService,
)
from intergrax.agent_distribution.task_scoped_agents import (
    TaskScopedAgentLeaseId,
    TaskScopeId,
)
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.multi_agent_coordination_governance import (
    MultiAgentCoordinationGovernancePort,
    MultiAgentCoordinationGovernanceResult,
    evidence_from_request_and_decision,
    multi_agent_coordination_governance_request_digest,
)
from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_governed_execution_task import (
    peek_governed_execution_task,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class CoordinationContributionBinding:
    """Per-contribution runtime binding supplied at execution."""

    contribution_id: CoordinationContributionId
    lease_id: TaskScopedAgentLeaseId


@dataclass(frozen=True, slots=True)
class CoordinationIntentBinding:
    """Runtime binding supplied at execution — not part of semantic intent."""

    task_scope_id: TaskScopeId
    application_id: str
    application_environment_id: str
    contribution_bindings: tuple[CoordinationContributionBinding, ...]
    collaborative_applicability: CoordinationCollaborativeApplicabilityClassification

    def __post_init__(self) -> None:
        if not self.contribution_bindings:
            raise ValueError("contribution_bindings must be non-empty")


@dataclass(frozen=True, slots=True)
class CoordinationIntentSingleResult(Generic[ResultT]):
    """Typed single-contribution coordination intent outcome."""

    intent_id: CoordinationIntentId
    contribution_id: CoordinationContributionId
    coordination: CoordinationResult[ResultT]


@dataclass(frozen=True, slots=True)
class CoordinationIntentFanOutResult(Generic[ResultT]):
    """Typed fan-out coordination intent outcome."""

    intent_id: CoordinationIntentId
    fan_out: FanOutResult[ResultT]


@dataclass(frozen=True, slots=True)
class CoordinationIntentResult(Generic[ResultT]):
    """Discriminated aggregate result for one coordination intent execution."""

    intent_id: CoordinationIntentId
    mode: CoordinationExecutionMode
    single: CoordinationIntentSingleResult[ResultT] | None = None
    fan_out: CoordinationIntentFanOutResult[ResultT] | None = None

    def __post_init__(self) -> None:
        if self.mode is CoordinationExecutionMode.SINGLE:
            if self.single is None or self.fan_out is not None:
                raise ValueError(
                    "SINGLE result requires single and forbids fan_out",
                )
            return
        if self.mode is CoordinationExecutionMode.FAN_OUT:
            if self.fan_out is None or self.single is not None:
                raise ValueError(
                    "FAN_OUT result requires fan_out and forbids single",
                )
            return
        raise ValueError(f"unsupported coordination execution mode: {self.mode}")


class CoordinationGovernanceDenied(Exception):
    """Mandatory coordination admission denied before execution side effects."""

    def __init__(self, result: MultiAgentCoordinationGovernanceResult) -> None:
        self.result = result
        super().__init__(result.decision.reason or "coordination governance denied")


class CoordinationGovernanceRequiresHuman(Exception):
    """Coordination admission requires canonical governed continuation."""

    def __init__(self, result: MultiAgentCoordinationGovernanceResult) -> None:
        self.result = result
        super().__init__(
            result.decision.reason or "coordination governance requires human approval",
        )


def _validate_and_index_binding(
    intent: CoordinationIntent[RequestT],
    binding: CoordinationIntentBinding,
) -> dict[CoordinationContributionId, CoordinationContributionBinding]:
    expected_ids = {contribution.contribution_id for contribution in intent.contributions}
    indexed_bindings: dict[CoordinationContributionId, CoordinationContributionBinding] = {}
    duplicate_ids: list[CoordinationContributionId] = []

    for contribution_binding in binding.contribution_bindings:
        contribution_id = contribution_binding.contribution_id
        if contribution_id in indexed_bindings:
            duplicate_ids.append(contribution_id)
            continue
        indexed_bindings[contribution_id] = contribution_binding

    if duplicate_ids:
        duplicate_id = sorted(duplicate_ids, key=str)[0]
        raise CoordinationIntentContractError(
            f"duplicate contribution binding id: {duplicate_id}",
        )

    actual_ids = set(indexed_bindings)
    missing_ids = sorted(expected_ids - actual_ids, key=str)
    if missing_ids:
        missing = ", ".join(str(contribution_id) for contribution_id in missing_ids)
        raise CoordinationIntentContractError(
            f"missing contribution binding ids: [{missing}]",
        )

    extra_ids = sorted(actual_ids - expected_ids, key=str)
    if extra_ids:
        extra = ", ".join(str(contribution_id) for contribution_id in extra_ids)
        raise CoordinationIntentContractError(
            f"extra contribution binding ids: [{extra}]",
        )

    return indexed_bindings


def _materialize_coordination_request(
    contribution: CoordinationContribution[RequestT],
    *,
    binding: CoordinationIntentBinding,
    contribution_binding: CoordinationContributionBinding,
) -> CoordinationRequest:
    return CoordinationRequest(
        coordination_id=CoordinationId(str(contribution.contribution_id)),
        delegation_id=str(contribution.contribution_id),
        task_scope_id=binding.task_scope_id,
        application_id=binding.application_id,
        application_environment_id=binding.application_environment_id,
        lease_id=contribution_binding.lease_id,
        capability_need=contribution.capability_need,
        policy=contribution.policy,
    )


def _materialize_fan_out_request(
    intent: CoordinationIntent[RequestT],
    *,
    binding: CoordinationIntentBinding,
    contribution_binding_index: dict[
        CoordinationContributionId,
        CoordinationContributionBinding,
    ],
) -> FanOutRequest[RequestT]:
    items: list[FanOutItem[RequestT]] = []
    for contribution in intent.contributions:
        contribution_binding = contribution_binding_index[contribution.contribution_id]
        items.append(
            FanOutItem(
                item_id=FanOutItemId(str(contribution.contribution_id)),
                request=_materialize_coordination_request(
                    contribution,
                    binding=binding,
                    contribution_binding=contribution_binding,
                ),
                delegation=CoordinationDelegation(payload=contribution.payload),
            ),
        )
    return FanOutRequest(
        fan_out_id=FanOutId(str(intent.intent_id)),
        items=tuple(items),
        max_concurrency=effective_fan_out_max_concurrency(intent),
    )


class CoordinationIntentExecutor(Generic[RequestT, ResultT]):
    """Validate intent semantics, evaluate governance, route through NPSC-5A / 5B."""

    __slots__ = ("_coordination", "_fan_out", "_governance")

    def __init__(
        self,
        *,
        coordination: MultiAgentCoordinationService[RequestT, ResultT],
        fan_out: BoundedMultiAgentFanOutService[RequestT, ResultT],
        governance: MultiAgentCoordinationGovernancePort,
    ) -> None:
        self._coordination = coordination
        self._fan_out = fan_out
        self._governance = governance

    def _fail_closed_governance(
        self,
        intent: CoordinationIntent[RequestT],
        *,
        binding: CoordinationIntentBinding,
        principal: RequestIdentity,
        reason: str,
    ) -> MultiAgentCoordinationGovernanceResult:
        request = build_multi_agent_coordination_governance_request(
            intent,
            task_scope_id=binding.task_scope_id,
            application_id=binding.application_id,
            application_environment_id=binding.application_environment_id,
            principal=principal,
            collaborative_applicability=binding.collaborative_applicability,
        )
        decision = PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            enforcement_level=EnforcementLevel.MANDATORY,
            policy_rule_id="multi_agent_coordination.collaborative_applicability.fail_closed",
        )
        request_digest = multi_agent_coordination_governance_request_digest(request)
        evidence = evidence_from_request_and_decision(
            request,
            decision=decision,
            request_digest=request_digest,
        )
        return MultiAgentCoordinationGovernanceResult(
            permitted=False,
            decision=decision,
            evidence=evidence,
            requires_governed_continuation=False,
            validation_failed=True,
        )

    def _enforce_governance(
        self,
        intent: CoordinationIntent[RequestT],
        *,
        binding: CoordinationIntentBinding,
        principal: RequestIdentity,
    ) -> MultiAgentCoordinationGovernanceResult:
        reconcile_reason = reconcile_coordination_collaborative_applicability(
            binding.collaborative_applicability,
            peek_governed_execution_task(),
        )
        if reconcile_reason is not None:
            raise CoordinationGovernanceDenied(
                self._fail_closed_governance(
                    intent,
                    binding=binding,
                    principal=principal,
                    reason=reconcile_reason,
                ),
            )

        request = build_multi_agent_coordination_governance_request(
            intent,
            task_scope_id=binding.task_scope_id,
            application_id=binding.application_id,
            application_environment_id=binding.application_environment_id,
            principal=principal,
            collaborative_applicability=binding.collaborative_applicability,
        )
        result = self._governance.evaluate(request)
        if result.permitted:
            return result
        if result.requires_governed_continuation:
            raise CoordinationGovernanceRequiresHuman(result)
        raise CoordinationGovernanceDenied(result)

    async def execute(
        self,
        intent: CoordinationIntent[RequestT],
        *,
        binding: CoordinationIntentBinding,
        principal: RequestIdentity,
    ) -> CoordinationIntentResult[ResultT]:
        try:
            validate_coordination_intent(intent)
            contribution_binding_index = _validate_and_index_binding(intent, binding)
        except CoordinationIntentContractError as exc:
            raise CoordinationIntentContractError(str(exc)) from exc

        self._enforce_governance(intent, binding=binding, principal=principal)

        if intent.mode is CoordinationExecutionMode.SINGLE:
            contribution = intent.contributions[0]
            contribution_binding = contribution_binding_index[
                contribution.contribution_id
            ]
            coordination_result = await self._coordination.coordinate(
                _materialize_coordination_request(
                    contribution,
                    binding=binding,
                    contribution_binding=contribution_binding,
                ),
                delegation=CoordinationDelegation(payload=contribution.payload),
                principal=principal,
            )
            return CoordinationIntentResult(
                intent_id=intent.intent_id,
                mode=CoordinationExecutionMode.SINGLE,
                single=CoordinationIntentSingleResult(
                    intent_id=intent.intent_id,
                    contribution_id=contribution.contribution_id,
                    coordination=coordination_result,
                ),
            )

        fan_out_result = await self._fan_out.fan_out(
            _materialize_fan_out_request(
                intent,
                binding=binding,
                contribution_binding_index=contribution_binding_index,
            ),
            principal=principal,
        )
        return CoordinationIntentResult(
            intent_id=intent.intent_id,
            mode=CoordinationExecutionMode.FAN_OUT,
            fan_out=CoordinationIntentFanOutResult(
                intent_id=intent.intent_id,
                fan_out=fan_out_result,
            ),
        )


__all__ = [
    "CoordinationContributionBinding",
    "CoordinationGovernanceDenied",
    "CoordinationGovernanceRequiresHuman",
    "CoordinationIntentBinding",
    "CoordinationIntentExecutor",
    "CoordinationIntentFanOutResult",
    "CoordinationIntentResult",
    "CoordinationIntentSingleResult",
]
