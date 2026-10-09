# © Artur Czarnecki. All rights reserved.

"""Configured handler validates opaque execution_target_correlation (P3-R2-R1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

from intergrax.contracts.execution.bound_capability_execution_dispatch import (
    BoundCapabilityExecutionDispatchRequest,
)
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.tools.marketplace_qualified_capability_execution_handler import (
    MarketplaceToolQualifiedCapabilityExecutionHandler,
)
from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
    QualifiedMarketplaceToolActivationOutcome,
    QualifiedMarketplaceToolActivationResolver,
)
from intergrax.tools.marketplace_tool_execution_routing import (
    derive_marketplace_configured_tool_execution_intent_target_correlation,
    derive_marketplace_configured_tool_execution_target_reference,
)
from tests.qualification.trace_x._trace_x_p5_r2_p3_r2_configured_negative_support import (
    _ATTEMPT_ID,
    _RUN_ID,
    _TENANT,
    _TASK_ID,
    adoption,
    configured_intent,
    configured_target,
)

_EXECUTION_ID = ExecutionId("execution_" + "a" * 24)


@dataclass
class _CountingCatalogInvoker:
    calls: int = 0

    @property
    def caller_agent_id(self) -> str:
        return "agent-test"

    def invoke(self, request):
        self.calls += 1
        return MagicMock(success=True)


@dataclass
class _RecordingIntentRepo:
    recorded: list[object] = field(default_factory=list)

    def get(self, *, execution_request_id: str):
        if not self.recorded:
            return None
        return self.recorded[0]


@dataclass
class _CountingActivation:
    calls: int = 0

    def ensure_exact_active_for_identity(self, **_kwargs):
        self.calls += 1
        return MagicMock(
            outcome=QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
            reason_detail="activated",
            registry_tool_id="tool-1",
        )


def _handler(
    repo: _RecordingIntentRepo,
    *,
    catalog: _CountingCatalogInvoker,
    activation: _CountingActivation,
) -> MarketplaceToolQualifiedCapabilityExecutionHandler:
    activation_resolver = MagicMock(spec=QualifiedMarketplaceToolActivationResolver)
    activation_resolver.ensure_exact_active_for_identity.side_effect = (
        activation.ensure_exact_active_for_identity
    )
    from intergrax.contracts.tools.qualified_tool_invocation import (
        QualifiedToolInvocationMaterialOutcome,
    )

    material = MagicMock()
    material.provide.return_value = MagicMock(
        outcome=QualifiedToolInvocationMaterialOutcome.AVAILABLE,
        material=MagicMock(),
    )
    return MarketplaceToolQualifiedCapabilityExecutionHandler(
        intent_repository=repo,  # pyright: ignore[reportArgumentType]
        stage_repository=MagicMock(),
        activation_resolver=activation_resolver,
        material_provider=material,
        invocation_resolver=MagicMock(),
        catalog_tool_invoker=catalog,  # pyright: ignore[reportArgumentType]
        configured_invocation_projection=MagicMock(return_value=MagicMock()),
        package_resolver=MagicMock(),
    )


def _dispatch(target) -> BoundCapabilityExecutionDispatchRequest:
    return BoundCapabilityExecutionDispatchRequest(
        execution_request_id="exec-req-configured-1",
        execution_target=target,
        tenant_id=_TENANT,
        task_id=_TASK_ID,
    )


def test_configured_handler_accepts_matching_target_correlation() -> None:
    binding_id = "bind-configured-1"
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent(binding_operation_id=binding_id))
    catalog = _CountingCatalogInvoker()
    activation = _CountingActivation()
    handler = _handler(repo, catalog=catalog, activation=activation)
    result = handler.dispatch_once(
        _dispatch(configured_target(binding_operation_id=binding_id)),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail != "configured_execution_target_correlation_mismatch"
    assert activation.calls == 1


def test_configured_handler_fail_closed_on_correlation_mismatch() -> None:
    binding_id = "bind-configured-1"
    repo = _RecordingIntentRepo()
    intent = configured_intent(binding_operation_id=binding_id)
    repo.recorded.append(
        intent.model_copy(
            update={
                "execution_target_correlation": (
                    derive_marketplace_configured_tool_execution_intent_target_correlation(
                        "other-binding-id",
                    )
                ),
            },
        ),
    )
    catalog = _CountingCatalogInvoker()
    activation = _CountingActivation()
    handler = _handler(repo, catalog=catalog, activation=activation)
    result = handler.dispatch_once(
        _dispatch(configured_target(binding_operation_id=binding_id)),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        integration_configuration_adoption=adoption(),
    )
    assert result.disposition is QualifiedCapabilityExecutionDispatchDisposition.FAILED
    assert result.reason_detail == "configured_execution_target_correlation_mismatch"
    assert catalog.calls == 0
    assert activation.calls == 0


def test_configured_handler_does_not_parse_binding_from_target_reference() -> None:
    """Mismatched target vs intent correlation must fail even if provenance binding matches target derivation."""
    binding_id = "bind-configured-1"
    repo = _RecordingIntentRepo()
    repo.recorded.append(configured_intent(binding_operation_id=binding_id))
    catalog = _CountingCatalogInvoker()
    activation = _CountingActivation()
    handler = _handler(repo, catalog=catalog, activation=activation)
    wrong_ref = derive_marketplace_configured_tool_execution_target_reference("other-binding")
    target = configured_target(binding_operation_id=binding_id).model_copy(
        update={"execution_target_reference": wrong_ref},
    )
    result = handler.dispatch_once(
        _dispatch(target),
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        integration_configuration_adoption=adoption(),
    )
    assert result.reason_detail == "configured_execution_target_correlation_mismatch"
    assert catalog.calls == 0
    assert activation.calls == 0
