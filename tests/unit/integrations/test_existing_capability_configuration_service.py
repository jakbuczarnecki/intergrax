# © Artur Czarnecki. All rights reserved.

"""P1 governed façade and pure realization service qualification."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationEvidence,
    ControlPlaneMutationAuthorizationPort,
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationRequest,
    ControlPlaneMutationRisk,
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.execution_identity import RunId, TaskId
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.binding import IntegrationBinding
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationStrategy,
    ExistingCapabilityIntegrationTarget,
    IntegrationConfigurationPayload,
    MUTATION_TYPE_INTEGRATION_CONFIGURATION_REALIZE_V1,
    RESOURCE_TYPE_INTEGRATION_CONFIGURATION,
    project_control_plane_mutation_request,
)
from intergrax.integrations.contracts.shipped_manifests import SQLITE
from intergrax.integrations.existing_capability_configuration_facade import (
    ExistingCapabilityConfigurationRealizationFacade,
)
from intergrax.integrations.existing_capability_configuration_service import (
    ExistingCapabilityConfigurationRealizationService,
)

pytestmark = pytest.mark.unit

_TASK_ID = TaskId("task_01234567890123456789012345678901")
_RUN_ID = RunId("run_01234567890123456789012345678901")

_PRODUCTION_PATHS = (
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/contracts/existing_capability_configuration.py",
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/existing_capability_configuration_service.py",
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/existing_capability_configuration_facade.py",
)


@dataclass(frozen=True)
class _TestConfigurationPayload:
    _configuration_type: str
    _configuration_version: str
    _configuration_fingerprint: str

    @property
    def configuration_type(self) -> str:
        return self._configuration_type

    @property
    def configuration_version(self) -> str:
        return self._configuration_version

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint


def _principal(*, tenant_id: str = "tenant-a") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )


def _payload(fingerprint: str = "fp-test-001") -> IntegrationConfigurationPayload:
    return _TestConfigurationPayload(
        _configuration_type="test.config.v1",
        _configuration_version="1",
        _configuration_fingerprint=fingerprint,
    )


def _request(
    *,
    tenant_id: str = "tenant-a",
    principal: RequestIdentity | None = None,
    provider_id: str = "sqlite",
    current_revision: str = "rev-1",
    configuration_fingerprint: str = "fp-test-001",
    resource_scope: str = "scope-a",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
) -> ExistingCapabilityConfigurationRealizationRequest:
    return ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-1",
        tenant_id=tenant_id,
        principal=principal or _principal(tenant_id=tenant_id),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=provider_id,
        resource_scope=resource_scope,
        configuration=_payload(configuration_fingerprint),
        configuration_fingerprint=configuration_fingerprint,
        current_revision=current_revision,
        risk_classification=ControlPlaneMutationRisk.LOW,
        task_id=task_id,
        run_id=run_id,
    )


def _target_for(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> ExistingCapabilityIntegrationTarget:
    return ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id=request.provider_id,
        current_revision=request.current_revision,
        binding=IntegrationBinding.from_manifest(SQLITE),
    )


@dataclass
class _RecordingResolver:
    target: ExistingCapabilityIntegrationTarget | None = None
    calls: int = 0
    raise_error: bool = False

    def resolve_existing_integration(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityIntegrationTarget:
        self.calls += 1
        if self.raise_error or self.target is None:
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.TARGET_NOT_FOUND,
            )
        return self.target


@dataclass
class _StubStrategy:
    strategy_id_value: str = "stub-strategy"
    match: bool = True
    realize_calls: int = 0
    output_tenant: str | None = None
    output_provider: str | None = None

    @property
    def strategy_id(self) -> str:
        return self.strategy_id_value

    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool:
        return self.match

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> ConfiguredCapabilityBinding:
        self.realize_calls += 1
        tenant = self.output_tenant or request.tenant_id
        provider = self.output_provider or request.provider_id
        return ConfiguredCapabilityBinding(
            tenant_id=tenant,
            integration_category=request.integration_category,
            provider_id=provider,
            configuration_type=request.configuration.configuration_type,
            configuration_version=request.configuration.configuration_version,
            configuration_fingerprint=request.configuration_fingerprint,
            configured_binding=existing_target.binding,
            realization_evidence_refs=("evidence://realized/1",),
        )


@dataclass
class _RecordingAuthorizationPort:
    decision: PolicyDecision = field(
        default_factory=lambda: PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    )
    permitted: bool = True
    authorize_calls: int = 0
    raise_error: bool = False
    evidence_overrides: dict[str, object] = field(default_factory=dict)

    def authorize(
        self,
        request: ControlPlaneMutationRequest,
    ) -> ControlPlaneMutationAuthorizationResult:
        self.authorize_calls += 1
        if self.raise_error:
            raise RuntimeError("authorization exploded")
        digest = control_plane_mutation_request_digest(request)
        evidence = evidence_from_request_and_decision(
            request,
            decision=self.decision,
            request_digest=digest,
        )
        for key, value in self.evidence_overrides.items():
            evidence = evidence.model_copy(update={key: value})
        return ControlPlaneMutationAuthorizationResult(
            permitted=self.permitted,
            decision=self.decision,
            evidence=evidence,
        )


def _allow_evidence_for(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> ControlPlaneMutationAuthorizationEvidence:
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    return evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )


def _service_with(
    *,
    resolver: _RecordingResolver,
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
) -> ExistingCapabilityConfigurationRealizationService:
    return ExistingCapabilityConfigurationRealizationService(
        strategies=strategies,
        existing_integration_resolver=resolver,
    )


def _facade_with(
    *,
    port: _RecordingAuthorizationPort,
    resolver: _RecordingResolver,
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
) -> ExistingCapabilityConfigurationRealizationFacade:
    service = _service_with(resolver=resolver, strategies=strategies)
    return ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=port,
        realization_service=service,
    )


def test_p1_06_allow_matching_evidence_reaches_core() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort()
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    result = facade.realize(request)
    assert port.authorize_calls == 1
    assert resolver.calls == 1
    assert strategy.realize_calls == 1
    assert result.tenant_id == request.tenant_id
    assert result.configuration_fingerprint == request.configuration_fingerprint


def test_p1_07_deny_core_calls_zero() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        permitted=False,
        decision=PolicyDecision(action=PolicyAction.DENY, reason="no"),
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED
    )
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


def test_p1_08_authorization_exception_core_calls_zero() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(raise_error=True)
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


def test_p1_09_forged_tenant_evidence_core_calls_zero() -> None:
    request = _request(tenant_id="tenant-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"tenant_id": "tenant-b"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )
    assert resolver.calls == 0


def test_p1_10_wrong_provider_resource_evidence_core_calls_zero() -> None:
    request = _request(provider_id="sqlite")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"resource_id": "postgres"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_p1_11_wrong_current_revision_core_calls_zero() -> None:
    request = _request(current_revision="rev-1")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"current_revision": "rev-stale"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_p1_12_wrong_target_revision_fingerprint_core_calls_zero() -> None:
    request = _request(configuration_fingerprint="fp-test-001")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"target_revision": "fp-forged"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_p1_13_wrong_request_digest_core_calls_zero() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"request_digest": "sha256:deadbeef"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_p1_14_task_run_mismatch_core_calls_zero() -> None:
    request = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"task_id": "task_wrong"},
    )
    facade = _facade_with(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_p1_15_exact_resolver_target_one_strategy_success() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    result = service.realize_admitted(request, authorization_evidence=evidence)
    assert result.provider_id == "sqlite"
    assert result.realization_evidence_refs == ("evidence://realized/1",)


def test_p1_16_missing_target_fail_closed_no_strategy() -> None:
    request = _request()
    resolver = _RecordingResolver(target=None)
    strategy = _StubStrategy()
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TARGET_NOT_FOUND
    )
    assert strategy.realize_calls == 0


def test_p1_17_zero_strategy_matches_unsupported() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(match=False)
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY
    )


def test_p1_18_two_strategies_match_ambiguity() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    service = _service_with(
        resolver=resolver,
        strategies=(
            _StubStrategy(strategy_id_value="a"),
            _StubStrategy(strategy_id_value="b"),
        ),
    )
    evidence = _allow_evidence_for(request)
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.STRATEGY_AMBIGUITY
    )


def test_p1_19_strategy_result_tenant_mismatch_reject() -> None:
    request = _request(tenant_id="tenant-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_tenant="tenant-b")
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_p1_20_strategy_result_provider_mismatch_reject() -> None:
    request = _request(provider_id="sqlite")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_provider="postgres")
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_p1_21_result_config_identity_continuity() -> None:
    request = _request(configuration_fingerprint="fp-test-001")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    service = _service_with(resolver=resolver, strategies=(strategy,))
    evidence = _allow_evidence_for(request)
    result = service.realize_admitted(request, authorization_evidence=evidence)
    assert result.configuration_type == request.configuration.configuration_type
    assert result.configuration_version == request.configuration.configuration_version
    assert result.configuration_fingerprint == request.configuration_fingerprint
    assert result.provider_id == request.provider_id


def test_p1_22_no_forbidden_imports_in_p1_modules() -> None:
    forbidden = (
        "intergrax.runtime.governance",
        "ToolRuntime",
        "Capability Acquisition",
        "Marketplace",
    )
    for path in _PRODUCTION_PATHS:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"getattr", "setattr", "hasattr"}:
                    pytest.fail(f"{path.name} uses {node.func.id}")


def test_tenant_a_principal_tenant_b_request_fails_before_governance() -> None:
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        _request(tenant_id="tenant-b", principal=_principal(tenant_id="tenant-a"))


def test_resolver_provider_mismatch_fail_closed() -> None:
    request = _request(provider_id="sqlite")
    wrong_target = ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id="postgres",
        current_revision=request.current_revision,
        binding=IntegrationBinding.from_manifest(SQLITE),
    )
    resolver = _RecordingResolver(target=wrong_target)
    service = _service_with(resolver=resolver, strategies=(_StubStrategy(),))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_strategy_replaceability_without_core_modification() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    service_a = _service_with(
        resolver=resolver, strategies=(_StubStrategy(strategy_id_value="a"),)
    )
    service_b = _service_with(
        resolver=resolver, strategies=(_StubStrategy(strategy_id_value="b"),)
    )
    evidence = _allow_evidence_for(request)
    assert (
        service_a.realize_admitted(request, authorization_evidence=evidence).provider_id
        == "sqlite"
    )
    assert (
        service_b.realize_admitted(request, authorization_evidence=evidence).provider_id
        == "sqlite"
    )


def test_governance_projection_constants() -> None:
    request = _request()
    projected = project_control_plane_mutation_request(request)
    assert projected.mutation_type == MUTATION_TYPE_INTEGRATION_CONFIGURATION_REALIZE_V1
    assert projected.resource_type == RESOURCE_TYPE_INTEGRATION_CONFIGURATION
    assert projected.resource_id == request.provider_id
    assert projected.target_revision == request.configuration_fingerprint


def test_facade_structurally_implements_authorization_port_injection() -> None:
    port = _RecordingAuthorizationPort()
    resolver = _RecordingResolver(target=_target_for(_request()))
    facade = _facade_with(port=port, resolver=resolver, strategies=(_StubStrategy(),))
    assert isinstance(port, ControlPlaneMutationAuthorizationPort)
    assert isinstance(facade, ExistingCapabilityConfigurationRealizationFacade)
