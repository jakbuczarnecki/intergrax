# © Artur Czarnecki. All rights reserved.

"""INT-CONFIG-REAL-X-CERT — adversarial configuration realization certification matrix."""

from __future__ import annotations

import ast
import inspect
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationPort,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationStrategy,
    ExistingCapabilityIntegrationTarget,
    IntegrationConfigurationPayload,
    project_control_plane_mutation_request,
    verify_admitted_authorization_evidence,
)
from intergrax.integrations.existing_capability_configuration_facade import (
    ExistingCapabilityConfigurationRealizationFacade,
)
from intergrax.integrations.existing_capability_configuration_service import (
    ExistingCapabilityConfigurationRealizationService,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
    SQLiteRelationalStoreConfigurationPayload,
    SQLiteRelationalStoreConfigurationRealizationStrategy,
    compute_sqlite_relational_store_configuration_fingerprint,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    ENV_SQLITE_DATA_DIR,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLOSED_WORLD_PATHS = (
    _REPO_ROOT
    / "intergrax/integrations/contracts/existing_capability_configuration.py",
    _REPO_ROOT / "intergrax/integrations/existing_capability_configuration_service.py",
    _REPO_ROOT / "intergrax/integrations/existing_capability_configuration_facade.py",
    _REPO_ROOT
    / "intergrax/integrations/providers/relational_store/sqlite/configuration_realization.py",
)
_GENERIC_P1_PATHS = _CLOSED_WORLD_PATHS[:3]
_TASK_ID = TaskId("task_01234567890123456789012345678901")
_RUN_ID = RunId("run_01234567890123456789012345678901")


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


def _sqlite_payload(
    *,
    data_dir: Path = Path("/data/tenant-a/store"),
    relational_db: Path | None = Path("/data/tenant-a/store/app.db"),
) -> SQLiteRelationalStoreConfigurationPayload:
    return SQLiteRelationalStoreConfigurationPayload(
        data_dir=data_dir,
        relational_db=relational_db,
    )


def _request(
    *,
    tenant_id: str = "tenant-a",
    principal: RequestIdentity | None = None,
    provider_id: str = SQLITE_RELATIONAL_STORE_PROVIDER_ID,
    category: IntegrationCategory = IntegrationCategory.RELATIONAL_STORE,
    current_revision: str = "rev-2",
    configuration_fingerprint: str | None = None,
    resource_scope: str = "scope-tenant-a-relational",
    configuration: IntegrationConfigurationPayload | None = None,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
) -> ExistingCapabilityConfigurationRealizationRequest:
    payload = configuration or _sqlite_payload()
    fp = configuration_fingerprint or payload.configuration_fingerprint
    return ExistingCapabilityConfigurationRealizationRequest(
        request_id="cert-req-1",
        tenant_id=tenant_id,
        principal=principal or _principal(tenant_id=tenant_id),
        integration_category=category,
        provider_id=provider_id,
        resource_scope=resource_scope,
        configuration=payload,
        configuration_fingerprint=fp,
        current_revision=current_revision,
        risk_classification=ControlPlaneMutationRisk.LOW,
        task_id=task_id,
        run_id=run_id,
    )


def _target_for(
    request: ExistingCapabilityConfigurationRealizationRequest,
    *,
    tenant_id: str | None = None,
    integration_category: IntegrationCategory | None = None,
    provider_id: str | None = None,
    resource_scope: str | None = None,
    current_revision: str | None = None,
) -> ExistingCapabilityIntegrationTarget:
    return ExistingCapabilityIntegrationTarget(
        tenant_id=tenant_id if tenant_id is not None else request.tenant_id,
        integration_category=(
            integration_category
            if integration_category is not None
            else request.integration_category
        ),
        provider_id=provider_id if provider_id is not None else request.provider_id,
        resource_scope=(
            resource_scope if resource_scope is not None else request.resource_scope
        ),
        current_revision=(
            current_revision
            if current_revision is not None
            else request.current_revision
        ),
    )


@dataclass
class _RecordingResolver:
    target: ExistingCapabilityIntegrationTarget | None = None
    calls: int = 0

    def resolve_existing_integration(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityIntegrationTarget:
        self.calls += 1
        assert self.target is not None
        return self.target


@dataclass
class _StubStrategy:
    strategy_id_value: str = "cert-stub-strategy"
    match: bool = True
    realize_calls: int = 0
    realize_error: Exception | None = None
    output_tenant: str | None = None
    output_provider: str | None = None
    output_category: IntegrationCategory | None = None
    output_scope: str | None = None
    output_fingerprint: str | None = None

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
        if self.realize_error is not None:
            raise self.realize_error
        return ConfiguredCapabilityBinding(
            tenant_id=self.output_tenant or request.tenant_id,
            integration_category=self.output_category or request.integration_category,
            provider_id=self.output_provider or request.provider_id,
            resource_scope=self.output_scope or request.resource_scope,
            configuration_type=request.configuration.configuration_type,
            configuration_version=request.configuration.configuration_version,
            configuration_fingerprint=self.output_fingerprint
            or request.configuration_fingerprint,
            realization_evidence_refs=("audit://configured-only",),
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
            raise RuntimeError("authorization port failure")
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


def _facade_stack(
    *,
    port: ControlPlaneMutationAuthorizationPort,
    resolver: _RecordingResolver,
    strategies: tuple[ExistingCapabilityConfigurationRealizationStrategy, ...],
) -> ExistingCapabilityConfigurationRealizationFacade:
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=strategies,
        existing_integration_resolver=resolver,
    )
    return ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=port,
        realization_service=service,
    )


@dataclass
class _StaleEvidenceReplayPort:
    evidence: ControlPlaneMutationAuthorizationEvidence

    def authorize(
        self,
        request: ControlPlaneMutationRequest,
    ) -> ControlPlaneMutationAuthorizationResult:
        decision = PolicyDecision(action=PolicyAction.ALLOW, reason="replay")
        return ControlPlaneMutationAuthorizationResult(
            permitted=True,
            decision=decision,
            evidence=self.evidence,
        )


def _assert_fail_closed(
    exc_info: pytest.ExceptionInfo[ExistingCapabilityConfigurationRealizationError],
    *,
    reason: ExistingCapabilityConfigurationRealizationFailureReason,
    resolver: _RecordingResolver,
    strategy: _StubStrategy,
) -> None:
    assert exc_info.value.reason is reason
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


# --- Authority provenance (CERT-A*) ---


def test_cert_a01_no_public_realize_with_caller_supplied_evidence() -> None:
    port_sig = inspect.signature(ExistingCapabilityConfigurationRealizationPort.realize)
    assert list(port_sig.parameters) == ["self", "request"]
    facade_sig = inspect.signature(
        ExistingCapabilityConfigurationRealizationFacade.realize
    )
    assert list(facade_sig.parameters) == ["self", "request"]
    assert not hasattr(
        ExistingCapabilityConfigurationRealizationFacade, "realize_admitted"
    )


def test_cert_a01_handcrafted_evidence_cannot_admit_via_public_facade() -> None:
    request = _request()
    evidence = _allow_evidence_for(request)
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        permitted=False,
        decision=PolicyDecision(action=PolicyAction.DENY, reason="no"),
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert port.authorize_calls == 1
    assert resolver.calls == 0
    assert strategy.realize_calls == 0
    verify_admitted_authorization_evidence(
        request=request,
        governance_request=project_control_plane_mutation_request(request),
        evidence=evidence,
    )


def test_cert_a02_authorization_deny_zero_downstream() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        permitted=False,
        decision=PolicyDecision(action=PolicyAction.DENY, reason="deny"),
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    _assert_fail_closed(
        exc,
        reason=ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
        resolver=resolver,
        strategy=strategy,
    )


@pytest.mark.parametrize(
    "action",
    (
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.ESCALATE,
        PolicyAction.MODIFY,
        PolicyAction.DENY,
    ),
)
def test_cert_a03_permitted_true_but_action_not_allow(
    action: PolicyAction,
) -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        permitted=True,
        decision=PolicyDecision(action=action, reason="not-allow"),
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    _assert_fail_closed(
        exc,
        reason=ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED,
        resolver=resolver,
        strategy=strategy,
    )


def test_cert_a04_authorization_port_raises_fail_closed() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(raise_error=True)
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.AUTHORIZATION_REJECTED
    )
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


# --- Tenant matrix (CERT-T*) ---


def test_cert_t01_principal_tenant_mismatch_before_governance() -> None:
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        _request(tenant_id="tenant-b", principal=_principal(tenant_id="tenant-a"))
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_cert_t02_evidence_tenant_mismatch_before_core() -> None:
    request = _request(tenant_id="tenant-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(evidence_overrides={"tenant_id": "tenant-b"})
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        facade.realize(request)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


def test_cert_t03_resolver_target_tenant_mismatch_before_strategy() -> None:
    request = _request(tenant_id="tenant-a")
    wrong = _target_for(request, tenant_id="tenant-b")
    resolver = _RecordingResolver(target=wrong)
    strategy = _StubStrategy()
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )
    assert strategy.realize_calls == 0


def test_cert_t04_strategy_output_tenant_widening_rejected() -> None:
    request = _request(tenant_id="tenant-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_tenant="tenant-b")
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


@pytest.mark.parametrize(
    "tenant_id,expectation",
    (
        ("", ValidationError),
        (
            "   ",
            ExistingCapabilityConfigurationRealizationError,
        ),
    ),
)
def test_cert_t05_missing_blank_tenant_fail_closed(
    tenant_id: str,
    expectation: type[BaseException],
) -> None:
    if expectation is ValidationError:
        with pytest.raises(ValidationError):
            _request(tenant_id=tenant_id)
        return
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        ExistingCapabilityConfigurationRealizationRequest(
            request_id="cert-req-blank-tenant",
            tenant_id=tenant_id,
            principal=_principal(tenant_id="tenant-a"),
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
            resource_scope="scope-a",
            configuration=_sqlite_payload(),
            configuration_fingerprint=_sqlite_payload().configuration_fingerprint,
            current_revision="rev-1",
            risk_classification=ControlPlaneMutationRisk.LOW,
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_cert_t06_same_config_fingerprint_distinct_tenant_identities() -> None:
    req_a = _request(tenant_id="tenant-a", resource_scope="scope-shared")
    req_b = _request(tenant_id="tenant-b", resource_scope="scope-shared")
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    bind_a = strategy.realize(req_a, _target_for(req_a))
    bind_b = strategy.realize(req_b, _target_for(req_b))
    assert bind_a.configuration_fingerprint == bind_b.configuration_fingerprint
    assert bind_a.tenant_id == "tenant-a"
    assert bind_b.tenant_id == "tenant-b"


# --- Provider / category (CERT-P*) ---


def test_cert_p01_evidence_provider_mismatch_before_core() -> None:
    request = _request(provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID)
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(evidence_overrides={"resource_id": "postgres"})
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_cert_p02_resolver_target_provider_mismatch_before_strategy() -> None:
    request = _request()
    wrong = _target_for(request, provider_id="postgres")
    resolver = _RecordingResolver(target=wrong)
    strategy = _StubStrategy()
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )
    assert strategy.realize_calls == 0


def test_cert_p03_strategy_output_provider_mismatch_rejected() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_provider="postgres")
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_cert_p04_category_mismatch_fail_closed() -> None:
    request = _request(category=IntegrationCategory.RELATIONAL_STORE)
    wrong = _target_for(
        request, integration_category=IntegrationCategory.DOCUMENT_STORE
    )
    resolver = _RecordingResolver(target=wrong)
    strategy = _StubStrategy()
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )
    assert strategy.realize_calls == 0


# --- Resource scope (CERT-S*) ---


def test_cert_s01_authorization_scope_mismatch_before_core() -> None:
    request = _request(resource_scope="scope-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"resource_scope": "scope-b"},
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_cert_s02_resolver_target_scope_mismatch_before_strategy() -> None:
    request = _request(resource_scope="scope-a")
    wrong = _target_for(request, resource_scope="scope-b")
    resolver = _RecordingResolver(target=wrong)
    strategy = _StubStrategy()
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert strategy.realize_calls == 0


def test_cert_s03_strategy_output_scope_mismatch_rejected() -> None:
    request = _request(resource_scope="scope-a")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_scope="scope-b")
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_cert_s04_sqlite_path_does_not_redefine_resource_scope() -> None:
    scope = "scope-platform-tenant-a"
    payload = _sqlite_payload(
        data_dir=Path("/other/tenant/b/data"),
        relational_db=Path("/other/tenant/b/data/db.sqlite"),
    )
    request = _request(resource_scope=scope, configuration=payload)
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    binding = strategy.realize(request, _target_for(request))
    assert binding.resource_scope == scope
    assert binding.resource_scope != payload.data_dir.as_posix()


# --- Revision / configuration (CERT-R*) ---


def test_cert_r01_stale_current_revision_in_evidence() -> None:
    request = _request(current_revision="rev-2")
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"current_revision": "rev-1"},
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_cert_r02_resolver_target_stale_current_revision() -> None:
    request = _request(current_revision="rev-2")
    wrong = _target_for(request, current_revision="rev-1")
    resolver = _RecordingResolver(target=wrong)
    strategy = _StubStrategy()
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert strategy.realize_calls == 0


def test_cert_r03_wrong_target_revision_in_evidence() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"target_revision": "sha256:forged"},
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


def test_cert_r04_request_fingerprint_differs_from_payload() -> None:
    payload = _sqlite_payload()
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        _request(configuration=payload, configuration_fingerprint="not-the-payload-fp")
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_cert_r05_strategy_output_fingerprint_mismatch() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(output_fingerprint="sha256:wrong")
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH
    )


def test_cert_r06_unsupported_configuration_version_no_fallback() -> None:
    bad = _TestConfigurationPayload(
        SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
        "99",
        "fp-version-99",
    )
    request = _request(
        configuration=bad,
        configuration_fingerprint="fp-version-99",
    )
    resolver = _RecordingResolver(target=_target_for(request))
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY
    )


def test_cert_r07_wrong_configuration_type_no_sqlite_match() -> None:
    alt = _TestConfigurationPayload("wrong.provider.type", "1", "fp-wrong-type")
    request = _request(configuration=alt, configuration_fingerprint="fp-wrong-type")
    resolver = _RecordingResolver(target=_target_for(request))
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY
    )


# --- Digest / identity (CERT-D*) ---


def test_cert_d01_forged_request_digest_before_core() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"request_digest": "sha256:deadbeefdeadbeef"},
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


@pytest.mark.parametrize(
    "mutator",
    (
        lambda req: _request(
            tenant_id=req.tenant_id,
            provider_id="postgres",
            resource_scope=req.resource_scope,
            current_revision=req.current_revision,
            configuration=req.configuration,  # type: ignore[arg-type]
        ),
        lambda req: _request(
            tenant_id=req.tenant_id,
            provider_id=req.provider_id,
            resource_scope="scope-mutated",
            current_revision=req.current_revision,
            configuration=req.configuration,  # type: ignore[arg-type]
        ),
        lambda req: _request(
            tenant_id=req.tenant_id,
            provider_id=req.provider_id,
            resource_scope=req.resource_scope,
            current_revision="rev-mutated",
            configuration=req.configuration,  # type: ignore[arg-type]
        ),
        lambda req: _request(
            configuration=_sqlite_payload(data_dir=Path("/mutated/configuration/path")),
        ),
        lambda req: _request(
            tenant_id=req.tenant_id,
            provider_id=req.provider_id,
            resource_scope=req.resource_scope,
            current_revision=req.current_revision,
            configuration=req.configuration,  # type: ignore[arg-type]
            task_id=_TASK_ID,
            run_id=RunId("run_99999999999999999999999999999999"),
        ),
    ),
)
def test_cert_d02_evidence_from_request_a_cannot_authorize_mutated_request_b(
    mutator: object,
) -> None:
    base = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    evidence_a = _allow_evidence_for(base)
    mutated = mutator(base)  # type: ignore[operator]
    resolver = _RecordingResolver(target=_target_for(mutated))
    strategy = _StubStrategy()
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        verify_admitted_authorization_evidence(
            request=mutated,
            governance_request=project_control_plane_mutation_request(mutated),
            evidence=evidence_a,
        )
    facade = _facade_stack(
        port=_StaleEvidenceReplayPort(evidence=evidence_a),
        resolver=resolver,
        strategies=(strategy,),
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(mutated)
    assert resolver.calls == 0
    assert strategy.realize_calls == 0


def test_cert_d03_task_run_mismatch_before_core() -> None:
    request = _request(task_id=_TASK_ID, run_id=_RUN_ID)
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy()
    port = _RecordingAuthorizationPort(
        evidence_overrides={"task_id": "task_wrong"},
    )
    facade = _facade_stack(port=port, resolver=resolver, strategies=(strategy,))
    with pytest.raises(ExistingCapabilityConfigurationRealizationError):
        facade.realize(request)
    assert resolver.calls == 0


# --- Strategy selection (CERT-ST*) ---


def test_cert_st01_zero_matching_strategies() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(match=False)
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY
    )


def test_cert_st02_two_strategies_ambiguity() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(
            _StubStrategy(strategy_id_value="a"),
            _StubStrategy(strategy_id_value="b"),
        ),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.STRATEGY_AMBIGUITY
    )


def test_cert_st03_strategy_realize_raises_maps_realization_failed() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    strategy = _StubStrategy(realize_error=RuntimeError("provider edge failure"))
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(strategy,),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.REALIZATION_FAILED
    )


@pytest.mark.parametrize(
    "mutator,expected_reason",
    (
        (
            lambda req, tgt: (req, _target_for(req, tenant_id="tenant-b")),
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
        ),
        (
            lambda req, tgt: (req, _target_for(req, provider_id="postgres")),
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
        ),
        (
            lambda req, tgt: (
                req,
                _target_for(
                    req, integration_category=IntegrationCategory.DOCUMENT_STORE
                ),
            ),
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
        ),
        (
            lambda req, tgt: (req, _target_for(req, resource_scope="scope-other")),
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
        ),
    ),
)
def test_cert_st04_sqlite_direct_strategy_fail_closed(
    mutator: object,
    expected_reason: ExistingCapabilityConfigurationRealizationFailureReason,
) -> None:
    request = _request()
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    req2, target = mutator(request, _target_for(request))  # type: ignore[operator]
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        strategy.realize(req2, target)
    assert exc.value.reason is expected_reason


# --- Ambient configuration (CERT-CFG*) ---


def test_cert_cfg01_hostile_env_cannot_override_explicit_sqlite_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_SQLITE_DATA_DIR, "/env/hostile/data")
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", "/env/hostile/db.sqlite")
    payload = _sqlite_payload(
        data_dir=Path("/explicit/tenant/data"),
        relational_db=Path("/explicit/tenant/db.sqlite"),
    )
    expected = compute_sqlite_relational_store_configuration_fingerprint(
        data_dir=payload.data_dir,
        relational_db=payload.relational_db,
    )
    assert payload.configuration_fingerprint == expected


def test_cert_cfg02_fingerprint_stable_after_environment_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _sqlite_payload()
    fp_before = payload.configuration_fingerprint
    monkeypatch.setenv(ENV_SQLITE_DATA_DIR, "/changed/after/payload")
    assert payload.configuration_fingerprint == fp_before


def test_cert_cfg03_filesystem_state_does_not_change_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        db_path = data_dir / "app.db"
        payload = _sqlite_payload(data_dir=data_dir, relational_db=db_path)
        fp_before = payload.configuration_fingerprint
        data_dir.mkdir(parents=True, exist_ok=True)
        db_path.write_text("sqlite", encoding="utf-8")
        assert payload.configuration_fingerprint == fp_before
        db_path.unlink()
        data_dir.rmdir()
        assert payload.configuration_fingerprint == fp_before


def test_cert_cfg04_no_implicit_default_sqlite_provider() -> None:
    alt = _TestConfigurationPayload("generic.config", "1", "fp-generic")
    request = _request(
        provider_id="unknown-provider",
        configuration=alt,
        configuration_fingerprint="fp-generic",
    )
    resolver = _RecordingResolver(target=_target_for(request))
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=resolver,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(
            request, authorization_evidence=_allow_evidence_for(request)
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY
    )


# --- Configured != effective (CERT-L*) ---


def test_cert_l01_governed_success_does_not_create_database_files() -> None:
    ghost = Path("/tmp/intergrax-cert-ghost-never-created")
    payload = _sqlite_payload(data_dir=ghost, relational_db=None)
    request = _request(configuration=payload)
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_RecordingResolver(target=_target_for(request)),
    )
    facade = ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=_RecordingAuthorizationPort(),
        realization_service=service,
    )
    result = facade.realize(request)
    assert (
        result.configured_binding.configuration_fingerprint
        == payload.configuration_fingerprint
    )
    assert not ghost.exists()


def test_cert_l02_provider_factory_not_invoked_on_realization() -> None:
    request = _request()
    bundle = "intergrax.integrations.providers.relational_store.sqlite.bundle"
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    with patch(f"{bundle}.create_sqlite_relational_store") as factory_mock:
        with patch(f"{bundle}.SQLiteRelationalStoreFactory") as factory_cls_mock:
            strategy.realize(request, _target_for(request))
    factory_mock.assert_not_called()
    factory_cls_mock.assert_not_called()


def test_cert_l03_strategy_module_has_no_execution_runtime_imports() -> None:
    path = _CLOSED_WORLD_PATHS[3]
    source = path.read_text(encoding="utf-8")
    for token in (
        "ToolRuntime",
        "intergrax.runtime.execution",
        "intergrax.runtime.governance",
    ):
        assert token not in source


def test_cert_l04_configured_binding_has_no_effective_execution_claim_fields() -> None:
    forbidden = (
        "effective",
        "executed",
        "active",
        "authorized_for_use",
    )
    for field_info in fields(ConfiguredCapabilityBinding):
        name = field_info.name.lower()
        for token in forbidden:
            assert token not in name


# --- Static architecture gates (§16–21) ---


def test_cert_static_weak_boundary_scan() -> None:
    forbidden_tokens = (
        "IntegrationBinding",
        "dict[str, Any]",
        "dict[str, object]",
    )
    for path in _CLOSED_WORLD_PATHS:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"getattr", "setattr", "hasattr"}


def test_cert_static_generic_core_vendor_agnostic() -> None:
    for path in _GENERIC_P1_PATHS:
        assert "sqlite" not in path.read_text(encoding="utf-8").lower()


def test_cert_static_governance_execution_separation() -> None:
    forbidden = (
        "intergrax.runtime.governance",
        "ControlPlaneMutationAuthorizationBoundary",
        "MeaningfulSideEffectAuthorizationPort",
        "ToolRuntime",
        "intergrax.runtime.execution",
    )
    for path in _CLOSED_WORLD_PATHS[:3]:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source
    facade_source = _CLOSED_WORLD_PATHS[2].read_text(encoding="utf-8")
    assert "ControlPlaneMutationAuthorizationPort" in facade_source


def test_cert_static_no_duplicate_realization_authority_in_closed_world() -> None:
    integrations_root = _REPO_ROOT / "intergrax/integrations"
    facade_defs = list(integrations_root.rglob("*.py"))
    facade_count = sum(
        1
        for path in facade_defs
        if "class ExistingCapabilityConfigurationRealizationFacade"
        in path.read_text(encoding="utf-8")
    )
    assert facade_count == 1
    service_count = sum(
        1
        for path in facade_defs
        if "class ExistingCapabilityConfigurationRealizationService"
        in path.read_text(encoding="utf-8")
    )
    assert service_count == 1


def test_cert_public_api_classification() -> None:
    assert issubclass(
        ExistingCapabilityConfigurationRealizationFacade,
        ExistingCapabilityConfigurationRealizationPort,
    )
    service_module = inspect.getmodule(
        ExistingCapabilityConfigurationRealizationService
    )
    assert service_module is not None
    assert "ExistingCapabilityConfigurationRealizationPort" not in dir(service_module)


def test_cert_sanctioned_governance_admission_path_reaches_binding() -> None:
    request = _request()
    resolver = _RecordingResolver(target=_target_for(request))
    facade = _facade_stack(
        port=_RecordingAuthorizationPort(),
        resolver=resolver,
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
    )
    result = facade.realize(request)
    assert isinstance(result.configured_binding, ConfiguredCapabilityBinding)
    assert result.authorization_evidence.tenant_id == request.tenant_id
