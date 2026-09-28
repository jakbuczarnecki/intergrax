# © Artur Czarnecki. All rights reserved.

"""INT-CONFIG-REAL-X-P2 SQLite configuration realization qualification."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationRisk,
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationStrategy,
    ExistingCapabilityIntegrationTarget,
    IntegrationConfigurationPayload,
    project_control_plane_mutation_request,
)
from intergrax.integrations.existing_capability_configuration_facade import (
    ExistingCapabilityConfigurationRealizationFacade,
)
from intergrax.integrations.existing_capability_configuration_service import (
    ExistingCapabilityConfigurationRealizationService,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLITE_RELATIONAL_STORE_CONFIGURATION_STRATEGY_ID,
    SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
    SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION,
    SQLiteRelationalStoreConfigurationPayload,
    SQLiteRelationalStoreConfigurationRealizationStrategy,
    compute_sqlite_relational_store_configuration_fingerprint,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    ENV_SQLITE_DATA_DIR,
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[6]
_GENERIC_P1_PATHS = (
    _REPO_ROOT
    / "intergrax/integrations/contracts/existing_capability_configuration.py",
    _REPO_ROOT / "intergrax/integrations/existing_capability_configuration_service.py",
    _REPO_ROOT / "intergrax/integrations/existing_capability_configuration_facade.py",
)
_STRATEGY_MODULE = (
    _REPO_ROOT
    / "intergrax/integrations/providers/relational_store/sqlite/configuration_realization.py"
)


def _principal(tenant_id: str = "tenant-a") -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )


def _sqlite_payload(
    *,
    data_dir: Path = Path("/data/tenant/store"),
    relational_db: Path | None = Path("/data/tenant/store/custom.db"),
) -> SQLiteRelationalStoreConfigurationPayload:
    return SQLiteRelationalStoreConfigurationPayload(
        data_dir=data_dir,
        relational_db=relational_db,
    )


def _request_for(
    payload: SQLiteRelationalStoreConfigurationPayload,
    *,
    tenant_id: str = "tenant-a",
    resource_scope: str = "scope-relational-1",
    current_revision: str = "rev-1",
) -> ExistingCapabilityConfigurationRealizationRequest:
    return ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-sqlite-p2",
        tenant_id=tenant_id,
        principal=_principal(tenant_id=tenant_id),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope=resource_scope,
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision=current_revision,
        risk_classification=ControlPlaneMutationRisk.LOW,
    )


def _target_for(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> ExistingCapabilityIntegrationTarget:
    return ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id=request.provider_id,
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )


@dataclass
class _StaticResolver:
    target: ExistingCapabilityIntegrationTarget

    def resolve_existing_integration(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
    ) -> ExistingCapabilityIntegrationTarget:
        return self.target


@dataclass
class _AllowAuthorizationPort:
    def authorize(self, request):  # noqa: ANN001
        from intergrax.contracts.control_plane_mutation import (
            ControlPlaneMutationAuthorizationResult,
        )

        digest = control_plane_mutation_request_digest(request)
        decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
        evidence = evidence_from_request_and_decision(
            request,
            decision=decision,
            request_digest=digest,
        )
        return ControlPlaneMutationAuthorizationResult(
            permitted=True,
            decision=decision,
            evidence=evidence,
        )


@dataclass(frozen=True)
class _AlternateTestPayload:
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


@dataclass
class _AlternateTestStrategy:
    strategy_id_value: str = "alternate.test.configuration.v1"

    @property
    def strategy_id(self) -> str:
        return self.strategy_id_value

    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool:
        return (
            request.configuration.configuration_type == "alternate.test.configuration"
            and request.configuration.configuration_version == "1"
        )

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> ConfiguredCapabilityBinding:
        return ConfiguredCapabilityBinding(
            tenant_id=request.tenant_id,
            integration_category=request.integration_category,
            provider_id=request.provider_id,
            resource_scope=request.resource_scope,
            configuration_type=request.configuration.configuration_type,
            configuration_version=request.configuration.configuration_version,
            configuration_fingerprint=request.configuration_fingerprint,
            realization_evidence_refs=(),
        )


def test_p2_01_payload_satisfies_integration_configuration_payload() -> None:
    payload = _sqlite_payload()
    assert isinstance(payload, IntegrationConfigurationPayload)
    assert payload.configuration_type == SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE
    assert (
        payload.configuration_version == SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION
    )
    assert len(payload.configuration_fingerprint) == 64


def test_p2_02_fingerprint_deterministic_for_equal_configuration() -> None:
    a = _sqlite_payload()
    b = _sqlite_payload()
    assert a.configuration_fingerprint == b.configuration_fingerprint
    assert (
        compute_sqlite_relational_store_configuration_fingerprint(
            data_dir=a.data_dir,
            relational_db=a.relational_db,
        )
        == a.configuration_fingerprint
    )


def test_p2_03_fingerprint_changes_when_relational_config_changes() -> None:
    base = _sqlite_payload()
    changed = _sqlite_payload(relational_db=Path("/data/tenant/store/other.db"))
    assert base.configuration_fingerprint != changed.configuration_fingerprint


def test_p2_04_fingerprint_independent_of_sqlite_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_SQLITE_DATA_DIR, "/env/override/data")
    monkeypatch.setenv("INTERGRAX_RELATIONAL_DB", "/env/override/db.sqlite")
    payload = _sqlite_payload(
        data_dir=Path("/explicit/data"),
        relational_db=Path("/explicit/relational.db"),
    )
    expected = compute_sqlite_relational_store_configuration_fingerprint(
        data_dir=Path("/explicit/data"),
        relational_db=Path("/explicit/relational.db"),
    )
    assert payload.configuration_fingerprint == expected


def test_p2_05_fingerprint_has_zero_filesystem_side_effects() -> None:
    data_dir = Path("/no/fs/read/data")
    relational_db = Path("/no/fs/read/db.sqlite")
    with patch.object(Path, "resolve", MagicMock()) as resolve_mock:
        with patch.object(Path, "exists", MagicMock()) as exists_mock:
            with patch.object(Path, "stat", MagicMock()) as stat_mock:
                fingerprint = compute_sqlite_relational_store_configuration_fingerprint(
                    data_dir=data_dir,
                    relational_db=relational_db,
                )
    assert len(fingerprint) == 64
    resolve_mock.assert_not_called()
    exists_mock.assert_not_called()
    stat_mock.assert_not_called()


def test_p2_06_exact_relational_store_sqlite_v1_matches() -> None:
    request = _request_for(_sqlite_payload())
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    assert strategy.can_realize(request, _target_for(request))


def test_p2_07_different_provider_does_not_match() -> None:
    payload = _sqlite_payload()
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-2",
        tenant_id="tenant-a",
        principal=_principal(),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="postgres",
        resource_scope="scope-1",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    target = ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id="postgres",
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )
    assert not strategy.can_realize(request, target)


def test_p2_08_different_category_does_not_match() -> None:
    payload = _sqlite_payload()
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-3",
        tenant_id="tenant-a",
        principal=_principal(),
        integration_category=IntegrationCategory.DOCUMENT_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-1",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    target = ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=IntegrationCategory.DOCUMENT_STORE,
        provider_id=request.provider_id,
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )
    assert not strategy.can_realize(request, target)


def test_p2_09_unsupported_configuration_type_does_not_match() -> None:
    alt = _AlternateTestPayload("wrong.type", "1", "fp-alt")
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-4",
        tenant_id="tenant-a",
        principal=_principal(),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-1",
        configuration=alt,
        configuration_fingerprint="fp-alt",
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    assert not strategy.can_realize(request, _target_for(request))


def test_p2_10_sqlite_strategy_through_generic_service_succeeds() -> None:
    request = _request_for(_sqlite_payload())
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(_target_for(request)),
    )
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    result = service.realize_admitted(request, authorization_evidence=evidence)
    assert result.configured_binding.provider_id == SQLITE_RELATIONAL_STORE_PROVIDER_ID


def test_p2_11_result_preserves_tenant_provider_category_scope_fingerprint() -> None:
    payload = _sqlite_payload()
    request = _request_for(
        payload,
        tenant_id="tenant-z",
        resource_scope="scope-unique",
    )
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    binding = strategy.realize(request, _target_for(request))
    assert binding.tenant_id == "tenant-z"
    assert binding.integration_category is IntegrationCategory.RELATIONAL_STORE
    assert binding.provider_id == SQLITE_RELATIONAL_STORE_PROVIDER_ID
    assert binding.resource_scope == "scope-unique"
    assert binding.configuration_fingerprint == payload.configuration_fingerprint
    assert binding.configuration_type == SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE
    assert (
        binding.configuration_version == SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION
    )


def test_p2_12_no_provider_activation_or_filesystem_materialization() -> None:
    ghost_dir = Path("/tmp/intergrax-p2-ghost-never-created")
    payload = _sqlite_payload(data_dir=ghost_dir, relational_db=None)
    request = _request_for(payload)
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    bundle = "intergrax.integrations.providers.relational_store.sqlite.bundle"
    with patch(f"{bundle}.create_sqlite_relational_store") as factory_mock:
        with patch(f"{bundle}.SQLiteRelationalStoreFactory") as factory_cls_mock:
            binding = strategy.realize(request, _target_for(request))
    factory_mock.assert_not_called()
    factory_cls_mock.assert_not_called()
    assert binding.configuration_fingerprint == payload.configuration_fingerprint
    assert not ghost_dir.exists()


def test_p2_13_invalid_sqlite_payload_fails_typed() -> None:
    payload = _sqlite_payload()
    request = _request_for(payload)
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    with patch.object(
        SQLiteIntegrationConfig,
        "__init__",
        side_effect=ValidationError.from_exception_data("SQLiteIntegrationConfig", []),
    ):
        with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
            strategy.realize(request, _target_for(request))
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.INVALID_CONFIGURATION
    )


def test_p2_14_tenant_a_request_cannot_consume_tenant_b_target() -> None:
    request = _request_for(_sqlite_payload(), tenant_id="tenant-a")
    wrong_target = ExistingCapabilityIntegrationTarget(
        tenant_id="tenant-b",
        integration_category=request.integration_category,
        provider_id=request.provider_id,
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(wrong_target),
    )
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_p2_15_strategy_cannot_widen_tenant() -> None:
    request = _request_for(_sqlite_payload(), tenant_id="tenant-a")
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    target = _target_for(request)
    with patch.object(
        SQLiteRelationalStoreConfigurationRealizationStrategy,
        "realize",
        return_value=ConfiguredCapabilityBinding(
            tenant_id="tenant-b",
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
            resource_scope=request.resource_scope,
            configuration_type=SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
            configuration_version=SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION,
            configuration_fingerprint=request.configuration_fingerprint,
        ),
    ):
        service = ExistingCapabilityConfigurationRealizationService(
            strategies=(strategy,),
            existing_integration_resolver=_StaticResolver(target),
        )
        governance = project_control_plane_mutation_request(request)
        digest = control_plane_mutation_request_digest(governance)
        decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
        evidence = evidence_from_request_and_decision(
            governance,
            decision=decision,
            request_digest=digest,
        )
        with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
            service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_p2_16_same_config_distinct_tenant_binding_identity() -> None:
    payload = _sqlite_payload()
    req_a = _request_for(payload, tenant_id="tenant-a", resource_scope="scope")
    req_b = _request_for(payload, tenant_id="tenant-b", resource_scope="scope")
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    bind_a = strategy.realize(req_a, _target_for(req_a))
    bind_b = strategy.realize(req_b, _target_for(req_b))
    assert bind_a.configuration_fingerprint == bind_b.configuration_fingerprint
    assert bind_a.tenant_id != bind_b.tenant_id


def test_p2_17_sqlite_strategy_structurally_satisfies_spi() -> None:
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    assert isinstance(strategy, ExistingCapabilityConfigurationRealizationStrategy)
    assert strategy.strategy_id == SQLITE_RELATIONAL_STORE_CONFIGURATION_STRATEGY_ID


def test_p2_18_alternate_test_strategy_satisfies_spi() -> None:
    assert isinstance(
        _AlternateTestStrategy(), ExistingCapabilityConfigurationRealizationStrategy
    )


def test_p2_19_service_swaps_implementations_without_core_modification() -> None:
    sqlite_request = _request_for(_sqlite_payload())
    alt_fp = "fp-alternate-test"
    alt_request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-alt",
        tenant_id="tenant-a",
        principal=_principal(),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-alt",
        configuration=_AlternateTestPayload(
            "alternate.test.configuration", "1", alt_fp
        ),
        configuration_fingerprint=alt_fp,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    sqlite_service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(_target_for(sqlite_request)),
    )
    alt_service = ExistingCapabilityConfigurationRealizationService(
        strategies=(_AlternateTestStrategy(),),
        existing_integration_resolver=_StaticResolver(_target_for(alt_request)),
    )
    governance = project_control_plane_mutation_request(sqlite_request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    sqlite_binding = sqlite_service.realize_admitted(
        sqlite_request, authorization_evidence=evidence
    ).configured_binding
    alt_governance = project_control_plane_mutation_request(alt_request)
    alt_digest = control_plane_mutation_request_digest(alt_governance)
    alt_evidence = evidence_from_request_and_decision(
        alt_governance,
        decision=decision,
        request_digest=alt_digest,
    )
    alt_binding = alt_service.realize_admitted(
        alt_request, authorization_evidence=alt_evidence
    ).configured_binding
    assert (
        sqlite_binding.configuration_type == SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE
    )
    assert alt_binding.configuration_type == "alternate.test.configuration"


def test_p2_20_two_matching_strategies_remain_ambiguity() -> None:
    request = _request_for(_sqlite_payload())
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(
            SQLiteRelationalStoreConfigurationRealizationStrategy(),
            SQLiteRelationalStoreConfigurationRealizationStrategy(),
        ),
        existing_integration_resolver=_StaticResolver(_target_for(request)),
    )
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.STRATEGY_AMBIGUITY
    )


def test_p2_21_generic_p1_modules_contain_no_sqlite_branch() -> None:
    for path in _GENERIC_P1_PATHS:
        source = path.read_text(encoding="utf-8").lower()
        assert "sqlite" not in source


def test_p2_22_sqlite_strategy_has_no_governance_execution_imports() -> None:
    source = _STRATEGY_MODULE.read_text(encoding="utf-8")
    forbidden = (
        "ControlPlaneMutationAuthorizationPort",
        "ControlPlaneMutationPolicyEvaluator",
        "ControlPlaneMutationAuthorizationBoundary",
        "MeaningfulSideEffectAuthorizationPort",
        "ToolRuntime",
        "intergrax.runtime.execution",
    )
    for token in forbidden:
        assert token not in source


def test_p2_23_no_environment_lookup_in_strategy_module() -> None:
    source = _STRATEGY_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "os" and node.attr == "environ":
                pytest.fail("strategy module must not read os.environ")
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "environ":
                    pytest.fail("strategy module must not import os.environ")


def test_p2_35_governed_facade_path_with_sqlite_reference_strategy() -> None:
    request = _request_for(_sqlite_payload())
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(_target_for(request)),
    )
    facade = ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=_AllowAuthorizationPort(),
        realization_service=service,
    )
    result = facade.realize(request)
    binding = result.configured_binding
    assert binding.provider_id == SQLITE_RELATIONAL_STORE_PROVIDER_ID
    assert binding.configuration_fingerprint == request.configuration_fingerprint
    assert binding.realization_evidence_refs == ()
    assert result.authorization_evidence.tenant_id == request.tenant_id


def test_p2_realize_rejects_non_sqlite_configuration_type_on_direct_call() -> None:
    alt = _AlternateTestPayload("alternate.test.configuration", "1", "fp-x")
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-direct",
        tenant_id="tenant-a",
        principal=_principal(),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-1",
        configuration=alt,
        configuration_fingerprint="fp-x",
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    strategy = SQLiteRelationalStoreConfigurationRealizationStrategy()
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        strategy.realize(request, _target_for(request))
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION
    )


def test_p2_ambient_realization_unaffected_by_env_through_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_SQLITE_DATA_DIR, "/env/must/not/influence")
    payload = _sqlite_payload(
        data_dir=Path("/payload/only/data"),
        relational_db=Path("/payload/only/db.sqlite"),
    )
    request = _request_for(payload)
    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(_target_for(request)),
    )
    facade = ExistingCapabilityConfigurationRealizationFacade(
        authorization_port=_AllowAuthorizationPort(),
        realization_service=service,
    )
    result = facade.realize(request)
    assert (
        result.configured_binding.configuration_fingerprint
        == payload.configuration_fingerprint
    )
    assert os.environ.get(ENV_SQLITE_DATA_DIR) == "/env/must/not/influence"
