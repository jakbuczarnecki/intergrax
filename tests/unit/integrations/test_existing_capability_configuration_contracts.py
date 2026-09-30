# © Artur Czarnecki. All rights reserved.

"""P1 contract surface for existing-capability configuration realization."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityConfigurationRealizationResult,
    ExistingCapabilityConfigurationRealizationStrategy,
    ExistingCapabilityIntegrationTarget,
    IntegrationConfigurationPayload,
    validate_realization_request_invariants,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk

pytestmark = pytest.mark.unit

_CONTRACTS_PATH = Path(__file__).resolve().parents[3] / (
    "intergrax/integrations/contracts/existing_capability_configuration.py"
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


def _payload(**overrides: str) -> IntegrationConfigurationPayload:
    return _TestConfigurationPayload(
        _configuration_type=overrides.get("configuration_type", "test.config.v1"),
        _configuration_version=overrides.get("configuration_version", "1"),
        _configuration_fingerprint=overrides.get(
            "configuration_fingerprint", "fp-test-001"
        ),
    )


def _request(
    *,
    tenant_id: str = "tenant-a",
    principal: RequestIdentity | None = None,
    configuration_fingerprint: str = "fp-test-001",
) -> ExistingCapabilityConfigurationRealizationRequest:
    return ExistingCapabilityConfigurationRealizationRequest(
        request_id="req-1",
        tenant_id=tenant_id,
        principal=principal or _principal(tenant_id=tenant_id),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id="sqlite",
        resource_scope="scope-a",
        configuration=_payload(configuration_fingerprint=configuration_fingerprint),
        configuration_fingerprint=configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )


def test_p1_01_typed_payload_structural_implementation_works() -> None:
    payload = _payload()
    assert isinstance(payload, IntegrationConfigurationPayload)
    assert payload.configuration_type == "test.config.v1"
    assert payload.configuration_version == "1"
    assert payload.configuration_fingerprint == "fp-test-001"
    request = _request()
    assert request.configuration.configuration_type == "test.config.v1"


def test_p1_02_request_rejects_missing_tenant() -> None:
    principal = RequestIdentity.model_construct(
        tenant_id="",
        user_id="user-1",
        principal_type=PrincipalType.USER,
        auth_subject="user-1",
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        ExistingCapabilityConfigurationRealizationRequest(
            request_id="req-1",
            tenant_id="",
            principal=principal,
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id="sqlite",
            resource_scope="scope-a",
            configuration=_payload(),
            configuration_fingerprint="fp-test-001",
            current_revision="rev-1",
            risk_classification=ControlPlaneMutationRisk.LOW,
        )
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_p1_03_request_rejects_principal_tenant_mismatch() -> None:
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc:
        _request(tenant_id="tenant-a", principal=_principal(tenant_id="tenant-b"))
    assert (
        exc.value.reason
        is ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH
    )


def test_p1_04_no_semantic_any_dict_object_boundary() -> None:
    source = _CONTRACTS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned_names = {"Any", "Mapping", "MutableMapping"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in banned_names:
            pytest.fail(f"forbidden name {node.id} in contracts module")
        if isinstance(node, ast.Attribute) and node.attr in banned_names:
            pytest.fail(f"forbidden attribute {node.attr} in contracts module")
    assert "dict[str, Any]" not in source
    assert "Mapping[str, Any]" not in source


@dataclass(frozen=True)
class _ExternalStrategy:
    strategy_id_value: str = "external-strategy"

    @property
    def strategy_id(self) -> str:
        return self.strategy_id_value

    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool:
        return True

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
        )


def test_p1_05_external_strategy_structurally_satisfies_spi() -> None:
    strategy = _ExternalStrategy()
    assert isinstance(strategy, ExistingCapabilityConfigurationRealizationStrategy)


def test_request_validate_invariants_direct() -> None:
    request = _request()
    validate_realization_request_invariants(request)
    target = ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id=request.provider_id,
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )
    assert target.provider_id == "sqlite"


_P1_PRODUCTION_PATHS = (
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/contracts/existing_capability_configuration.py",
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/existing_capability_configuration_service.py",
    Path(__file__).resolve().parents[3]
    / "intergrax/integrations/existing_capability_configuration_facade.py",
)

_FORBIDDEN_SEMANTIC_NAMES = frozenset(
    {"Any", "Mapping", "MutableMapping", "IntegrationBinding"}
)
_FORBIDDEN_CALL_NAMES = frozenset({"getattr", "setattr", "hasattr"})


def _annotation_names(node: ast.expr) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Attribute):
        names.add(node.attr)
    elif isinstance(node, ast.Subscript):
        names |= _annotation_names(node.value)
        if isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                names |= _annotation_names(elt)
        else:
            names |= _annotation_names(node.slice)
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        names |= _annotation_names(node.left)
        names |= _annotation_names(node.right)
    return names


def test_p1_r1_no_integration_binding_on_p1_production_surface() -> None:
    for path in _P1_PRODUCTION_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "intergrax.integrations.contracts.binding" not in source
        assert "IntegrationBinding" not in source


def test_p1_r1_strong_typing_regression_gate_on_p1_production() -> None:
    for path in _P1_PRODUCTION_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in _FORBIDDEN_SEMANTIC_NAMES:
                pytest.fail(f"{path.name}: forbidden name {node.id}")
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _FORBIDDEN_SEMANTIC_NAMES
            ):
                pytest.fail(f"{path.name}: forbidden attribute {node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALL_NAMES:
                    pytest.fail(f"{path.name}: forbidden call {node.func.id}")


def _type_hint_tokens(type_hint: object) -> set[str]:
    if isinstance(type_hint, str):
        tree = ast.parse(type_hint, mode="eval")
        return _annotation_names(tree.body)
    if isinstance(type_hint, type):
        return {type_hint.__name__}
    return _annotation_names(type_hint)  # type: ignore[arg-type]


def test_p1_r1_contract_dto_field_audit() -> None:
    dto_types = (
        ExistingCapabilityIntegrationTarget,
        ConfiguredCapabilityBinding,
        ExistingCapabilityConfigurationRealizationResult,
    )
    banned = frozenset({"IntegrationBinding", "Any", "object"})
    for dto in dto_types:
        for field in dto.__dataclass_fields__.values():
            found = _type_hint_tokens(field.type) & banned
            if found:
                pytest.fail(f"{dto.__name__}.{field.name} exposes {found}")
