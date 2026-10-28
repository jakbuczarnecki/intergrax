# © Artur Czarnecki. All rights reserved.

"""EBH-5 L3 representative behavioral replaceability proofs."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.runtime_event_delivery_wiring import (
    resolve_application_runtime_event_delivery_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import GovernanceBundle
from intergrax.contracts.control_plane_mutation import (
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.event_delivery import ObservabilityExportPayload
from intergrax.contracts.observability_export import (
    ConfigurationError,
    ExporterKind,
    OtlpTransportError,
    OtlpTransportPort,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations._shared.conformance import assert_key_value_cache
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityIntegrationTarget,
    project_control_plane_mutation_request,
)
from intergrax.integrations.examples.custom_memory_kv import MANIFEST, CustomMemoryKvPlugin
from intergrax.integrations.examples.custom_memory_kv.adapter import InProcessKeyValueCache
from intergrax.integrations.examples.custom_memory_kv.integration import CustomMemoryKvIntegration
from intergrax.integrations.existing_capability_configuration_service import (
    ExistingCapabilityConfigurationRealizationService,
)
from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
    SQLiteRelationalStoreConfigurationRealizationStrategy,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
)
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.observability.event_delivery.export_factory import ObservabilityExportSinkFactory
from intergrax.runtime.observability.event_delivery.otlp_event_export_sink import OtlpEventExportSink
from intergrax.runtime.observability.exporters.distributed.collector_transport import CollectorTransport
from intergrax.runtime.observability.exporters.otlp.otlp_transport import OtlpTransport
from intergrax.tokenizers.bootstrap.tokenizer_bootstrap import create_default_tokenizer_engine
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tokenizers.tokenizer_manager import TokenizerManager

pytestmark = [
    pytest.mark.unit,
    pytest.mark.qualification,
]


class _Ebh5RecordingOtlpTransport(OtlpTransportPort):
    def __init__(self) -> None:
        self.received: list[ObservabilityExportPayload] = []

    def export(self, payload: ObservabilityExportPayload) -> None:
        self.received.append(payload)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


_EBH5_OTLP_ENDPOINT = "http://127.0.0.1:4318/v1/logs"
_EBH5_OTLP_SERVICE = "intergrax-ebh5-otlp"


def _ebh5_bounded_otlp_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh5.otlp.transport")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.OTLP,
            "otlp_export_endpoint": _EBH5_OTLP_ENDPOINT,
        },
    )
    return env


def _ebh5_distributed_otlp_env() -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh5.otlp.distributed")
    observability = GovernanceBundle.enterprise_cluster_observability(
        otlp_export_endpoint=_EBH5_OTLP_ENDPOINT,
        observability_export_service_name=_EBH5_OTLP_SERVICE,
    )
    return env.model_copy(
        update={
            "governance": env.governance.model_copy(update={"observability": observability}),
        },
    )


def _ebh5_tokenizer_registry(*, default_tokenizer_id: str) -> TokenizerRegistry:
    registry = TokenizerRegistry(default_tokenizer_id=default_tokenizer_id)
    registry.register(SimpleTokenizer())
    registry.register(TiktokenTokenizer())
    return registry


def _ebh5_tokenizer_manager(*, default_tokenizer_id: str) -> TokenizerManager:
    return TokenizerManager(
        engine=create_default_tokenizer_engine(
            registry=_ebh5_tokenizer_registry(default_tokenizer_id=default_tokenizer_id),
        ),
    )


def _ebh5_int_config_target(
    request: ExistingCapabilityConfigurationRealizationRequest,
) -> ExistingCapabilityIntegrationTarget:
    return ExistingCapabilityIntegrationTarget(
        tenant_id=request.tenant_id,
        integration_category=request.integration_category,
        provider_id=request.provider_id,
        resource_scope=request.resource_scope,
        current_revision=request.current_revision,
    )


def test_ebh5_behavioral_int_config_sqlite_strategy_via_generic_service() -> None:
    from intergrax.contracts.agent_run import RequestIdentity
    from intergrax.contracts.agent_run_enums import PrincipalType
    from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
    from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
        SQLiteRelationalStoreConfigurationPayload,
    )

    payload = SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="ebh5-int-config",
        tenant_id="tenant-a",
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
            auth_subject="user-1",
        ),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-relational-ebh5",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )
    target = _ebh5_int_config_target(request)

    class _StaticResolver:
        def resolve_existing_integration(
            self,
            req: ExistingCapabilityConfigurationRealizationRequest,
        ) -> ExistingCapabilityIntegrationTarget:
            return target

    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(SQLiteRelationalStoreConfigurationRealizationStrategy(),),
        existing_integration_resolver=_StaticResolver(),
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


def test_ebh5_behavioral_int_config_negative_unsupported_strategy_fail_closed() -> None:
    from intergrax.contracts.agent_run import RequestIdentity
    from intergrax.contracts.agent_run_enums import PrincipalType
    from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
    from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
        SQLiteRelationalStoreConfigurationPayload,
    )

    payload = SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="ebh5-int-config-neg",
        tenant_id="tenant-a",
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
            auth_subject="user-1",
        ),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-relational-ebh5-neg",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )

    class _StaticResolver:
        def resolve_existing_integration(
            self,
            req: ExistingCapabilityConfigurationRealizationRequest,
        ) -> ExistingCapabilityIntegrationTarget:
            return _ebh5_int_config_target(req)

    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(),
        existing_integration_resolver=_StaticResolver(),
    )
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc_info:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert exc_info.value.reason == ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_STRATEGY


def test_ebh5_behavioral_relational_store_sqlite_via_sanctioned_profile(tmp_path: Path) -> None:
    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(tmp_path)}},
    )
    integration = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(integration, RelationalStoreIntegrationContract)
    integration.connect()
    integration.close()


def test_ebh5_behavioral_relational_store_negative_unregistered_provider_fail_closed(tmp_path: Path) -> None:
    clear_catalog()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(tmp_path)}},
    )
    with pytest.raises(UnknownIntegrationError):
        resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)


def test_ebh5_behavioral_tokenizer_selection_via_bootstrap_composition() -> None:
    text = "Intergrax EBH-5 tokenizer selection proof"
    simple_default = _ebh5_tokenizer_manager(default_tokenizer_id="simple")
    tiktoken_default = _ebh5_tokenizer_manager(default_tokenizer_id="tiktoken")

    def _count_tokens(manager: TokenizerManager) -> int:
        return manager.count_tokens(text)

    assert _count_tokens(simple_default) >= 1
    assert _count_tokens(tiktoken_default) >= 1
    engine = create_default_tokenizer_engine(registry=_ebh5_tokenizer_registry(default_tokenizer_id="simple"))
    assert engine.count_tokens(tokenizer_id="tiktoken", text=text) >= 1


def test_ebh5_behavioral_tokenizer_selection_negative_unknown_default() -> None:
    registry = TokenizerRegistry(default_tokenizer_id="does-not-exist")
    registry.register(SimpleTokenizer())
    with pytest.raises(ValueError, match="Default tokenizer not found"):
        registry.get(None)


@pytest.mark.asyncio
async def test_ebh5_behavioral_otlp_transport_via_sanctioned_composition_wiring() -> None:
    pytest.importorskip("opentelemetry.sdk._logs")

    env_otlp = _ebh5_bounded_otlp_env()
    wiring_otlp = resolve_application_runtime_event_delivery_wiring(env_otlp)
    assert wiring_otlp.bounded_sink is not None
    assert isinstance(wiring_otlp.otlp_transport, OtlpTransport)
    assert isinstance(wiring_otlp.event_export_sink, OtlpEventExportSink)

    env_distributed = _ebh5_distributed_otlp_env()
    wiring_distributed = resolve_application_runtime_event_delivery_wiring(env_distributed)
    assert isinstance(wiring_distributed.otlp_transport, CollectorTransport)
    assert isinstance(wiring_distributed.event_export_sink, OtlpEventExportSink)
    assert type(wiring_otlp.otlp_transport) is not type(wiring_distributed.otlp_transport)

    recording = _Ebh5RecordingOtlpTransport()
    factory = ObservabilityExportSinkFactory(otlp_transport=recording)
    wiring_injected = resolve_application_runtime_event_delivery_wiring(
        env_otlp,
        export_sink_factory=factory,
    )
    sink = wiring_injected.event_export_sink
    assert isinstance(sink, OtlpEventExportSink)
    payload = ObservabilityExportPayload(event_id="ebh5-otlp-wiring", kind="runtime.event")
    await sink.export(payload)
    assert len(recording.received) == 1
    assert recording.received[0].event_id == "ebh5-otlp-wiring"
    assert isinstance(wiring_injected.otlp_transport, OtlpTransport)


def test_ebh5_behavioral_otlp_transport_negative_invalid_distributed_configuration() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh5.otlp.invalid-distributed")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.DISTRIBUTED_OTLP,
            "otlp_export_endpoint": "",
            "observability_export_service_name": _EBH5_OTLP_SERVICE,
        },
    )
    with pytest.raises(ConfigurationError):
        resolve_application_runtime_event_delivery_wiring(env)


@pytest.mark.asyncio
async def test_ebh5_behavioral_otlp_transport_negative_otlp_without_endpoint_fail_closed() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh5.otlp.no-endpoint")
    env.observability_profile = env.observability_profile.model_copy(
        update={
            "bounded_event_delivery_enabled": True,
            "observability_exporter_kind": ExporterKind.OTLP,
            "otlp_export_endpoint": "",
        },
    )
    wiring = resolve_application_runtime_event_delivery_wiring(env)
    assert wiring.otlp_transport is None
    assert isinstance(wiring.event_export_sink, OtlpEventExportSink)
    payload = ObservabilityExportPayload(event_id="ebh5-otlp-disabled", kind="test")
    spy = _Ebh5RecordingOtlpTransport()
    assert wiring.otlp_transport is not spy
    await wiring.event_export_sink.export(payload)
    assert len(spy.received) == 0
    if wiring.delivery_metrics is not None:
        assert wiring.delivery_metrics.snapshot().export_success_total == 0


def test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition() -> None:
    clear_catalog()
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache = resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
    cache.set("tenant-a", "key", b"value")
    assert cache.get("tenant-a", "key") == b"value"


def test_ebh5_behavioral_kv_negative_unregistered_plugin_fail_closed() -> None:
    clear_catalog()
    profile = IntegrationProfile(
        key_value_cache=MANIFEST,
        options={MANIFEST.slug: {}},
    )
    with pytest.raises(UnknownIntegrationError):
        resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)


def test_ebh5_behavioral_kv_manifest_slug_alternate() -> None:
    clear_catalog()
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(
        key_value_cache=MANIFEST,
        options={MANIFEST.slug: {}},
    )
    cache = resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)


class _Ebh5FailingOtlpTransport(OtlpTransportPort):
    def export(self, payload: ObservabilityExportPayload) -> None:
        _ = payload
        raise OtlpTransportError("injected active transport export fault")

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_ebh5_rpl_f_otlp_active_transport_export_failure_propagates() -> None:
    env = _ebh5_bounded_otlp_env()
    factory = ObservabilityExportSinkFactory(otlp_transport=_Ebh5FailingOtlpTransport())
    wiring = resolve_application_runtime_event_delivery_wiring(
        env,
        export_sink_factory=factory,
    )
    assert isinstance(wiring.event_export_sink, OtlpEventExportSink)
    payload = ObservabilityExportPayload(event_id="ebh5-rpl-f-otlp-fault", kind="runtime.event")
    with pytest.raises(OtlpTransportError, match="injected active transport"):
        await wiring.event_export_sink.export(payload)


class _Ebh5FailingActiveTokenizer(SimpleTokenizer):
    id = "failing-active"

    def count_tokens(self, *, text: str) -> int:
        _ = text
        raise RuntimeError("injected active tokenizer fault")


def test_ebh5_rpl_f_tokenizer_active_provider_tokenization_failure() -> None:
    registry = TokenizerRegistry(default_tokenizer_id="failing-active")
    registry.register(_Ebh5FailingActiveTokenizer())
    manager = TokenizerManager(
        engine=create_default_tokenizer_engine(registry=registry),
    )
    with pytest.raises(RuntimeError, match="injected active tokenizer"):
        manager.count_tokens("Intergrax RPL-F proof")


def test_ebh5_rpl_f_relational_active_store_execute_failure(tmp_path: Path) -> None:
    clear_catalog()
    register_sqlite_integration()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE.slug: {"data_dir": str(tmp_path)}},
    )
    integration = resolve_from_profile(profile, IntegrationCategory.RELATIONAL_STORE)
    assert isinstance(integration, RelationalStoreIntegrationContract)
    integration.connect()
    with pytest.raises(Exception):
        integration.execute("NOT VALID SQL FOR RPL-F INJECTION")


class _Ebh5InjectedRealizationFailureStrategy(SQLiteRelationalStoreConfigurationRealizationStrategy):
    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool:
        _ = request, existing_target
        return True

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ):
        _ = request, existing_target
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.REALIZATION_FAILED,
            detail="injected active strategy fault",
        )


def test_ebh5_rpl_f_int_config_active_strategy_realization_failure() -> None:
    from intergrax.contracts.agent_run import RequestIdentity
    from intergrax.contracts.agent_run_enums import PrincipalType
    from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRisk
    from intergrax.integrations.providers.relational_store.sqlite.configuration_realization import (
        SQLiteRelationalStoreConfigurationPayload,
    )

    payload = SQLiteRelationalStoreConfigurationPayload(
        data_dir=Path("/data/tenant/store"),
        relational_db=Path("/data/tenant/store/custom.db"),
    )
    request = ExistingCapabilityConfigurationRealizationRequest(
        request_id="ebh5-rpl-f-int-config",
        tenant_id="tenant-a",
        principal=RequestIdentity(
            tenant_id="tenant-a",
            user_id="user-1",
            principal_type=PrincipalType.USER,
            auth_subject="user-1",
        ),
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
        resource_scope="scope-rpl-f",
        configuration=payload,
        configuration_fingerprint=payload.configuration_fingerprint,
        current_revision="rev-1",
        risk_classification=ControlPlaneMutationRisk.LOW,
    )

    class _StaticResolver:
        def resolve_existing_integration(
            self,
            req: ExistingCapabilityConfigurationRealizationRequest,
        ) -> ExistingCapabilityIntegrationTarget:
            return _ebh5_int_config_target(req)

    service = ExistingCapabilityConfigurationRealizationService(
        strategies=(_Ebh5InjectedRealizationFailureStrategy(),),
        existing_integration_resolver=_StaticResolver(),
    )
    governance = project_control_plane_mutation_request(request)
    digest = control_plane_mutation_request_digest(governance)
    decision = PolicyDecision(action=PolicyAction.ALLOW, reason="ok")
    evidence = evidence_from_request_and_decision(
        governance,
        decision=decision,
        request_digest=digest,
    )
    with pytest.raises(ExistingCapabilityConfigurationRealizationError) as exc_info:
        service.realize_admitted(request, authorization_evidence=evidence)
    assert exc_info.value.reason == ExistingCapabilityConfigurationRealizationFailureReason.REALIZATION_FAILED


class _Ebh5FaultInjectedKvClient(InProcessKeyValueCache):
    def set(
        self,
        tenant_id: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        _ = tenant_id, key, value, ttl_seconds
        raise RuntimeError("injected active kv provider fault")


def test_ebh5_rpl_f_kv_active_plugin_operation_failure() -> None:
    integration = CustomMemoryKvIntegration.from_client(_Ebh5FaultInjectedKvClient())
    profile = IntegrationProfile(key_value_cache=integration)
    cache = resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
    with pytest.raises(RuntimeError, match="injected active kv"):
        cache.set("tenant-a", "key", b"value")
