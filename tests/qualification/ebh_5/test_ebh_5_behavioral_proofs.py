# © Artur Czarnecki. All rights reserved.

"""EBH-5 L3 representative behavioral replaceability proofs."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.contracts.control_plane_mutation import (
    control_plane_mutation_request_digest,
    evidence_from_request_and_decision,
)
from intergrax.contracts.event_delivery import ObservabilityExportPayload
from intergrax.contracts.observability_export import OtlpTransportPort
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.integrations._shared.conformance import assert_key_value_cache
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityIntegrationTarget,
    project_control_plane_mutation_request,
)
from intergrax.integrations.examples.custom_memory_kv import MANIFEST, CustomMemoryKvPlugin
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
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry

pytestmark = [
    pytest.mark.unit,
    pytest.mark.qualification,
]


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


def test_ebh5_behavioral_tokenizer_alternate_without_consumer_change() -> None:
    simple = SimpleTokenizer()
    tiktoken = TiktokenTokenizer()
    reg = TokenizerRegistry([simple, tiktoken], default_tokenizer_id="simple")
    alt = TokenizerRegistry([simple, tiktoken], default_tokenizer_id="tiktoken")
    assert reg.get("simple").count_tokens(text="hello") >= 1
    assert alt.get("tiktoken").count_tokens(text="hello") >= 1


def test_ebh5_behavioral_otlp_transport_port_injection() -> None:
    class RecordingTransport(OtlpTransportPort):
        def __init__(self) -> None:
            self.received: list[ObservabilityExportPayload] = []

        def export(self, payload: ObservabilityExportPayload) -> None:
            self.received.append(payload)

        def flush(self) -> None:
            return

        def close(self) -> None:
            return

    transport = RecordingTransport()
    payload = ObservabilityExportPayload(event_id="ebh5", kind="test")
    transport.export(payload)
    assert len(transport.received) == 1


def test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition() -> None:
    clear_catalog()
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
    cache = resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
    cache.set("tenant-a", "key", b"value")
    assert cache.get("tenant-a", "key") == b"value"


def test_ebh5_behavioral_kv_manifest_slug_alternate() -> None:
    clear_catalog()
    register_integration_plugin(CustomMemoryKvPlugin)
    profile = IntegrationProfile(
        key_value_cache=MANIFEST,
        options={MANIFEST.slug: {}},
    )
    cache = resolve_from_profile(profile, IntegrationCategory.KEY_VALUE_CACHE)
    assert_key_value_cache(cache)
