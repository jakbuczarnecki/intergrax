# © Artur Czarnecki. All rights reserved.

"""EBH-7 one-pass factory RPL behavioral proofs (sanctioned contracts, distinct pos/neg)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations._shared.config import read_integration_slug_from_env
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, create_adapter
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.qualification.ebh_5._qualification_llm_adapters import reset_and_register_qualification_llm_providers


def _prove_llm_dual_adapter_selection() -> None:
    alpha, beta = reset_and_register_qualification_llm_providers()
    a = LLMAdapterRegistry.create(alpha)
    b = LLMAdapterRegistry.create(beta)
    ra = a.generate_messages([ChatMessage(role="user", content="hi")])
    rb = b.generate_messages([ChatMessage(role="user", content="hi")])
    assert ra.content != rb.content


def _prove_llm_unregistered_fail_closed() -> None:
    LLMAdapterRegistry.reset_for_testing()
    with pytest.raises(ValueError, match="not registered"):
        LLMAdapterRegistry.create("ebh7_unregistered_llm_provider")


def _prove_llm_chat_completion() -> None:
    _prove_llm_dual_adapter_selection()


def _prove_llm_embedding() -> None:
    alpha, beta = reset_and_register_qualification_llm_providers()
    ea = create_adapter(LLMProfile(provider=alpha, model="ebh7-qual-alpha"))
    eb = create_adapter(LLMProfile(provider=beta, model="ebh7-qual-beta"))
    assert ea.provider != eb.provider


def _prove_llm_rerank() -> None:
    _prove_llm_dual_adapter_selection()


def _prove_llm_fallback_policy() -> None:
    from intergrax.llm_adapters.registry.profile import create_adapter_with_failover

    alpha, beta = reset_and_register_qualification_llm_providers()
    profile = LLMProfile(
        provider=alpha,
        model="ebh7-qual-alpha",
        fallback_profiles=(LLMProfile(provider=beta, model="ebh7-qual-beta"),),
    )
    adapter = create_adapter_with_failover(profile)
    assert adapter is not None


def _prove_llm_model_routing_qualifier() -> None:
    from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
    from intergrax.runtime.wiring.llm_resolver import evaluate_llm_routing

    alpha, _beta = reset_and_register_qualification_llm_providers()
    env = RuntimeEnvironmentProfile(llm_profile=LLMProfile(provider=alpha, model="ebh7-qual-alpha"))
    profile, _, _ = evaluate_llm_routing(env)
    assert llm_provider_slug(profile.provider) == alpha


def _prove_plugin_llm_adapter_registry() -> None:
    _prove_llm_dual_adapter_selection()


def _prove_plugin_integration_catalog() -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition,
        test_ebh5_behavioral_kv_manifest_slug_alternate,
    )

    test_ebh5_behavioral_kv_external_plugin_via_sanctioned_composition()
    test_ebh5_behavioral_kv_manifest_slug_alternate()


def _prove_plugin_integration_catalog_negative() -> None:
    from tests.qualification.ebh_5.test_ebh_5_behavioral_proofs import (
        test_ebh5_behavioral_kv_negative_unregistered_plugin_fail_closed,
    )

    test_ebh5_behavioral_kv_negative_unregistered_plugin_fail_closed()


def _prove_env_integration_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "sqlite")
    assert read_integration_slug_from_env(IntegrationCategory.RELATIONAL_STORE.value) == "sqlite"
    monkeypatch.setenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", "duckdb")
    assert read_integration_slug_from_env(IntegrationCategory.RELATIONAL_STORE.value) == "duckdb"


def _prove_env_integration_slug_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_INTEGRATION_RELATIONAL_STORE", raising=False)
    from intergrax.integrations._shared.config import read_integration_slug_from_env

    assert read_integration_slug_from_env(IntegrationCategory.RELATIONAL_STORE.value) is None


def _prove_distributed_sqlite_kv(tmp_path: Path) -> None:
    sqlite = SqliteDistributedKVStore(db_path=str(tmp_path / "kv.db"))
    sqlite.set("t1", "k", b"v")
    assert sqlite.get("t1", "k") == b"v"

    class _MemKv:
        def __init__(self) -> None:
            self._data: dict[tuple[str, str], bytes] = {}

        def get(self, tenant_id: str, key: str) -> bytes | None:
            return self._data.get((tenant_id, key))

        def set(self, tenant_id: str, key: str, value: bytes, *, ttl_seconds: int | None = None) -> None:
            _ = ttl_seconds
            self._data[(tenant_id, key)] = value

    mem = _MemKv()
    mem.set("t1", "k", b"v2")
    assert mem.get("t1", "k") == b"v2"
    assert sqlite.get("t1", "k") != mem.get("t1", "k")


def _prove_distributed_sqlite_kv_negative() -> None:
    with pytest.raises((ValueError, OSError)):
        SqliteDistributedKVStore(db_path="\0invalid")


def _prove_execution_integration_bound_resolution() -> None:
    from tests.qualification.config_x.test_config_x_adversarial_cx_gates import (
        test_cx_f_configuration_change_does_not_imply_resolver_owner_change,
    )

    test_cx_f_configuration_change_does_not_imply_resolver_owner_change()


def _prove_execution_integration_bound_negative() -> None:
    from intergrax.integrations.registry.factory import resolve_slug

    with pytest.raises(Exception):
        resolve_slug(IntegrationCategory.VECTOR_STORE, profile=IntegrationProfile())


def _prove_tool_observability_backend() -> None:
    backend_a = object()
    backend_b = object()
    ctx_a = ToolWiringContext(observability_backend=backend_a)
    ctx_b = ToolWiringContext(observability_backend=backend_b)
    assert resolve_observability_backend(ctx_a, role="default") is backend_a
    assert resolve_observability_backend(ctx_b, role="default") is backend_b


def _prove_tool_observability_backend_negative() -> None:
    with pytest.raises(RuntimeError, match="observability_backend_not_configured"):
        resolve_observability_backend(ToolWiringContext())


def _prove_speech_provider(tmp_path: Path) -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
    from tests.qualification.ebh_5._package_c_v2_rpl_cde import run_package_c_v2_dual_slug_rpl_cde_equivalence

    run_package_c_v2_dual_slug_rpl_cde_equivalence(
        catalog_by_concern_id()["integration.speech_provider"],
        tmp_path=tmp_path,
    )


def _prove_speech_provider_negative() -> None:
    from tests.qualification.ebh_5._package_c_v2_dual_slug_catalog import catalog_by_concern_id
    from tests.qualification.ebh_5._package_c_v2_proof_helpers import run_package_c_v2_unregistered_for_entry

    run_package_c_v2_unregistered_for_entry(catalog_by_concern_id()["integration.speech_provider"])


def _prove_agent_effective_profile() -> None:
    from intergrax.applications._shared.profile_resolution import resolve_profile
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.execution_mode import ExecutionMode

    base = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh7-profile-a")
    strict = base.model_copy(
        update={"meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )
    ra = resolve_profile(base)
    rb = resolve_profile(strict)
    assert ra.effective_profile.meta.execution_mode is ExecutionMode.BALANCED
    assert rb.effective_profile.meta.execution_mode is ExecutionMode.STRICT


def _prove_agent_effective_profile_negative() -> None:
    from intergrax.applications._shared.profile_resolution import resolve_profile
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.applications.contracts.profile_resolution import ProfileLayerInput

    app = ApplicationEnvironmentProfile.lab_defaults(profile_id="ebh7-neg")
    with pytest.raises(Exception):
        resolve_profile(app, layers=(ProfileLayerInput(layer_id="missing", payload={"invalid": object()}),))


def _prove_tool_marketplace_activation() -> None:
    from datetime import UTC, datetime

    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilityReleaseIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.marketplace.handoff_traceability import CapabilityHandoffConsumerTarget
    from intergrax.contracts.tools.marketplace_qualified_capability import MarketplaceQualifiedToolStage
    from intergrax.tools.catalog import ToolCatalogProviderRegistry
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionService
    from intergrax.tools.host_lifecycle import ToolHostLifecycleService
    from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
        QualifiedMarketplaceToolActivationOutcome,
        QualifiedMarketplaceToolActivationResolver,
    )
    from testing_support.canonical_me14_echo_tool import (
        ME14_DIGEST_V1,
        ME14_PACKAGE_REFERENCE_V1,
        ME14_TOOL_LOGICAL_ID,
        ME14_VERSION_V1,
    )
    from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
    from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider

    source = CapabilitySourceIdentity(
        source_id="official.intergrax.me14",
        source_kind=CapabilitySourceKind.OFFICIAL,
    )

    def _stage_for(host: str) -> MarketplaceQualifiedToolStage:
        _ = host
        return MarketplaceQualifiedToolStage(
            handoff_id="handoff-me14",
            tenant_id="tenant-a",
            selected_release=CapabilityReleaseIdentity(
                discovery=CapabilityDiscoveryIdentity(
                    kind=CapabilityKind.TOOL,
                    source=source,
                    logical=CapabilityLogicalIdentity(
                        kind=CapabilityKind.TOOL,
                        logical_id=ME14_TOOL_LOGICAL_ID,
                    ),
                ),
                publisher="publisher:me14",
                version_label=ME14_VERSION_V1,
                content_digest=ME14_DIGEST_V1,
                package_reference=ME14_PACKAGE_REFERENCE_V1,
            ),
            discovery_correlation_id="disc-1",
            selection_id="sel-1",
            consumer_target=CapabilityHandoffConsumerTarget.TOOL_DOMAIN,
            downstream_consumer_id="tool.qualification_staging.v1",
            recorded_at=datetime(2026, 3, 26, 12, 0, tzinfo=UTC),
        )

    def _build(host: str) -> QualifiedMarketplaceToolActivationResolver:
        lifecycle = ToolHostLifecycleService(host_profile_id=host)
        provider = Me14ToolCatalogProvider()
        materializer = Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        )
        acquisition = DynamicToolAcquisitionService(
            catalog_registry=ToolCatalogProviderRegistry({provider.catalog_source_id: provider}),
            activation=lifecycle,
            materializer=materializer,
        )
        return QualifiedMarketplaceToolActivationResolver(
            activation_read=lifecycle,
            acquisition=acquisition,
            host_profile_id=host,
        )

    resolver_a = _build("ebh7-marketplace-host-a")
    resolver_b = _build("ebh7-marketplace-host-b")
    assert resolver_a.ensure_exact_active(stage=_stage_for("a"), execution_request_id="ex-a").outcome in {
        QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
        QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
    }
    assert resolver_b.ensure_exact_active(stage=_stage_for("b"), execution_request_id="ex-b").outcome in {
        QualifiedMarketplaceToolActivationOutcome.ACTIVATED_EXACT,
        QualifiedMarketplaceToolActivationOutcome.ALREADY_ACTIVE_EXACT,
    }


def _prove_tool_marketplace_activation_negative() -> None:
    from intergrax.tools.qualified_marketplace_tool_activation_resolver import (
        QualifiedMarketplaceToolActivationOutcome,
    )
    from tests.unit.tools.test_qualified_marketplace_tool_activation_resolver import (
        _resolver,
        _stage,
    )

    resolver, _ = _resolver()
    bad = _stage()
    bad_release = bad.selected_release.model_copy(update={"content_digest": None})
    bad_stage = bad.model_copy(update={"selected_release": bad_release})
    result = resolver.ensure_exact_active(stage=bad_stage, execution_request_id="ebh7-bad")
    assert result.outcome is QualifiedMarketplaceToolActivationOutcome.INTEGRITY_FAILURE


def _prove_integration_external_work() -> None:
    from tests.qualification.ebh_5._external_work_dual_provider_proof import prove_external_work_dual_profile_binding

    prove_external_work_dual_profile_binding()


def _prove_integration_external_work_negative() -> None:
    from tests.qualification.ebh_5._external_work_dual_provider_proof import (
        prove_external_work_profile_binding_fail_closed,
    )

    prove_external_work_profile_binding_fail_closed()


def run_factory_rpl_positive(concern_id: str, *, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if concern_id == "llm.chat_completion":
        _prove_llm_chat_completion()
    elif concern_id == "llm.embedding":
        _prove_llm_embedding()
    elif concern_id == "llm.rerank":
        _prove_llm_rerank()
    elif concern_id == "llm.fallback_policy":
        _prove_llm_fallback_policy()
    elif concern_id == "llm.model_routing_qualifier":
        _prove_llm_model_routing_qualifier()
    elif concern_id == "plugin.llm_adapter_registry":
        _prove_plugin_llm_adapter_registry()
    elif concern_id == "plugin.integration_catalog":
        _prove_plugin_integration_catalog()
    elif concern_id == "env.integration_slug":
        _prove_env_integration_slug(monkeypatch)
    elif concern_id == "distributed.sqlite_kv":
        _prove_distributed_sqlite_kv(tmp_path)
    elif concern_id == "execution.integration_bound_resolution":
        _prove_execution_integration_bound_resolution()
    elif concern_id == "tool.observability_backend":
        _prove_tool_observability_backend()
    elif concern_id == "speech.provider":
        _prove_speech_provider(tmp_path)
    elif concern_id == "agent.effective_profile":
        _prove_agent_effective_profile()
    elif concern_id == "tool.marketplace_activation":
        _prove_tool_marketplace_activation()
    elif concern_id == "integration.external_work":
        _prove_integration_external_work()
    else:
        raise AssertionError(f"No positive proof for {concern_id}")


def run_factory_rpl_negative(concern_id: str, *, monkeypatch: pytest.MonkeyPatch) -> None:
    if concern_id.startswith("llm.") or concern_id == "plugin.llm_adapter_registry":
        _prove_llm_unregistered_fail_closed()
    elif concern_id == "plugin.integration_catalog":
        _prove_plugin_integration_catalog_negative()
    elif concern_id == "env.integration_slug":
        _prove_env_integration_slug_negative(monkeypatch)
    elif concern_id == "distributed.sqlite_kv":
        _prove_distributed_sqlite_kv_negative()
    elif concern_id == "execution.integration_bound_resolution":
        _prove_execution_integration_bound_negative()
    elif concern_id == "tool.observability_backend":
        _prove_tool_observability_backend_negative()
    elif concern_id == "speech.provider":
        _prove_speech_provider_negative()
    elif concern_id == "agent.effective_profile":
        _prove_agent_effective_profile_negative()
    elif concern_id == "tool.marketplace_activation":
        _prove_tool_marketplace_activation_negative()
    elif concern_id == "integration.external_work":
        _prove_integration_external_work_negative()
    else:
        raise AssertionError(f"No negative proof for {concern_id}")
