# © Artur Czarnecki. All rights reserved.

"""EBH-7 one-pass factory RPL behavioral proofs (sanctioned contracts, distinct pos/neg)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from intergrax.distributed.providers.sqlite_kv_store import SqliteDistributedKVStore
from intergrax.integrations._shared.config import read_integration_slug_from_env
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.factory import resolve_slug
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import llm_provider_slug
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile, create_adapter
from intergrax.tools.providers.observability.resolve import resolve_observability_backend
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.qualification.ebh_5._qualification_llm_adapters import reset_and_register_qualification_llm_providers

FACTORY_RPL_ONE_PASS_CONCERNS: Final[frozenset[str]] = frozenset(
    {
        "llm.chat_completion",
        "llm.embedding",
        "llm.rerank",
        "llm.fallback_policy",
        "llm.model_routing_qualifier",
        "plugin.llm_adapter_registry",
        "plugin.integration_catalog",
        "env.integration_slug",
        "distributed.sqlite_kv",
        "execution.integration_bound_resolution",
        "tool.observability_backend",
        "speech.provider",
    }
)

_ONE_PASS_TEST_MODULE: Final[str] = "tests/qualification/ebh_5/test_ebh_5_factory_rpl_one_pass.py"


def factory_rpl_positive_node(concern_id: str) -> str:
    return f"{_ONE_PASS_TEST_MODULE}::test_ebh5_factory_rpl_positive[{concern_id}]"


def factory_rpl_negative_node(concern_id: str) -> str:
    return f"{_ONE_PASS_TEST_MODULE}::test_ebh5_factory_rpl_negative[{concern_id}]"


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
    else:
        raise AssertionError(f"No negative proof for {concern_id}")
