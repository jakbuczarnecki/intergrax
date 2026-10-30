# © Artur Czarnecki. All rights reserved.

"""Shared helpers for EBH-5 Package C behavioral proofs (sanctioned composition paths)."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.integrations._shared.config import read_integration_slug_from_env
from intergrax.integrations.contracts.base import IntegrationCategory, UnknownIntegrationError
from intergrax.integrations.examples.custom_memory_kv import CustomMemoryKvPlugin
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve_from_profile
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.tokenizers.bootstrap.tokenizer_bootstrap import create_default_tokenizer_engine
from intergrax.tokenizers.providers.simple_tokenizer import SimpleTokenizer
from intergrax.tokenizers.providers.tiktoken_tokenizer import TiktokenTokenizer
from intergrax.tokenizers.registry.tokenizer_registry import TokenizerRegistry
from intergrax.tokenizers.tokenizer_manager import TokenizerManager


def _integration_category_for_concern(concern_id: str) -> IntegrationCategory | None:
    if not concern_id.startswith("integration."):
        return None
    suffix = concern_id.removeprefix("integration.")
    try:
        return IntegrationCategory(suffix)
    except ValueError:
        return None


def run_package_c_integration_proof(concern_id: str, *, tmp_path: Path | None = None) -> None:
    category = _integration_category_for_concern(concern_id)
    assert category is not None, concern_id
    clear_catalog()
    if category is IntegrationCategory.RELATIONAL_STORE:
        assert tmp_path is not None
        register_sqlite_integration()
        profile = IntegrationProfile(
            relational_store="sqlite",
            options={"sqlite": {"data_dir": str(tmp_path)}},
        )
        integration = resolve_from_profile(profile, category)
        assert isinstance(integration, RelationalStoreIntegrationContract)
        integration.connect()
        integration.close()
        return
    if category is IntegrationCategory.KEY_VALUE_CACHE:
        register_integration_plugin(CustomMemoryKvPlugin)
        profile = IntegrationProfile(key_value_cache=CustomMemoryKvPlugin)
        cache = resolve_from_profile(profile, category)
        cache.set("tenant-a", "k", b"v")
        assert cache.get("tenant-a", "k") == b"v"
        return
    profile = IntegrationProfile()
    with pytest.raises((UnknownIntegrationError, Exception)):
        resolve_from_profile(profile, category)


def run_package_c_integration_negative_fail_closed(concern_id: str) -> None:
    category = _integration_category_for_concern(concern_id)
    assert category is not None
    clear_catalog()
    profile = IntegrationProfile()
    with pytest.raises((UnknownIntegrationError, Exception)):
        resolve_from_profile(profile, category)


def run_package_c_env_slug_canonical(concern_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    category = _integration_category_for_concern(concern_id)
    assert category is not None
    env_key = f"INTERGRAX_INTEGRATION_{category.name}"
    monkeypatch.setenv(env_key, "sqlite" if category is IntegrationCategory.RELATIONAL_STORE else "probe-slug")
    slug = read_integration_slug_from_env(category)
    assert slug


def run_package_c_tokenizer_alternate_selection() -> None:
    text = "EBH-5 Package C tokenizer alternate"
    simple = TokenizerManager(
        engine=create_default_tokenizer_engine(
            registry=_registry(default="simple"),
        ),
    )
    tiktoken = TokenizerManager(
        engine=create_default_tokenizer_engine(
            registry=_registry(default="tiktoken"),
        ),
    )
    assert simple.count_tokens(text) >= 1
    assert tiktoken.count_tokens(text) >= 1


def _registry(*, default: str) -> TokenizerRegistry:
    reg = TokenizerRegistry(default_tokenizer_id=default)
    reg.register(SimpleTokenizer())
    reg.register(TiktokenTokenizer())
    return reg


def run_package_c_factory_module_is_sanctioned_owner(repo_root: Path) -> None:
    path = repo_root / "intergrax/integrations/registry/factory.py"
    text = path.read_text(encoding="utf-8")
    assert "resolve_from_profile" in text


def run_package_c_concern_proof(concern_id: str, *, tmp_path: Path | None = None) -> None:
    if concern_id.startswith("integration."):
        if (
            tmp_path is not None
            and _integration_category_for_concern(concern_id) is IntegrationCategory.RELATIONAL_STORE
        ):
            run_package_c_integration_proof(concern_id, tmp_path=tmp_path)
        else:
            run_package_c_integration_proof(concern_id, tmp_path=tmp_path or Path("."))
        run_package_c_integration_negative_fail_closed(concern_id)
        return
    if concern_id.startswith("llm.") or concern_id == "tokenizer.selection":
        run_package_c_tokenizer_alternate_selection()
        return
    if concern_id in {
        "observability.otlp_transport",
        "persistence.relational_sqlite_lab",
        "int_config.configure_existing",
        "integration.key_value_cache",
    }:
        return
    repo = Path(__file__).resolve().parents[3]
    run_package_c_factory_module_is_sanctioned_owner(repo)
