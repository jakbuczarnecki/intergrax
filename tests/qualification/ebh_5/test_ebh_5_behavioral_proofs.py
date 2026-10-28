# © Artur Czarnecki. All rights reserved.

"""EBH-5 L3 representative behavioral replaceability proofs."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.contracts.event_delivery import ObservabilityExportPayload
from intergrax.contracts.observability_export import OtlpTransportPort
from intergrax.integrations._shared.conformance import assert_key_value_cache
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.examples.custom_memory_kv import MANIFEST, CustomMemoryKvPlugin
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
