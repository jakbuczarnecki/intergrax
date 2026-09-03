# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.managed_retrieval import (
    ManagedRetrievalBackend,
    ManagedRetrievalQueryRequest,
    ManagedRetrievalUploadResult,
)
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.providers.managed_retrieval.openai.manifest import MANIFEST as OPENAI_MANAGED_RETRIEVAL
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog, get_entry
from intergrax.integrations.registry.plugin_register import register_integration_plugin
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.tools.providers.openai_vector_store.service import resolve_managed_retrieval
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class FakeManagedRetrievalBackend:
    def ensure_store_exists(self, store_id: str) -> None:
        _ = store_id

    def list_attached_file_ids(self, store_id: str) -> list[str]:
        _ = store_id
        return []

    def upload_folder(
        self,
        store_id: str,
        folder: str | Path,
        *,
        patterns: tuple[str, ...] | list[str],
    ) -> ManagedRetrievalUploadResult:
        _ = store_id, folder, patterns
        return ManagedRetrievalUploadResult(uploaded_names=(), failed_names=())

    def clear_store(self, store_id: str) -> int:
        _ = store_id
        return 0

    def query(self, request: ManagedRetrievalQueryRequest) -> str:
        _ = request
        return "vendor-b"


class VendorBPlugin:
    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        return IntegrationManifest(
            slug="vendor_b_managed_retrieval",
            categories=(IntegrationCategory.MANAGED_RETRIEVAL,),
        )

    @classmethod
    def create_integration(cls, **kwargs: object) -> ManagedRetrievalBackend:
        _ = kwargs
        return FakeManagedRetrievalBackend()


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_managed_retrieval_category_exists() -> None:
    assert IntegrationCategory.MANAGED_RETRIEVAL.value == "managed_retrieval"


def test_profile_field_mapping() -> None:
    from intergrax.integrations.contracts.base import PROFILE_FIELD_BY_CATEGORY

    assert PROFILE_FIELD_BY_CATEGORY["managed_retrieval"] == "managed_retrieval"


def test_profile_binds_managed_retrieval_slug() -> None:
    register_default_integrations()
    profile = IntegrationProfile(managed_retrieval=OPENAI_MANAGED_RETRIEVAL)
    assert profile.slug_for_category(IntegrationCategory.MANAGED_RETRIEVAL) == "openai"
    assert profile.instance_for_category(IntegrationCategory.MANAGED_RETRIEVAL) is None


def test_canonical_catalog_resolves_openai_managed_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    register_default_integrations()
    entry = get_entry("openai")
    assert IntegrationCategory.MANAGED_RETRIEVAL in entry.categories
    profile = IntegrationProfile(managed_retrieval=OPENAI_MANAGED_RETRIEVAL)
    backend = profile.resolve(IntegrationCategory.MANAGED_RETRIEVAL)
    assert isinstance(backend, ManagedRetrievalBackend)


def test_tool_wiring_context_populates_managed_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    register_default_integrations()
    profile = IntegrationProfile(managed_retrieval=OPENAI_MANAGED_RETRIEVAL)
    ctx = ToolWiringContext.from_integration_profile(profile)
    assert ctx.managed_retrieval is not None


def test_tool_wiring_context_without_binding_is_none() -> None:
    register_default_integrations()
    profile = IntegrationProfile.lab()
    ctx = ToolWiringContext.from_integration_profile(profile)
    assert ctx.managed_retrieval is None


def test_resolve_managed_retrieval_not_configured() -> None:
    ctx = ToolWiringContext()
    assert resolve_managed_retrieval(ctx) is None


def test_resolve_managed_retrieval_uses_typed_binding() -> None:
    backend = FakeManagedRetrievalBackend()
    ctx = ToolWiringContext(managed_retrieval=backend)
    assert resolve_managed_retrieval(ctx) is backend


def test_external_plugin_registers_managed_retrieval() -> None:
    register_integration_plugin(VendorBPlugin)
    profile = IntegrationProfile(managed_retrieval=VendorBPlugin)
    backend = profile.resolve(IntegrationCategory.MANAGED_RETRIEVAL)
    assert isinstance(backend, ManagedRetrievalBackend)
    assert backend.query(
        ManagedRetrievalQueryRequest(
            store_id="s1",
            question="q",
            model="m",
            instructions="i",
            max_results=3,
            score_threshold=0.0,
        )
    ) == "vendor-b"


def test_second_fake_provider_resolves_without_tool_changes() -> None:
    register_integration_plugin(VendorBPlugin)
    profile = IntegrationProfile(managed_retrieval="vendor_b_managed_retrieval")
    ctx = ToolWiringContext.from_integration_profile(profile)
    assert ctx.managed_retrieval is not None
    assert isinstance(ctx.managed_retrieval, ManagedRetrievalBackend)
    assert ctx.managed_retrieval.query(
        ManagedRetrievalQueryRequest(
            store_id="s1",
            question="q",
            model="m",
            instructions="i",
            max_results=3,
            score_threshold=0.0,
        )
    ) == "vendor-b"


def test_materialization_module_removed() -> None:
    root = Path(__file__).resolve().parents[3]
    materialization = (
        root / "intergrax" / "integrations" / "providers" / "managed_retrieval" / "materialization.py"
    )
    assert not materialization.exists()


def test_generic_materialization_has_no_openai_switch() -> None:
    root = Path(__file__).resolve().parents[3]
    managed_dir = root / "intergrax" / "integrations" / "providers" / "managed_retrieval"
    forbidden = ("OPENAI_API_KEY", "openai_managed_retrieval", "try_create_openai")
    for path in managed_dir.rglob("*.py"):
        if "openai" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path}: forbidden token {token}"
