# © Artur Czarnecki. All rights reserved.

"""Narrow host test fixtures aligned with production wiring contracts (NPSC-3C-BT)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from intergrax.applications._shared.production_platform_persistence import (
    build_reference_production_platform_persistence,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentStore
from intergrax.skills.registry.profile import SkillProfile
from testing_support.builder import MeteringFakeLLMAdapter


def host_wiring_test_environment(
    environment: ApplicationEnvironmentProfile,
) -> ApplicationEnvironmentProfile:
    """Host smoke/wiring tests exercise factory wiring, not full skill-pack composition."""
    return environment.model_copy(update={"skill_profile": SkillProfile()})


def reference_host_document_store() -> DocumentStore:
    """Durable conditional document store backing for strict host profile pinning."""
    return InMemoryDocumentStore()


def reference_host_platform_persistence_kwargs() -> dict[str, object]:
    """Canonical reference-production platform persistence for strict host factories."""
    platform = build_reference_production_platform_persistence()
    return {
        "document_store": platform.document_store,
        "key_value_cache": platform.kv_store,
    }


def patch_factory_manifest_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory_module: str,
    builder_name: str,
    manifest_builder: Callable[..., object],
) -> None:
    monkeypatch.setattr(f"{factory_module}.{builder_name}", manifest_builder)


def install_host_llm_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def install_diagnostic_cursor_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET",
        "unit-test-diagnostic-problem-list-cursor-secret",
    )
