# © Artur Czarnecki. All rights reserved.

"""PLUG-04-R1 — authoritative RAG plugin evidence from real bootstrap."""

from __future__ import annotations

import importlib.metadata

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
)
from intergrax.applications.contracts.platform_plugin_evidence import (
    PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS,
)
from intergrax.core.plugins.discovery import EP_RAG_RETRIEVERS, reset_entry_point_spec_cache_for_tests
from intergrax.rag.bootstrap import rag_stack_bootstrap
from intergrax.rag.bootstrap import entry_point_load as entry_point_load_module
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    BaseRetrieverPlugin,
    RetrievalHit,
    RetrieverQuery,
)
from lab_application.host.settings import LabApplicationSettings
from lab_application.manifest import build_lab_manifest

pytestmark = [pytest.mark.unit, pytest.mark.no_ci]


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _CountingRetriever(BaseRetriever):
    @classmethod
    def name(cls) -> str:
        return "counting_retriever"

    def retrieve(self, query: RetrieverQuery) -> list[RetrievalHit]:
        return []


class _CountingRetrieverPlugin(BaseRetrieverPlugin):
    factory_calls = 0

    @classmethod
    def create(cls, **kwargs: object) -> BaseRetriever:
        cls.factory_calls += 1
        return _CountingRetriever()

    @classmethod
    def name(cls) -> str:
        return "counting_plugin"


@pytest.fixture(autouse=True)
def _reset_ep_cache() -> None:
    reset_entry_point_spec_cache_for_tests()
    yield
    reset_entry_point_spec_cache_for_tests()
    _CountingRetrieverPlugin.factory_calls = 0


@pytest.mark.no_ci
def test_wire_application_environment_skips_host_rag_bootstrap_when_stack_is_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(**kwargs: object) -> object:
        raise AssertionError(
            "host RAG evidence bootstrap must not run when rag_stack already captured evidence"
        )

    monkeypatch.setattr(
        entry_point_load_module,
        "bootstrap_rag_plugin_load_evidence_for_host_context",
        _forbidden,
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rag.evidence.no-host-rediscovery")
    env = env.model_copy(
        update={
            "context_profile": env.context_profile.model_copy(update={"enable_rag": True}),
        },
    )
    wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-rag-evidence",
        conformance_check=False,
    )


@pytest.mark.no_ci
def test_wire_application_environment_rag_report_matches_rag_stack_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []
    original = rag_stack_bootstrap.create_default_rag_stack

    def _capture(*args: object, **kwargs: object) -> object:
        stack = original(*args, **kwargs)
        captured.append(stack.plugin_load_evidence)
        return stack

    monkeypatch.setattr(rag_stack_bootstrap, "create_default_rag_stack", _capture)
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rag.evidence.same-pass")
    env = env.model_copy(
        update={
            "context_profile": env.context_profile.model_copy(update={"enable_rag": True}),
        },
    )
    wiring = wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-rag-evidence",
        conformance_check=False,
    )

    assert captured
    evidence = captured[0]
    assert evidence is not None
    assert (
        wiring.platform_plugin_evidence.report_for(PLATFORM_PLUGIN_DOMAIN_RAG_RETRIEVERS)
        is evidence.retriever_report
    )


@pytest.mark.no_ci
def test_wire_application_environment_rag_factory_invoked_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint(
                "counting",
                f"{__name__}:_CountingRetrieverPlugin",
                EP_RAG_RETRIEVERS,
            ),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)
    monkeypatch.setattr(
        "intergrax.rag.bootstrap.rag_stack_bootstrap.discover_plugins_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "intergrax.rag.retrievers.bootstrap.retriever_bootstrap.discover_plugins_enabled",
        lambda: True,
    )
    settings = LabApplicationSettings.from_env()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="rag.evidence.factory-once")
    env = env.model_copy(
        update={
            "context_profile": env.context_profile.model_copy(update={"enable_rag": True}),
        },
    )
    wire_application_environment(
        build_lab_manifest(settings),
        env,
        tenant_id="tenant-rag-evidence",
        conformance_check=False,
    )

    assert _CountingRetrieverPlugin.factory_calls == 1
