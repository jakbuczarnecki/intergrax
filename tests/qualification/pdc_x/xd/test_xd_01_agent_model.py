# © Artur Czarnecki. All rights reserved.

"""XD-01 (Agent <-> Model): provider swap by config leaves the agent contract unchanged; mismatches fail closed.

The agent side is the real ``AgentPlatformAdminService`` over the durable SQLite agent-distribution
stores (install, bind, enable, build and activate a revision); the model side is the canonical
resolution path (``RuntimeEnvironmentProfile.llm_profile`` -> ``resolve_llm_adapter`` -> the real
``LLMAdapterRegistry``). The providers are the only fakes (no network, deterministic).

Two clauses of the scenario were recorded as strict xfail findings (FINDING-9, FINDING-10) and are
closed by B-P6-M (T-IGX-036-U): the activated revision pins the profile/model, and the adapter
resolver checks the agent's model needs before the first call.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest
from intergrax.agent_distribution.runtime_revision import RuntimeRevision
from intergrax.contracts.runtime_environment import RuntimeEnvironmentProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError, LLMFeature
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent, LLMStreamEventKind
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.registry.registration_contract import LLMProviderNotConfiguredError
from intergrax.runtime.wiring.llm_resolver import resolve_llm_adapter
from tests.qualification.pdc_x.d1 import _d1_harness as h
from tests.qualification.pdc_x.xd._support import REPO_ROOT, build_sqlite_agent_stack

pytestmark = [pytest.mark.integration]

NOOP_TOOL = {"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}
USER = [ChatMessage(role="user", content="hello")]
_TOOLS: Any = [NOOP_TOOL]


class _PlainAdapter(BaseLLMAdapter):
    """External provider stand-in that only buffers: it declares no streaming and no tool streaming."""

    def __init__(self, *, provider: str, model: str | None = None, **_: object) -> None:
        super().__init__()
        self.provider = provider
        self.model = model or f"{provider}-default"
        self.calls: list[str] = []

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def generate_messages(self, messages: Sequence[ChatMessage], **_: object) -> LLMAdapterResponse:
        self.calls.append("generate_messages")
        return LLMAdapterResponse(content=f"answer from {self.provider}", provider=self.provider, model=self.model)

    def generate_with_tools(self, messages, tools, **_: object) -> LLMAdapterResponse:  # type: ignore[no-untyped-def]
        self.calls.append("generate_with_tools")
        return LLMAdapterResponse(content="tools", provider=self.provider, model=self.model)


class _RichAdapter(_PlainAdapter):
    """Honest full declaration: streaming, tools and tool streaming."""

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_streaming(self) -> bool:
        return True

    def stream_messages(self, messages: Sequence[ChatMessage], **_: object) -> Iterable[LLMStreamEvent]:
        self.calls.append("stream_messages")
        yield LLMStreamEvent(
            kind=LLMStreamEventKind.FINAL,
            response=LLMAdapterResponse(content="s", provider=self.provider, model=self.model),
        )

    def stream_with_tools(self, messages, tools, **_: object) -> Iterable[LLMStreamEvent]:  # type: ignore[no-untyped-def]
        self.calls.append("stream_with_tools")
        yield LLMStreamEvent(
            kind=LLMStreamEventKind.FINAL,
            response=LLMAdapterResponse(content="t", provider=self.provider, model=self.model),
        )


def _register_providers() -> None:
    def rich(**kw: Any) -> _RichAdapter:
        return _RichAdapter(provider="xd01_rich", **kw)

    def plain(**kw: Any) -> _PlainAdapter:
        return _PlainAdapter(provider="xd01_plain", **kw)

    LLMAdapterRegistry.register("xd01_rich", rich, override=True)
    LLMAdapterRegistry.register("xd01_plain", plain, override=True)


def _env(provider: str, model: str) -> RuntimeEnvironmentProfile:
    return RuntimeEnvironmentProfile(profile_id="xd01.env", llm_profile=LLMProfile(provider=provider, model=model))


def _digest(payload: str) -> str:
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _activated(tmp_path: Path):  # type: ignore[no-untyped-def]
    stack = build_sqlite_agent_stack(tmp_path / "agents.sqlite3")
    release = h.release("1.0.0")
    h.install_bind_enable(stack, release)
    h.build_revision(stack, "rev-1")
    h.activate_revision(stack, "rev-1")
    return stack, release


def _agent_identity(stack, release):  # type: ignore[no-untyped-def]
    """Everything that identifies the activated agent: contract digest, revision and roster identity."""
    serving = stack.service.inspect_serving(application_id=h.APP, application_environment_id=h.ENV)
    revision = stack.service.inspect_revision(
        application_id=h.APP, application_environment_id=h.ENV, runtime_revision_id="rev-1"
    )
    roster = stack.service.inspect_effective_roster(application_id=h.APP, application_environment_id=h.ENV)
    contract = release.contracts[0]
    return {
        "contract_digest": _digest(contract.model_dump_json()),
        "serving_revision": serving.traffic_serving_revision_id,
        "revision_package_digests": tuple(revision.installed_agent_package_digests),
        "revision_graph_digest": revision.runtime_graph_digest,
        "roster_revision": roster.effective_roster_revision_id,
        "roster_agents": tuple((e.logical_agent_id, e.package_digest) for e in roster.entries),
    }


def test_xd01_provider_swap_by_config_leaves_the_activated_agent_unchanged(tmp_path: Path) -> None:
    _register_providers()
    stack, release = _activated(tmp_path)
    before = _agent_identity(stack, release)
    assert before["serving_revision"] == "rev-1"

    adapter_a = resolve_llm_adapter(_env("xd01_rich", "model-a"))
    answer_a = adapter_a.generate_messages(USER)
    # The operator swaps the provider in configuration only: no agent lifecycle operation is involved.
    adapter_b = resolve_llm_adapter(_env("xd01_plain", "model-b"))
    answer_b = adapter_b.generate_messages(USER)

    assert (adapter_a.provider, adapter_b.provider) == ("xd01_rich", "xd01_plain")  # type: ignore[attr-defined]
    assert (answer_a.provider, answer_b.provider) == ("xd01_rich", "xd01_plain")
    after = _agent_identity(stack, release)
    assert after == before, "a provider swap by configuration must not change the agent contract or revision"
    # ... and it also survives a cold reopen of the durable stores.
    assert _agent_identity(build_sqlite_agent_stack(tmp_path / "agents.sqlite3"), release) == before


def test_xd01_an_unconfigured_provider_fails_closed_instead_of_defaulting() -> None:
    with pytest.raises(LLMProviderNotConfiguredError):
        resolve_llm_adapter(RuntimeEnvironmentProfile(profile_id="xd01.empty"))


def test_xd01_capability_mismatch_fails_closed_before_any_provider_call() -> None:
    _register_providers()
    rich = resolve_llm_adapter(_env("xd01_rich", "model-a"))
    plain = resolve_llm_adapter(_env("xd01_plain", "model-b"))

    # The agent needs tool streaming. The swapped-in provider does not declare it.
    LLMAdapterRegistry.require_capabilities(rich, LLMFeature.TOOL_STREAMING)  # the rich provider is fine
    with pytest.raises(LLMCapabilityMismatchError) as caught:
        LLMAdapterRegistry.require_capabilities(plain, LLMFeature.TOOL_STREAMING)
    assert caught.value.features == (LLMFeature.TOOL_STREAMING,)

    # Even without a preflight the provider call itself is refused, never silently downgraded.
    with pytest.raises(LLMCapabilityMismatchError):
        list(plain.stream_with_tools(USER, _TOOLS))
    assert getattr(plain, "calls", []) == [], "no provider method was invoked"
    assert [e.is_final for e in rich.stream_with_tools(USER, _TOOLS)] == [True]


def test_xd01_the_activated_revision_pins_the_model_profile() -> None:
    pinned = [
        name
        for name in RuntimeRevision.model_fields
        if "llm" in name or "model_profile" in name or "profile_digest" in name
    ]
    assert pinned, "RuntimeRevision has no model/profile pin field"


def test_xd01_required_model_capabilities_are_checked_at_composition_time() -> None:
    callers: list[str] = []
    for root in ("intergrax", "agents", "applications"):
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "llm_provider_registry.py" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "require_capabilities" not in text:
                continue
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "require_capabilities":
                    callers.append(path.relative_to(REPO_ROOT).as_posix())
    assert callers, "no production module calls LLMAdapterRegistry.require_capabilities"
