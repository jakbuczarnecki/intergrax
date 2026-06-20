# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ECC-MAINT-02 — dedicated codegen LLM adapter wiring."""

from __future__ import annotations

from typing import Optional, Sequence

import pytest

from intergrax.applications._shared.codegen_llm_resolver import resolve_codegen_llm_adapter
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.codecraft.codegen_adapter import TemplateCodeGenerationAdapter
from intergrax.codecraft.llm_codegen_adapter import LLMCodeGenerationAdapter
from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.codecraft.orchestrator import resolve_codegen_adapter
from intergrax.runtime.codecraft.sandbox_resolver import resolve_craft_sandbox_session
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _StubLLM(LLMAdapter):
    provider = LLMProvider.OLLAMA
    model = "stub-model"

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        del temperature, max_tokens, run_id
        user = next((m.content for m in messages if m.role == "user"), "")
        return LLMAdapterResponse(content=f'print("{user[:12]}")\n')

    def context_window_tokens(self) -> int:
        return 8192


def test_resolve_codegen_llm_adapter_uses_embedded_profile() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={
            "codecraft_profile": CodeCraftProfile(
                mode="autonomous",
                codegen_llm_profile_ref="codegen",
                codegen_llm_profile=LLMProfile(provider=LLMProvider.OLLAMA, model="codegen-model"),
            ),
        },
    )
    adapter = resolve_codegen_llm_adapter(env, producer_adapter=_StubLLM())
    assert isinstance(adapter, LLMCodeGenerationAdapter)
    assert "codegen-model" in adapter.model_id


def test_resolve_codegen_adapter_from_wiring_extras() -> None:
    dedicated = LLMCodeGenerationAdapter(_StubLLM(), profile_ref="codegen")
    ctx = ToolWiringContext(extras={"codecraft_codegen_adapter": dedicated})
    assert resolve_codegen_adapter(ctx) is dedicated


def test_container_tier_falls_back_to_local_sandbox(tmp_path) -> None:
    session = SandboxSession.create(
        tmp_path,
        tenant_id="t",
        task_id="task",
        allowed_operations=frozenset({"run_python"}),
    )
    profile = CodeCraftProfile(mode="autonomous", isolation_tier="container")
    ctx = ToolWiringContext(sandbox_session=session)
    resolved = resolve_craft_sandbox_session(ctx, profile, tenant_id="t", task_id="task")
    assert resolved is session


def test_metrics_snapshot_emitted_on_dispose() -> None:
    from intergrax.runtime.codecraft.trace import CodeCraftTraceEmitter

    emitter = CodeCraftTraceEmitter(run_id="run-1")
    emitter.generation(
        craft_id="c1",
        mode="autonomous",
        iteration=1,
        tenant_id="t",
        task_id="task",
    )
    emitter.static_gate(
        craft_id="c1",
        mode="autonomous",
        passed=False,
        rule_ids=("forbidden_import",),
        tenant_id="t",
        task_id="task",
    )
    emitter.disposed(craft_id="c1", mode="autonomous", tenant_id="t", task_id="task")
    steps = {evt.step for evt in emitter.events}
    assert "codecraft.metrics_snapshot" in steps
    assert emitter.metrics_snapshot().generation_events == 1
    assert emitter.metrics_snapshot().static_gate_failures == 1


def test_template_adapter_when_codegen_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults().model_copy(
        update={"codecraft_profile": CodeCraftProfile(mode="disabled")},
    )
    adapter = resolve_codegen_llm_adapter(env, producer_adapter=_StubLLM())
    assert isinstance(adapter, TemplateCodeGenerationAdapter)
