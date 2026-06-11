# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent scaffold CLI — typed cognitive-pattern agents under ``agents/`` (ACP default)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent

from intergrax.scaffold.adr_templates import write_agent_adr_scaffold
from intergrax.scaffold.tracing_templates import write_agent_tracing_scaffold
from intergrax.scaffold.doc_templates import (
    render_agent_architecture_doc,
    render_agent_implementation_plan,
)

SCAFFOLD_PATTERNS: dict[str, str] = {
    "reflex": "ReflexAgent",
    "react": "ReActAgent",
    "plan_execute": "PlanExecuteAgent",
    "decomposition": "DecompositionAgent",
    "reflection": "ReflectionAgent",
}

_PATTERN_ENUM_MEMBER: dict[str, str] = {
    "reflex": "REFLEX",
    "react": "REACT",
    "plan_execute": "PLAN_EXECUTE",
    "decomposition": "DECOMPOSITION",
    "reflection": "REFLECTION",
}

_PATTERN_IMPORTS: dict[str, str] = {
    "reflex": "from intergrax.agents.authoring.patterns.reflex import ReflexAgent",
    "react": "from intergrax.agents.authoring.patterns.react import ReActAgent",
    "plan_execute": "from intergrax.agents.authoring.patterns.plan_execute import PlanExecuteAgent",
    "decomposition": "from intergrax.agents.authoring.patterns.decomposition import DecompositionAgent",
    "reflection": "from intergrax.agents.authoring.patterns.reflection import ReflectionAgent",
}


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or slug[0].isdigit():
        raise ValueError(f"Invalid agent name: {name!r}")
    return slug


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_")) + "Agent"


def _pascal_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _agent_py(slug: str, class_name: str, primary_capability: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        from __future__ import annotations

        from {slug}.capabilities import CAPABILITIES
        from {slug}.contract import build_agent_contract
        from {slug}.steps.pipeline import build_pipeline, run_domain_step
        from intergrax.agents.agent_contract import Agent
        from intergrax.contracts.agent_decision import AgentDecision
        from intergrax.contracts.agent_step import AgentStep, StepOutput
        from intergrax.contracts.capability import CapabilityMatchResult
        from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
        from intergrax.runtime.task.task import TaskContext
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
        from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
        from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
        from intergrax.runtime.nexus.session.session_manager import SessionManager
        from intergrax.agents.authoring.uaep_pipeline_bridge import pipeline_agent_steps, pipeline_step_complete


        class {class_name}(Agent):
            """UAEP-first scaffolded agent — replace domain logic in ``steps/`` and ``prompts/``."""

            def get_contract(self):
                return build_agent_contract()

            def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
                capability = task_context.capability
                supported = set(CAPABILITIES)
                if capability is None or capability in supported:
                    return CapabilityMatchResult(
                        matched=True,
                        agent_id="{slug}",
                        matched_capabilities=list(supported),
                        score=1.0,
                        rationale="capability match",
                    )
                return CapabilityMatchResult(matched=False, rationale="capability not supported")

            def build_context(self, request: RuntimeRequest) -> RuntimeContext:
                from intergrax.agents.defaults import harness_production_mode

                config = RuntimeConfig(
                    llm_adapter=build_pipeline().llm_adapter,
                    enable_rag=False,
                    production_mode=harness_production_mode(),
                    tenant_id=request.tenant_id,
                )
                config.pipeline = build_pipeline().pipeline
                session_manager = SessionManager(storage=InMemorySessionStorage())
                return RuntimeContext.build(config=config, session_manager=session_manager)

            def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
                _ = context
                contract = self.get_contract()
                return pipeline_agent_steps(
                    step_id="{slug}_step",
                    step_name="{slug}_step",
                    trace_label="{primary_capability}",
                    allowed_tools=list(contract.allowed_tools),
                )

            async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
                return await run_domain_step(step, ctx)

            def decide_after_step(
                self,
                step: AgentStep,
                output: StepOutput | None,
                ctx: RuntimeExecutionContext,
            ) -> AgentDecision:
                _ = step, output, ctx
                return pipeline_step_complete(reason="{slug} step finished")
        '''
    )


def _reference_agent_py(slug: str, class_name: str, primary_capability: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        from __future__ import annotations

        from {slug}.capabilities import CAPABILITIES
        from {slug}.contract import build_agent_contract
        from {slug}.steps.pipeline import build_pipeline, run_domain_step
        from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
        from intergrax.agents.reference_harness import (
            LabHarnessContext,
            build_lab_agent_runtime_context,
            default_reference_harness,
        )
        from intergrax.contracts.agent_decision import AgentDecision
        from intergrax.contracts.agent_step import AgentStep, StepOutput
        from intergrax.contracts.capability import CapabilityMatchResult
        from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
        from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
        from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
        from intergrax.runtime.task.task import TaskContext
        from intergrax.agents.authoring.uaep_pipeline_bridge import pipeline_agent_steps, pipeline_step_complete


        class {class_name}(HarnessReferenceAgent):
            """Harness reference agent — inject ``LabHarnessContext`` from Tier-3 host builders."""

            def __init__(self, harness: LabHarnessContext | None = None) -> None:
                self._harness = harness or default_reference_harness()

            def get_contract(self):
                return build_agent_contract()

            def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
                capability = task_context.capability
                supported = set(CAPABILITIES)
                if capability is None or capability in supported:
                    return CapabilityMatchResult(
                        matched=True,
                        agent_id="{slug}",
                        matched_capabilities=list(supported),
                        score=1.0,
                        rationale="capability match",
                    )
                return CapabilityMatchResult(matched=False, rationale="capability not supported")

            def build_context(self, request: RuntimeRequest) -> RuntimeContext:
                built = build_pipeline()
                return build_lab_agent_runtime_context(
                    request=request,
                    llm_adapter=built.llm_adapter,
                    harness=self._harness,
                    pipeline=built.pipeline,
                )

            def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
                _ = context
                contract = self.get_contract()
                return pipeline_agent_steps(
                    step_id="{slug}_step",
                    step_name="{slug}_step",
                    trace_label="{primary_capability}",
                    allowed_tools=list(contract.allowed_tools),
                )

            async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
                return await run_domain_step(step, ctx)

            def decide_after_step(
                self,
                step: AgentStep,
                output: StepOutput | None,
                ctx: RuntimeExecutionContext,
            ) -> AgentDecision:
                _ = step, output, ctx
                return pipeline_step_complete(reason="{slug} step finished")
        '''
    )


def _normalize_pattern(pattern: str | None) -> str | None:
    if pattern is None:
        return None
    normalized = pattern.strip().lower().replace("-", "_")
    if normalized not in SCAFFOLD_PATTERNS:
        allowed = ", ".join(sorted(SCAFFOLD_PATTERNS))
        raise ValueError(f"Unknown pattern {pattern!r}; choose one of: {allowed}")
    return normalized


def _acp_agent_py(
    slug: str,
    class_name: str,
    primary_capability: str,
    *,
    pattern: str,
) -> str:
    base_class = SCAFFOLD_PATTERNS[pattern]
    pattern_import = _PATTERN_IMPORTS[pattern]
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from {slug}.capabilities import CAPABILITIES
        from {slug}.contract import build_agent_contract
        {pattern_import}
        from intergrax.agents.authoring.patterns.types import (
            AgentEvaluation,
            CognitiveEvaluation,
            Observation,
            ReasoningResult,
        )
        from intergrax.contracts.agent_contract_meta import AgentRiskLevel
        from intergrax.contracts.agent_step_context import AgentStepContext
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
        from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
        from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
        from intergrax.runtime.nexus.session.session_manager import SessionManager
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
        from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
        from intergrax.memory.conversational_memory import ChatMessage
        from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
        from typing import Optional, Sequence


        class _{ _pascal_name(slug) }StubLLM(LLMAdapter):
            provider = "{slug}"
            model = "{slug}-stub"

            @property
            def context_window_tokens(self) -> int:
                return 128_000

            def generate_messages(
                self,
                messages: Sequence[ChatMessage],
                *,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                run_id: Optional[str] = None,
            ) -> LLMAdapterResponse:
                for msg in reversed(messages):
                    if msg.content:
                        return build_adapter_response(content=msg.content[:200])
                return build_adapter_response(content="{slug}: (empty)")


        class {class_name}({base_class}):
            """Typed cognitive-pattern agent — implement hooks below (ACP §32.0)."""

            contract_id = "{slug}"
            capabilities = tuple(CAPABILITIES)
            agent_name = "{class_name}"
            agent_description = "Scaffolded {pattern} agent"
            risk_level = AgentRiskLevel.LOW
            max_steps = 10

            def build_context(self, request: RuntimeRequest) -> RuntimeContext:
                from intergrax.agents.defaults import harness_production_mode

                config = RuntimeConfig(
                    llm_adapter=_{_pascal_name(slug)}StubLLM(),
                    enable_rag=False,
                    production_mode=harness_production_mode(),
                    tenant_id=request.tenant_id,
                )
                session_manager = SessionManager(storage=InMemorySessionStorage())
                return RuntimeContext.build(config=config, session_manager=session_manager)

            async def perceive(self, step_ctx: AgentStepContext) -> Observation:
                _ = step_ctx
                return Observation(summary="TODO: domain perception")

            async def reason(
                self,
                step_ctx: AgentStepContext,
                observation: Observation,
            ) -> ReasoningResult:
                _ = step_ctx
                return ReasoningResult(thought=observation.summary)

            async def act(
                self,
                step_ctx: AgentStepContext,
                reasoning: ReasoningResult,
            ) -> dict[str, object]:
                _ = step_ctx
                return {{"summary": reasoning.thought, "capability": "{primary_capability}"}}

            def evaluate(
                self,
                step_ctx: AgentStepContext,
                output: dict[str, object],
            ) -> AgentEvaluation:
                _ = step_ctx
                return AgentEvaluation(
                    verdict=CognitiveEvaluation.COMPLETE,
                    reason="{pattern} scaffold complete",
                    confidence=0.9,
                )
        '''
    )


def _acp_contract_py(slug: str, class_name: str, primary_capability: str, *, pattern: str) -> str:
    enum_member = _PATTERN_ENUM_MEMBER[pattern]
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
        from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
        from intergrax.contracts.agent_run_enums import CognitivePattern
        from intergrax.agents.authoring.patterns.base import PATTERN_VERSION
        from {slug}.capabilities import CAPABILITIES

        _PATTERN = CognitivePattern.{enum_member}


        def build_agent_contract() -> AgentContract:
            return AgentContract(
                id="{slug}",
                name="{class_name}",
                description="Scaffolded typed {pattern} agent.",
                version="0.1.0",
                capabilities=CAPABILITIES,
                skills=[],
                extra_tools=[],
                risk_level=AgentRiskLevel.LOW,
                lifecycle_state=AgentLifecycleState.DEVELOPMENT,
                owner_team="platform",
                max_steps=10,
                cognitive_pattern=_PATTERN,
                pattern_version=PATTERN_VERSION,
                pattern_config={{"primary_capability": "{primary_capability}"}},
            )
        '''
    )


def _acp_test_agent_py(slug: str, class_name: str, primary_capability: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        import pytest

        from {slug}.{slug}_agent import {class_name}
        from {slug}.contract import build_agent_contract
        from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
        from intergrax.contracts.agent_run_enums import AgentRunStatus


        @pytest.mark.asyncio
        @pytest.mark.unit
        @pytest.mark.gate
        async def test_{slug}_typed_run_smoke():
            agent = {class_name}()
            contract = build_agent_contract()
            assert contract.cognitive_pattern is not None
            result = await agent.run(
                AgentRunRequest(
                    input="scaffold smoke",
                    identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                    agent_id=contract.id,
                )
            )
            assert result.status == AgentRunStatus.SUCCEEDED
            assert "{primary_capability}" in str(result.output)
        '''
    )


def _contract_py(slug: str, class_name: str, primary_capability: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
        from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
        from {slug}.capabilities import CAPABILITIES

        # Register skill packs on the contract — see docs/architecture/SKILLS.md


        def build_agent_contract() -> AgentContract:
            return AgentContract(
                id="{slug}",
                name="{class_name}",
                description="Scaffolded UAEP agent for Intergrax experiments.",
                version="0.1.0",
                capabilities=CAPABILITIES,
                skills=[],
                extra_tools=[],
                risk_level=AgentRiskLevel.LOW,
                lifecycle_state=AgentLifecycleState.DEVELOPMENT,
                owner_team="platform",
                max_steps=10,
            )
        '''
    )


def _capabilities_py(slug: str, capabilities: list[str]) -> str:
    caps_lines = ",\n    ".join(repr(c) for c in capabilities)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        """Capability ids exposed by the {slug} agent."""

        CAPABILITIES: list[str] = [
            {caps_lines}
        ]
        '''
    )


def _steps_pipeline_py(slug: str, primary_capability: str) -> str:
    pascal = _pascal_name(slug)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Optional, Sequence

        from intergrax.agents.authoring.uaep_pipeline_bridge import run_pipeline_step
        from intergrax.contracts.agent_step import AgentStep, StepOutput
        from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
        from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
        from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
        from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
        from intergrax.memory.conversational_memory import ChatMessage
        from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
        from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
        from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
        from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
        from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
        from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS


        class _{pascal}LLMStub(LLMAdapter):
            provider = "{slug}"
            model = "{slug}-stub"

            @property
            def context_window_tokens(self) -> int:
                return 128_000

            def generate_messages(
                self,
                messages: Sequence[ChatMessage],
                *,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                run_id: Optional[str] = None,
            ) -> LLMAdapterResponse:
                for msg in reversed(messages):
                    content = msg.content or ""
                    if content:
                        return build_adapter_response(content=f"{slug}: {{content[:200]}}")
                return build_adapter_response(content="{slug}: (empty)")


        class _{pascal}Pipeline(RuntimePipeline):
            async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
                await RuntimeStepRunner.execute_pipeline(
                    [*SETUP_STEPS, PersistAndBuildAnswerStep()],
                    state,
                )
                message = (state.request.message or "").strip()
                answer = f"{slug}: {{message}}"
                if state.runtime_answer is not None:
                    state.runtime_answer.answer = answer
                if state.runtime_answer is None:
                    raise RuntimeError("{slug} pipeline did not produce runtime_answer.")
                return state.runtime_answer


        @dataclass(frozen=True)
        class PipelineBundle:
            llm_adapter: LLMAdapter
            pipeline: _{pascal}Pipeline


        def build_pipeline() -> PipelineBundle:
            return PipelineBundle(
                llm_adapter=_{pascal}LLMStub(),
                pipeline=_{pascal}Pipeline(),
            )


        async def run_domain_step(step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
            """Replace with multi-step domain logic as the agent grows."""
            _ = step
            return await run_pipeline_step(step, ctx)
        '''
    )


def _schemas_init() -> str:
    return dedent(
        '''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        """Pydantic request/response models for the agent."""
        '''
    )


def _prompts_system_md(slug: str) -> str:
    return dedent(
        f"""\
        # {slug} — system prompt (draft)

        You are a scaffolded Intergrax agent. Replace this prompt with domain instructions.

        Capability focus: see ``capabilities.py``.
        """
    )


def _test_agent_py(slug: str, class_name: str, primary_capability: str) -> str:
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        import pytest

        from {slug}.{slug}_agent import {class_name}
        from intergrax.runtime.nexus.nexus_loop import NexusLoop
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from intergrax.runtime.task.task import Task, TaskContext, TaskState


        @pytest.mark.asyncio
        @pytest.mark.integration
        @pytest.mark.gate
        async def test_{slug}_agent_runs_through_nexus():
            registry = AgentRegistry()
            registry.register({class_name}())
            loop = NexusLoop(registry)
            result = await loop.handle_task(
                Task(
                    tenant_id="t1",
                    user_id="u1",
                    message="scaffold smoke",
                    context=TaskContext(capability="{primary_capability}"),
                )
            )
            assert result.state == TaskState.COMPLETED
            assert "scaffold smoke" in result.answer
            assert result.agent_id == "{slug}"
        '''
    )


def _notebook_stub(slug: str, primary_capability: str) -> str:
    return dedent(
        f"""\
        {{
         "cells": [
          {{
           "cell_type": "markdown",
           "metadata": {{}},
           "source": ["# {slug} experiment\\n", "\\n", "Run via NexusLoop or ``applications/lab_application``."]
          }},
          {{
           "cell_type": "code",
           "execution_count": null,
           "metadata": {{}},
           "outputs": [],
           "source": [
            "from {slug}.{slug}_agent import {_class_name(slug)}\\n",
            "from intergrax.runtime.nexus.nexus_loop import NexusLoop\\n",
            "from intergrax.runtime.registry.agent_registry import AgentRegistry\\n",
            "from intergrax.runtime.task.task import Task, TaskContext\\n",
            "\\n",
            "registry = AgentRegistry()\\n",
            "registry.register({_class_name(slug)}())\\n",
            "loop = NexusLoop(registry)\\n",
            "task = Task(tenant_id='t1', user_id='u1', message='hello', context=TaskContext(capability='{primary_capability}'))\\n",
            "result = await loop.handle_task(task)\\n",
            "result.answer"
           ]
          }}
         ],
         "metadata": {{"kernelspec": {{"display_name": "Python 3", "language": "python", "name": "python3"}}}},
         "nbformat": 4,
         "nbformat_minor": 5
        }}
        """
    )


def _readme(slug: str, class_name: str, capabilities: list[str]) -> str:
    caps = ", ".join(f"`{c}`" for c in capabilities)
    return dedent(
        f"""\
        # {slug} agent

        UAEP-first scaffold. Full process: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md) (single canonical guide).

        ## Docs

        - [`ARCHITECTURE.md`](ARCHITECTURE.md) — purpose, contracts, runtime layout
        - [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — task queue and verification
        - [`adr/README.md`](adr/README.md) — architecture decision records

        ## Quick start

        1. Implement domain logic in `steps/`
        2. Run smoke test: `uv run pytest agents/{slug}/tests -q`
        3. For lab HTTP: register in `applications/lab_application/host/wiring.py` (see guide Step 4C)

        ## Register (programmatic)

        ```python
        from intergrax.runtime.registry.agent_registry import AgentRegistry
        from {slug}.{slug}_agent import {class_name}

        registry = AgentRegistry()
        registry.register({class_name}())
        ```

        See **Step 4** in guides/AGENT_CREATION_GUIDE.md for all registration contexts.

        ## Capabilities

        {caps}

        ## Layout

        - ``{slug}_agent.py`` — Agent class (UAEP)
        - ``contract.py`` / ``capabilities.py`` — AgentContract
        - ``steps/`` — domain execution
        - ``prompts/`` — prompt assets
        - ``schemas/`` — I/O models
        - ``tracing/`` — DiagnosticPayload extensions (OBS extension SDK)
        - ``tests/`` — agent smoke tests
        - ``notebooks/`` — interactive experiments
        - ``adr/`` — architecture decision records
        """
    )


def create_acp_pattern_agent(
    *,
    name: str,
    capabilities: list[str],
    root: Path,
    pattern: str,
    force: bool = False,
    minimal: bool = False,
) -> Path:
    """Create typed cognitive-pattern agent (no UAEP boilerplate — ACP-8)."""
    normalized = _normalize_pattern(pattern)
    assert normalized is not None
    slug = _slug(name)
    class_name = _class_name(slug)
    if not capabilities:
        capabilities = [f"{slug}.basic"]

    agents_root = root / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    agents_init = agents_root / "__init__.py"
    if not agents_init.exists():
        agents_init.write_text("", encoding="utf-8")

    target = agents_root / slug
    if target.exists() and not force:
        raise FileExistsError(f"Agent directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=True)

    primary_capability = capabilities[0]
    _write(
        target / f"{slug}_agent.py",
        _acp_agent_py(slug, class_name, primary_capability, pattern=normalized),
        force=force,
    )
    _write(
        target / "contract.py",
        _acp_contract_py(slug, class_name, primary_capability, pattern=normalized),
        force=force,
    )
    _write(target / "capabilities.py", _capabilities_py(slug, capabilities), force=force)
    _write(target / "tests" / "__init__.py", "", force=force)
    _write(
        target / "tests" / f"test_{slug}_agent.py",
        _acp_test_agent_py(slug, class_name, primary_capability),
        force=force,
    )
    _write(target / "prompts" / "system.md", _prompts_system_md(slug), force=force)
    _write(target / "schemas" / "__init__.py", _schemas_init(), force=force)
    _write(
        target / "ARCHITECTURE.md",
        render_agent_architecture_doc(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=False,
            pattern=normalized,
        ),
        force=force,
    )
    _write(
        target / "IMPLEMENTATION_PLAN.md",
        render_agent_implementation_plan(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=False,
            pattern=normalized,
        ),
        force=force,
    )
    if not minimal:
        _write(
            target / "README.md",
            dedent(
                f"""\
                # {slug} agent ({normalized})

                Typed **{normalized}** cognitive pattern — ``on_next_step`` via ``CognitiveAgent``.
                See ``docs/guides/AGENT_CREATION_GUIDE.md`` § cognitive patterns.
                """
            ),
            force=force,
        )
    _write(
        target / "__init__.py",
        dedent(
            f'''\
            from {slug}.{slug}_agent import {class_name}

            __all__ = ["{class_name}"]
            '''
        ),
        force=force,
    )
    write_agent_adr_scaffold(agent_dir=target, slug=slug, force=force)
    write_agent_tracing_scaffold(target=target, slug=slug, force=force)
    return target


def create_uaep_agent(
    *,
    name: str,
    capabilities: list[str],
    root: Path,
    force: bool = False,
    reference: bool = False,
    minimal: bool = False,
) -> Path:
    """Legacy UAEP scaffold — use typed ``create_acp_pattern_agent`` for new agents (DEBT-ACP-05)."""
    slug = _slug(name)
    class_name = _class_name(slug)
    if not capabilities:
        capabilities = [f"{slug}.basic"]

    agents_root = root / "agents"
    agents_root.mkdir(parents=True, exist_ok=True)
    agents_init = agents_root / "__init__.py"
    if not agents_init.exists():
        agents_init.write_text("", encoding="utf-8")

    target = agents_root / slug
    if target.exists() and not force:
        raise FileExistsError(f"Agent directory already exists: {target}")

    target.mkdir(parents=True, exist_ok=True)

    primary_capability = capabilities[0]

    agent_src = _reference_agent_py if reference else _agent_py
    _write(target / f"{slug}_agent.py", agent_src(slug, class_name, primary_capability), force=force)
    _write(target / "contract.py", _contract_py(slug, class_name, primary_capability), force=force)
    _write(target / "capabilities.py", _capabilities_py(slug, capabilities), force=force)
    _write(target / "steps" / "__init__.py", "", force=force)
    _write(target / "steps" / "pipeline.py", _steps_pipeline_py(slug, primary_capability), force=force)
    _write(target / "schemas" / "__init__.py", _schemas_init(), force=force)
    _write(target / "prompts" / "system.md", _prompts_system_md(slug), force=force)
    _write(target / "tests" / "__init__.py", "", force=force)
    _write(target / "tests" / f"test_{slug}_agent.py", _test_agent_py(slug, class_name, primary_capability), force=force)
    if not minimal:
        _write(target / "notebooks" / f"01_{slug}_experiment.ipynb", _notebook_stub(slug, primary_capability), force=force)
    _write(
        target / "ARCHITECTURE.md",
        render_agent_architecture_doc(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=reference,
        ),
        force=force,
    )
    _write(
        target / "IMPLEMENTATION_PLAN.md",
        render_agent_implementation_plan(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=reference,
        ),
        force=force,
    )
    _write(
        target / "__init__.py",
        dedent(
            f'''\
            from {slug}.{slug}_agent import {class_name}

            __all__ = ["{class_name}"]
            '''
        ),
        force=force,
    )
    if not minimal:
        _write(target / "README.md", _readme(slug, class_name, capabilities), force=force)
    write_agent_adr_scaffold(agent_dir=target, slug=slug, force=force)
    write_agent_tracing_scaffold(target=target, slug=slug, force=force)

    return target


def create_agent(
    *,
    name: str,
    capabilities: list[str],
    root: Path,
    force: bool = False,
    reference: bool = False,
    minimal: bool = False,
    pattern: str | None = None,
    uaep: bool = False,
) -> Path:
    if reference or uaep:
        return create_uaep_agent(
            name=name,
            capabilities=capabilities,
            root=root,
            force=force,
            reference=reference,
            minimal=minimal,
        )
    effective_pattern = _normalize_pattern(pattern or "reflex")
    assert effective_pattern is not None
    return create_acp_pattern_agent(
        name=name,
        capabilities=capabilities,
        root=root,
        pattern=effective_pattern,
        force=force,
        minimal=minimal,
    )


def build_parser() -> argparse.ArgumentParser:
    from intergrax.scaffold.cli import build_parser as _build_cli_parser

    return _build_cli_parser()


def main(argv: list[str] | None = None) -> int:
    from intergrax.scaffold.cli import main as _cli_main

    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
