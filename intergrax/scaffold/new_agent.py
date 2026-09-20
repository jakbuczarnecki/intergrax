# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent scaffold CLI — typed cognitive-pattern agents under ``agents/`` (ACP default)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent

from intergrax.scaffold.agent_layout import (
    agent_docs_dir,
    write_agent_journal_scaffold,
)
from intergrax.scaffold.adr_templates import write_agent_adr_scaffold
from intergrax.scaffold.signal_templates import write_agent_signal_scaffold
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


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"File already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _normalize_pattern(pattern: str | None) -> str | None:
    if pattern is None:
        return None
    normalized = pattern.strip().lower().replace("-", "_")
    if normalized not in SCAFFOLD_PATTERNS:
        allowed = ", ".join(sorted(SCAFFOLD_PATTERNS))
        raise ValueError(f"Unknown pattern {pattern!r}; choose one of: {allowed}")
    return normalized


def _domain_job_py(slug: str) -> str:
    step_id = f"{slug}_step"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from intergrax.contracts.agent_step_context import AgentStepContext
        from intergrax.runtime.nexus.agents.runtime_tool_helpers import exec_ctx_from_step, request_metadata

        DOMAIN_STEP_ID = "{step_id}"


        async def run_domain_job(step_ctx: AgentStepContext) -> dict[str, object]:
            """Cursor implementation point — see intergrax/agents/authoring/runtime_tool_helpers.py."""
            _ = exec_ctx_from_step(step_ctx), request_metadata(None), DOMAIN_STEP_ID
            answer = "{slug}: domain job not implemented"
            return {{
                "summary": answer,
                "answer": answer,
                "run_id": step_ctx.run_id,
                "domain_summary": {{"used": False, "reason": "not_implemented"}},
            }}
        '''
    )


def _acp_agent_hooks(
    slug: str,
    class_name: str,
    primary_capability: str,
    *,
    pattern: str,
) -> str:
    return f"""\
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
                _ = reasoning
                from {slug}.steps.domain_job import run_domain_job
                return await run_domain_job(step_ctx)

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
"""


def _acp_agent_py(
    slug: str,
    class_name: str,
    primary_capability: str,
    *,
    pattern: str,
    reference: bool = False,
) -> str:
    base_class = SCAFFOLD_PATTERNS[pattern]
    pattern_import = _PATTERN_IMPORTS[pattern]
    hooks = _acp_agent_hooks(slug, class_name, primary_capability, pattern=pattern)
    if reference:
        class_doc = (
            "Harness reference agent — ACP hooks only; host/runtime owns LLM and execution wiring."
        )
        agent_description = f"Scaffolded {pattern} reference agent"
    else:
        class_doc = "Typed cognitive-pattern agent — implement hooks below (ACP §32.0)."
        agent_description = f"Scaffolded {pattern} agent"
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.

        from __future__ import annotations

        from {slug}.capabilities import CAPABILITIES
        {pattern_import}
        from intergrax.agents.authoring.patterns.types import (
            AgentEvaluation,
            CognitiveEvaluation,
            Observation,
            ReasoningResult,
        )
        from intergrax.contracts.agent_contract_meta import AgentRiskLevel
        from intergrax.contracts.agent_step_context import AgentStepContext

        # LLM / catalog (Tier-3 host): intergrax/llm_adapters/USAGE.md — LLMProfile, ModelCatalog,
        # optional LLMRoutingProfile on ApplicationEnvironmentProfile.
        # Offline smoke tests use Agent.run(AgentRunRequest) without constructing execution runtime.


        class {class_name}({base_class}):
            """{class_doc}"""

            contract_id = "{slug}"
            capabilities = tuple(CAPABILITIES)
            agent_name = "{class_name}"
            agent_description = "{agent_description}"
            risk_level = AgentRiskLevel.LOW
            max_steps = 10
{hooks}
        '''
    )


def _acp_contract_py(
    slug: str, class_name: str, primary_capability: str, *, pattern: str
) -> str:
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
        from intergrax.dev_support.execution_identity_scope import canonical_agent_run_smoke_scope


        @pytest.mark.asyncio
        @pytest.mark.unit
        @pytest.mark.gate
        async def test_{slug}_typed_run_smoke():
            agent = {class_name}()
            contract = build_agent_contract()
            assert contract.cognitive_pattern is not None
            with canonical_agent_run_smoke_scope(
                "scaffold-smoke",
                tenant_id="t1",
                principal_id="u1",
            ):
                result = await agent.run(
                    AgentRunRequest(
                        input="scaffold smoke",
                        identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                        agent_id=contract.id,
                    )
                )
            assert result.status == AgentRunStatus.SUCCEEDED
            assert "{slug}" in str(result.output)
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


def _notebook_stub(slug: str, primary_capability: str) -> str:
    return dedent(
        f"""\
        {{
         "cells": [
          {{
           "cell_type": "markdown",
           "metadata": {{}},
           "source": ["# {slug} experiment (historical / offline)\\n", "\\n", "Prefer ``agent.run()`` — not a canonical production quickstart. For lab/product HTTP use Agent Distribution lifecycle (see AGENT_CREATION_GUIDE Step 4)."]
          }},
          {{
           "cell_type": "code",
           "execution_count": null,
           "metadata": {{}},
           "outputs": [],
           "source": [
            "from {slug}.{slug}_agent import {_class_name(slug)}\\n",
            "from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity\\n",
            "\\n",
            "agent = {_class_name(slug)}()\\n",
            "result = await agent.run(\\n",
            "    AgentRunRequest(\\n",
            "        input='hello',\\n",
            "        identity=RequestIdentity(tenant_id='t1', user_id='u1'),\\n",
            "        agent_id=agent.contract_id,\\n",
            "    )\\n",
            ")\\n",
            "result.output"
           ]
          }}
         ],
         "metadata": {{"kernelspec": {{"display_name": "Python 3", "language": "python", "name": "python3"}}}},
         "nbformat": 4,
         "nbformat_minor": 5
        }}
        """
    )


def _readme(
    slug: str, class_name: str, capabilities: list[str], *, pattern: str
) -> str:
    caps = ", ".join(f"`{c}`" for c in capabilities)
    return dedent(
        f"""\
        # {slug} agent

        Typed **{pattern}** cognitive agent — standalone smoke tests under ``agents/{slug}/tests/``.

        **Architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · **Plan:** [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) · **ADRs:** [`docs/project/technical/adr/README.md`](docs/project/technical/adr/README.md)

        Full process: [`docs/project/technical/guides/AGENT_CREATION_GUIDE.md`](../../docs/project/technical/guides/AGENT_CREATION_GUIDE.md)

        ## Standalone verification

        From repository root:

        ```bash
        uv run pytest agents/{slug}/tests -q
        ```

        ``Agent.run(AgentRunRequest(...))`` keeps tests offline — no Tier-3 host and no
        execution-runtime construction in agent code.

        ## Unit-test authoring (isolated)

        ```python
        from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
        from {slug}.{slug}_agent import {class_name}

        agent = {class_name}()
        result = await agent.run(
            AgentRunRequest(
                input="hello",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=agent.contract_id,
            )
        )
        ```

        ## Lab / product integration

        Add ``AgentBinding.mount({class_name}, ...)`` to the Tier-3 manifest and run through
        **Agent Distribution → registry projection → Execution**. Do not use mutable
        registry construction or direct ``NexusLoop`` on serving paths.

        See **Step 4** in ``docs/project/technical/guides/AGENT_CREATION_GUIDE.md``.

        ## Capabilities

        {caps}

        ## Layout

        - ``{slug}_agent.py`` — Agent class (ACP hooks)
        - ``contract.py`` / ``capabilities.py`` — AgentContract
        - ``steps/`` — domain execution
        - ``prompts/`` — prompt assets
        - ``schemas/`` — I/O models
        - ``tracing/`` — DiagnosticPayload extensions
        - ``signals/`` — domain signal payloads
        - ``tests/`` — standalone agent smoke tests
        - ``notebooks/`` — interactive experiments
        - ``docs/`` — architecture, plan, ADRs, journal
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
    reference: bool = False,
) -> Path:
    """Create typed cognitive-pattern agent (ACP default — no legacy pipeline)."""
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
        _acp_agent_py(
            slug,
            class_name,
            primary_capability,
            pattern=normalized,
            reference=reference,
        ),
        force=force,
    )
    _write(
        target / "contract.py",
        _acp_contract_py(slug, class_name, primary_capability, pattern=normalized),
        force=force,
    )
    _write(
        target / "capabilities.py", _capabilities_py(slug, capabilities), force=force
    )
    _write(target / "steps" / "__init__.py", "", force=force)
    _write(target / "steps" / "domain_job.py", _domain_job_py(slug), force=force)
    _write(target / "tests" / "__init__.py", "", force=force)
    _write(
        target / "tests" / f"test_{slug}_agent.py",
        _acp_test_agent_py(slug, class_name, primary_capability),
        force=force,
    )
    _write(target / "prompts" / "system.md", _prompts_system_md(slug), force=force)
    _write(target / "schemas" / "__init__.py", _schemas_init(), force=force)
    docs_dir = agent_docs_dir(target)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write(
        docs_dir / "ARCHITECTURE.md",
        render_agent_architecture_doc(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=reference,
            pattern=normalized,
        ),
        force=force,
    )
    _write(
        docs_dir / "IMPLEMENTATION_PLAN.md",
        render_agent_implementation_plan(
            slug=slug,
            class_name=class_name,
            capabilities=capabilities,
            reference=reference,
            pattern=normalized,
        ),
        force=force,
    )
    write_agent_journal_scaffold(target, force=force)
    _write(
        target / "README.md",
        _readme(slug, class_name, capabilities, pattern=normalized),
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
    write_agent_signal_scaffold(target=target, slug=slug, force=force)
    from intergrax.applications._shared.application_runtime_graph import (
        agent_distribution_name,
    )
    from intergrax.scaffold.workspace_members import ensure_workspace_member

    packages = [slug, f"{slug}.steps", f"{slug}.schemas"]
    pkg_list = ",\n".join(f'  "{p}"' for p in packages)
    dist = agent_distribution_name(slug)
    _write(
        target / "pyproject.toml",
        dedent(
            f"""\
            # © Artur Czarnecki. All rights reserved.
            # Tier-2 agent dependency project (workspace member).
            # Import path preserved: {slug}
            # Canonical: docs/project/architecture/APPLICATION_RUNTIME_GRAPH_MODEL.md

            [project]
            name = "{dist}"
            version = "0.1.0"
            description = "Reusable Tier-2 agent package: {slug}"
            requires-python = ">=3.12,<3.13"
            dependencies = [
              "Intergrax-ai",
            ]

            [build-system]
            requires = ["setuptools>=68"]
            build-backend = "setuptools.build_meta"

            [tool.setuptools]
            packages = [
            {pkg_list},
            ]

            [tool.setuptools.package-dir]
            "{slug}" = "."

            [tool.uv.sources]
            Intergrax-ai = {{ workspace = true }}
            """
        ),
        force=force,
    )
    ensure_workspace_member(root, f"agents/{slug}")
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
    if uaep:
        raise ValueError(
            "UAEP pipeline scaffold (--uaep) was removed; use default ACP pattern scaffold "
            "(reflex, react, plan_execute, reflection, decomposition)."
        )
    effective_pattern = _normalize_pattern(pattern or "reflex")
    assert effective_pattern is not None
    target = create_acp_pattern_agent(
        name=name,
        capabilities=capabilities,
        root=root,
        pattern=effective_pattern,
        force=force,
        minimal=minimal,
        reference=reference,
    )
    from intergrax.scaffold.workspace_members import ensure_workspace_member

    ensure_workspace_member(root, f"agents/{target.name}")
    return target


def build_parser() -> argparse.ArgumentParser:
    from intergrax.scaffold.cli import build_parser as _build_cli_parser

    return _build_cli_parser()


def main(argv: list[str] | None = None) -> int:
    from intergrax.scaffold.cli import main as _cli_main

    return _cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
