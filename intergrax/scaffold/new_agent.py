# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent scaffold CLI — create new capability modules under ``agents/``."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from textwrap import dedent


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug or slug[0].isdigit():
        raise ValueError(f"Invalid agent name: {name!r}")
    return slug


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_")) + "Agent"


def _agent_py(slug: str, class_name: str, capabilities: list[str]) -> str:
    caps_repr = ", ".join(repr(c) for c in capabilities)
    return dedent(
        f'''\
        # © Artur Czarnecki. All rights reserved.
        # Intergrax framework – proprietary and confidential.

        from __future__ import annotations

        from intergrax.agents.agent_contract import Agent
        from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
        from intergrax.contracts.capability import CapabilityMatchResult
        from intergrax.runtime.nexus.config import RuntimeConfig
        from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
        from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
        from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
        from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest
        from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
        from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
        from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS
        from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
        from intergrax.runtime.nexus.session.session_manager import SessionManager


        class _{class_name}LLMStub:
            def generate(self, messages, **kwargs) -> str:
                for msg in reversed(messages):
                    content = getattr(msg, "content", None) or ""
                    if content:
                        return f"{slug}: {{content}}"
                return "{slug}: (empty)"


        class {class_name}Pipeline(RuntimePipeline):
            async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
                await RuntimeStepRunner.execute_pipeline(
                    [*SETUP_STEPS, PersistAndBuildAnswerStep()],
                    state,
                )
                message = (state.request.message or "").strip()
                if state.runtime_answer is not None:
                    state.runtime_answer.answer = f"{slug}: {{message}}"
                if state.runtime_answer is None:
                    raise RuntimeError("{class_name}Pipeline did not produce runtime_answer.")
                return state.runtime_answer


        class {class_name}(Agent):
            """Scaffolded agent — replace pipeline and domain logic as needed."""

            def get_contract(self) -> AgentContract:
                return AgentContract(
                    id="{slug}",
                    name="{class_name}",
                    description="Scaffolded agent for Intergrax experiments.",
                    version="0.1.0",
                    capabilities=[{caps_repr}],
                    allowed_tools=[],
                    risk_level=AgentRiskLevel.LOW,
                    max_steps=10,
                )

            def can_handle(self, task_context: object) -> CapabilityMatchResult:
                capability = getattr(task_context, "capability", None)
                supported = set(self.get_contract().capabilities)
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
                config = RuntimeConfig(
                    llm_adapter=_{class_name}LLMStub(),  # type: ignore[arg-type]
                    enable_rag=False,
                    production_mode=False,
                    tenant_id=request.tenant_id,
                )
                config.pipeline = {class_name}Pipeline()
                session_manager = SessionManager(storage=InMemorySessionStorage())
                return RuntimeContext.build(
                    config=config,
                    session_manager=session_manager,
                )
        '''
    )


def _readme(slug: str, capabilities: list[str]) -> str:
    caps = ", ".join(f"`{c}`" for c in capabilities)
    return dedent(
        f"""\
        # {slug} agent

        Scaffolded capability module. Register in Nexus:

        ```python
        from intergrax.runtime.registry import AgentRegistry
        from {slug}.{slug}_agent import {_class_name(slug)}

        registry = AgentRegistry()
        registry.register({_class_name(slug)}())
        ```

        Capabilities: {caps}

        See `docs/experiment_guide.md` for the experiment workflow.
        """
    )


def create_agent(
    *,
    name: str,
    capabilities: list[str],
    root: Path,
    force: bool = False,
) -> Path:
    slug = _slug(name)
    class_name = _class_name(slug)
    if not capabilities:
        capabilities = [f"{slug}.basic"]

    target = root / "agents" / slug
    if target.exists() and not force:
        raise FileExistsError(f"Agent directory already exists: {target}")

    target.mkdir(parents=True, exist_ok=True)

    agent_file = target / f"{slug}_agent.py"
    agent_file.write_text(_agent_py(slug, class_name, capabilities), encoding="utf-8")

    init_file = target / "__init__.py"
    init_file.write_text(
        dedent(
            f'''\
            from {slug}.{slug}_agent import {class_name}

            __all__ = ["{class_name}"]
            '''
        ),
        encoding="utf-8",
    )

    readme = target / "README.md"
    readme.write_text(_readme(slug, capabilities), encoding="utf-8")

    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax.scaffold",
        description="Scaffold new Intergrax agent capability modules.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    new_agent = sub.add_parser("new-agent", help="Create agents/<name>/ from template")
    new_agent.add_argument("name", help="Agent slug (e.g. research)")
    new_agent.add_argument(
        "--capabilities",
        nargs="+",
        default=[],
        help="Capability ids (default: <name>.basic)",
    )
    new_agent.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    new_agent.add_argument("--force", action="store_true", help="Overwrite if exists")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "new-agent":
        try:
            path = create_agent(
                name=args.name,
                capabilities=args.capabilities,
                root=args.root.resolve(),
                force=args.force,
            )
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Created agent scaffold at {path}")
        print(f"  python -m intergrax.scaffold new-agent {args.name}  # already done")
        print(f"  Register: from {path.name}.{path.name}_agent import {_class_name(_slug(args.name))}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
