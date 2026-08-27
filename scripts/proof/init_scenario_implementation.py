# © Artur Czarnecki. All rights reserved.

"""Initialize an implementation skeleton for an accepted Scenario Proof design package."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.proof.create_scenario_proof import (
    CANONICAL_SCENARIOS_ROOT,
    DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES,
    ScenarioSlug,
    ScenarioSlugError,
    resolve_repo_root,
    scenario_package_root,
    validate_scenario_slug,
)
from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
)
from scripts.proof.scenario_lifecycle import (
    ScenarioLifecycleGateError,
    ScenarioLifecycleMetadata,
    load_scenario_lifecycle_metadata,
    validate_implementation_init_preconditions,
    write_scenario_spec_frontmatter,
)

SCENARIO_SPEC_FILENAME = "SCENARIO_SPEC.md"
RUN_PROOF_FILENAME = "run_proof.py"
ENV_EXAMPLE_FILENAME = ".env.example"

IMPLEMENTATION_RELATIVE_PATHS: tuple[str, ...] = (
    "application",
    "proof",
    "fixtures",
    RUN_PROOF_FILENAME,
    PROOF_DESCRIPTOR_FILENAME,
    ENV_EXAMPLE_FILENAME,
)


class ScenarioImplementationExistsError(FileExistsError):
    """Implementation artifacts already exist for the scenario package."""


class ScenarioImplementationInitError(RuntimeError):
    """Implementation skeleton generation failed."""


@dataclass(frozen=True, slots=True)
class ScenarioImplementationRequest:
    slug: ScenarioSlug
    repo_root: Path


@dataclass(frozen=True, slots=True)
class ScenarioImplementationPackage:
    package_root: Path
    created_paths: tuple[Path, ...]


def _slug_to_class_prefix(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_"))


def _slug_to_proof_id(slug: str) -> str:
    return f"SCENARIO-{slug.replace('_', '-').upper()}"


def _extract_scenario_title(spec_body: str, *, fallback: str) -> str:
    match = re.search(r"^\*\*Scenario:\*\*\s+(.+?)\s{2,}$", spec_body, flags=re.MULTILINE)
    if match is None:
        return fallback
    title = match.group(1).strip()
    return title or fallback


def _implementation_artifact_paths(package_root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for relative in IMPLEMENTATION_RELATIVE_PATHS:
        paths.append(package_root / relative)
    for forbidden in DESIGN_STAGE_FORBIDDEN_ARTIFACT_NAMES:
        candidate = package_root / forbidden
        if candidate.name not in IMPLEMENTATION_RELATIVE_PATHS:
            paths.append(candidate)
    return tuple(paths)


def _assert_no_implementation_artifacts(package_root: Path) -> None:
    existing = [
        path.relative_to(package_root).as_posix()
        for path in _implementation_artifact_paths(package_root)
        if path.exists()
    ]
    if existing:
        raise ScenarioImplementationExistsError(
            "implementation artifacts already exist: " + ", ".join(sorted(existing))
        )


def _module_path(slug: str, suffix: str) -> str:
    return f"platform_proofs.scenarios.{slug}.{suffix}"


def _build_runtime_composition_py(slug: str, agent_class: str) -> str:
    module = _module_path(slug, "application.runtime_composition")
    agent_module = _module_path(slug, "application.agent")
    return (
        '"""Scenario runtime composition via platform scenario runtime baseline."""\n\n'
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "from intergrax.applications._shared.scenario_runtime_baseline import (\n"
        "    ScenarioRuntimeComposition,\n"
        "    build_scenario_runtime_from_environment,\n"
        ")\n"
        "from intergrax.applications.contracts.environment_profile import (\n"
        "    ApplicationEnvironmentProfile,\n"
        ")\n"
        "from intergrax.runtime.registry.agent_registry import AgentRegistry\n\n"
        f"from {agent_module} import {agent_class}\n\n"
        f"SYNTHETIC_SCENARIO_TENANT_ID = \"synthetic-scenario-{slug}\"\n\n\n"
        "def build_scenario_environment() -> ApplicationEnvironmentProfile:\n"
        "    return ApplicationEnvironmentProfile.lab_defaults(\n"
        f"        profile_id=\"{slug}.lab\",\n"
        "    )\n\n\n"
        "def build_scenario_runtime(\n"
        "    *,\n"
        "    tenant_id: str,\n"
        "    runtime_events_db_path: Path | None = None,\n"
        "    trace_db_path: Path | None = None,\n"
        ") -> ScenarioRuntimeComposition:\n"
        "    registry = AgentRegistry()\n"
        f"    registry.register({agent_class}())\n"
        "    return build_scenario_runtime_from_environment(\n"
        "        environment=build_scenario_environment(),\n"
        "        registry=registry,\n"
        "        tenant_id=tenant_id,\n"
        "        runtime_events_db_path=runtime_events_db_path,\n"
        "        trace_db_path=trace_db_path,\n"
        "    )\n"
    )


def _build_scenario_py(slug: str) -> str:
    runtime_module = _module_path(slug, "application.runtime_composition")
    return (
        '"""Scenario application execution entry."""\n\n'
        "from __future__ import annotations\n\n"
        "from intergrax.applications._shared.scenario_runtime_baseline import (\n"
        "    ScenarioExecutionRequest,\n"
        "    ScenarioRuntimeComposition,\n"
        "    ScenarioRuntimeExecutionResult,\n"
        "    execute_scenario_task,\n"
        ")\n\n"
        f"from {runtime_module} import build_scenario_runtime\n\n\n"
        "async def execute_scenario(\n"
        "    *,\n"
        "    tenant_id: str,\n"
        "    message: str,\n"
        "    composition: ScenarioRuntimeComposition | None = None,\n"
        ") -> ScenarioRuntimeExecutionResult:\n"
        '    """Execute one scenario task through the platform scenario runtime facade."""\n'
        "    runtime = composition or build_scenario_runtime(tenant_id=tenant_id)\n"
        "    return await execute_scenario_task(\n"
        "        runtime,\n"
        "        ScenarioExecutionRequest(tenant_id=tenant_id, message=message),\n"
        "    )\n"
    )


def _build_agent_py(slug: str, agent_class: str) -> str:
    capability = f"{slug}.run"
    return (
        '"""Minimal scenario agent skeleton — implement domain behavior."""\n\n'
        "from __future__ import annotations\n\n"
        "from intergrax.agents.authoring.patterns.reflex import ReflexAgent\n"
        "from intergrax.agents.authoring.patterns.types import (\n"
        "    AgentEvaluation,\n"
        "    CognitiveEvaluation,\n"
        "    Observation,\n"
        "    ReasoningResult,\n"
        ")\n"
        "from intergrax.contracts.agent_step_context import AgentStepContext\n"
        "from intergrax.contracts.capability import CapabilityMatchResult\n"
        "from intergrax.runtime.task.task import TaskContext\n\n\n"
        f"class {agent_class}(ReflexAgent):\n"
        '    """TODO: implement scenario agent contract."""\n\n'
        f'    contract_id = "{slug}"\n'
        f'    capabilities = ("{capability}",)\n\n'
        "    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:\n"
        f'        if task_context.capability in (None, "{capability}"):\n'
        "            return CapabilityMatchResult(\n"
        "                matched=True,\n"
        f'                agent_id="{slug}",\n'
        f'                matched_capabilities=["{capability}"],\n'
        "                score=1.0,\n"
        '                rationale="scenario skeleton agent",\n'
        "            )\n"
        '        return CapabilityMatchResult(matched=False, rationale="capability not supported")\n\n'
        "    async def perceive(self, step_ctx: AgentStepContext) -> Observation:\n"
        '        raise NotImplementedError("Implement scenario perception.")\n\n'
        "    async def reason(\n"
        "        self,\n"
        "        step_ctx: AgentStepContext,\n"
        "        observation: Observation,\n"
        "    ) -> ReasoningResult:\n"
        '        raise NotImplementedError("Implement scenario reasoning.")\n\n'
        "    async def act(\n"
        "        self,\n"
        "        step_ctx: AgentStepContext,\n"
        "        reasoning: ReasoningResult,\n"
        "    ) -> dict[str, object]:\n"
        '        raise NotImplementedError("Implement scenario action.")\n\n'
        "    def evaluate(\n"
        "        self,\n"
        "        step_ctx: AgentStepContext,\n"
        "        output: dict[str, object],\n"
        "    ) -> AgentEvaluation:\n"
        '        raise NotImplementedError("Implement scenario evaluation.")\n'
    )


def _build_observability_py() -> str:
    return (
        '"""Scenario application observability contract.\n\n'
        "Domain-specific DiagnosticPayload implementations belong here.\n"
        "Proof/report layers consume canonical runtime trace — they do not invent "
        "execution explanations.\n"
        '"""\n\n'
        "from __future__ import annotations\n"
    )


def _build_tools_py() -> str:
    return (
        '"""Scenario tool declarations.\n\n'
        "Declare ToolRegistry / ToolProfile bindings here when the scenario contract "
        "requires tools. Do not add fake business tool logic in the proof skeleton.\n"
        '"""\n\n'
        "from __future__ import annotations\n"
    )


def _build_evaluator_py() -> str:
    return (
        '"""Proof-owned evaluation seam — falsification assertions live here."""\n\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True, slots=True)\n"
        "class ScenarioEvaluation:\n"
        "    passed: bool\n"
        "    failures: tuple[str, ...] = ()\n\n\n"
        "def evaluate_scenario_run(domain_result: object) -> ScenarioEvaluation:\n"
        '    """Evaluate application output against the Scenario Specification contract."""\n'
        '    raise NotImplementedError("Implement proof evaluator contract.")\n'
    )


def _build_evidence_builder_py() -> str:
    return (
        '"""Proof-owned evidence projection — consumes application/runtime artifacts."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def build_platform_proof_evidence(\n"
        "    domain_result: object,\n"
        "    *,\n"
        "    source_revision: str,\n"
        ") -> object:\n"
        '    """Project runtime evidence into PlatformProofEvidence v3."""\n'
        '    raise NotImplementedError("Implement evidence projection.")\n'
    )


def _build_run_proof_py(slug: str) -> str:
    scenario_module = _module_path(slug, "application.scenario")
    runtime_module = _module_path(slug, "application.runtime_composition")
    evaluator_module = _module_path(slug, "proof.evaluator")
    return (
        '"""Thin scenario proof runner — configure, invoke application, evaluate, write artifacts."""\n\n'
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import asyncio\n"
        "import sys\n\n"
        f"from {runtime_module} import SYNTHETIC_SCENARIO_TENANT_ID\n"
        f"from {scenario_module} import execute_scenario\n"
        f"from {evaluator_module} import evaluate_scenario_run\n\n\n"
        "async def _run() -> int:\n"
        "    # TODO: wire configuration, invoke application, collect evidence, evaluate.\n"
        "    # Critic/HITL/RAG/web/memory/hosting are opt-in via ApplicationEnvironmentProfile.\n"
        '    raise NotImplementedError("Implement proof runner workflow.")\n\n\n'
        "def main(argv: list[str] | None = None) -> int:\n"
        '    parser = argparse.ArgumentParser(description="Scenario proof runner.")\n'
        '    parser.add_argument("--validate-only", action="store_true")\n'
        "    _ = parser.parse_args(argv)\n"
        "    try:\n"
        "        return asyncio.run(_run())\n"
        "    except NotImplementedError:\n"
        '        print("Proof runner not yet implemented.", file=sys.stderr)\n'
        "        return 2\n\n\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n"
    )


def _build_proof_json(slug: str, title: str) -> str:
    payload = {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": "SCENARIO",
        "proof_id": _slug_to_proof_id(slug),
        "title": title,
        "domains_exercised": ["EXECUTION"],
        "proof_kind": "scenario_skeleton",
        "mechanisms_exercised": ["runtime.scenario_baseline"],
        "package_version": "0.0.0-skeleton",
        "profiles": ["quick"],
        "command": {
            "executable": "uv",
            "argv": [
                "run",
                "python",
                f"platform_proofs/scenarios/{slug}/{RUN_PROOF_FILENAME}",
            ],
        },
        "timeout_seconds": 180,
        "safety_class": "LOCAL_READ_ONLY",
        "public_evidence_eligible": False,
        "problem_category": "scenario_skeleton",
        "problem_summary": f"Skeleton placeholder for {title} — replace before publication.",
        "failure_mode_summary": "Skeleton placeholder — define falsification boundary in Scenario Specification.",
        "evidence_required": False,
        "tags": ["scenario_skeleton", "platform_native"],
    }
    return json.dumps(payload, indent=2) + "\n"


def _build_env_example() -> str:
    return (
        "# Scenario proof configuration — provider-neutral placeholders.\n"
        "# Copy to .env and set values before running the proof.\n\n"
        "# INTERGRAX_LLM_PROVIDER=\n"
        "# INTERGRAX_LLM_MODEL=\n"
    )


def _planned_files(
    package_root: Path,
    *,
    slug: str,
    title: str,
) -> dict[Path, str]:
    agent_class = f"{_slug_to_class_prefix(slug)}Agent"
    return {
        package_root / "application" / "__init__.py": "",
        package_root / "application" / "runtime_composition.py": _build_runtime_composition_py(
            slug,
            agent_class,
        ),
        package_root / "application" / "scenario.py": _build_scenario_py(slug),
        package_root / "application" / "agent.py": _build_agent_py(slug, agent_class),
        package_root / "application" / "observability.py": _build_observability_py(),
        package_root / "application" / "tools.py": _build_tools_py(),
        package_root / "proof" / "__init__.py": "",
        package_root / "proof" / "evaluator.py": _build_evaluator_py(),
        package_root / "proof" / "evidence_builder.py": _build_evidence_builder_py(),
        package_root / "fixtures" / "__init__.py": "",
        package_root / RUN_PROOF_FILENAME: _build_run_proof_py(slug),
        package_root / PROOF_DESCRIPTOR_FILENAME: _build_proof_json(slug, title),
        package_root / ENV_EXAMPLE_FILENAME: _build_env_example(),
    }


def init_scenario_implementation(
    request: ScenarioImplementationRequest,
) -> ScenarioImplementationPackage:
    package_root = scenario_package_root(request.repo_root, request.slug)
    if not package_root.is_dir():
        raise ScenarioImplementationInitError(
            f"scenario design package does not exist: {package_root}"
        )

    scenario_spec_path = package_root / SCENARIO_SPEC_FILENAME
    if not scenario_spec_path.is_file():
        raise ScenarioImplementationInitError(
            f"missing {SCENARIO_SPEC_FILENAME} in {package_root}"
        )

    _assert_no_implementation_artifacts(package_root)

    metadata = load_scenario_lifecycle_metadata(
        scenario_spec_path,
        expected_slug=request.slug.value,
    )
    validate_implementation_init_preconditions(metadata)

    from scripts.proof.scenario_lifecycle import split_scenario_spec

    _frontmatter, body = split_scenario_spec(scenario_spec_path.read_text(encoding="utf-8"))
    title = _extract_scenario_title(body, fallback=request.slug.value.replace("_", " ").title())

    planned = _planned_files(package_root, slug=request.slug.value, title=title)
    created: list[Path] = []
    try:
        for path, content in planned.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            created.append(path)

        updated_metadata = metadata.with_implementation_initialized()
        write_scenario_spec_frontmatter(scenario_spec_path, updated_metadata)
    except OSError as exc:
        for path in reversed(created):
            if path.exists():
                path.unlink()
        raise ScenarioImplementationInitError(
            f"failed to write implementation skeleton: {exc}"
        ) from exc

    return ScenarioImplementationPackage(
        package_root=package_root,
        created_paths=tuple(created),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize implementation skeleton for an accepted Scenario Proof.",
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Filesystem-safe scenario slug (lowercase, underscores).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (defaults to parent of scripts/proof/).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    try:
        slug = validate_scenario_slug(args.slug)
        package = init_scenario_implementation(
            ScenarioImplementationRequest(slug=slug, repo_root=repo_root),
        )
    except ScenarioSlugError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ScenarioLifecycleGateError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ScenarioImplementationExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ScenarioImplementationInitError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(package.package_root.relative_to(repo_root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
