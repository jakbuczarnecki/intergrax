# © Artur Czarnecki. All rights reserved.

"""Universal scenario application architecture conformance (SCENARIO-PLATFORM-6A)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from scripts.proof.create_scenario_proof import CANONICAL_SCENARIOS_ROOT, scenario_package_root
from scripts.proof.create_scenario_proof import ScenarioSlug, validate_scenario_slug
from scripts.proof.scenario_lifecycle import (
    ScenarioImplementationStatus,
    ScenarioLifecycle,
    ScenarioLifecycleMetadata,
    ScenarioLifecycleParseStatus,
    load_scenario_lifecycle_metadata,
)

SCENARIO_SPEC_FILENAME = "SCENARIO_SPEC.md"


class ScenarioArchitectureRuleId(StrEnum):
    APP_IMPORT_FIXTURES = "SCENARIO_ARCH_APP_IMPORT_FIXTURES"
    APP_IMPORT_PROOF = "SCENARIO_ARCH_APP_IMPORT_PROOF"
    FORBIDDEN_EXECUTION_SYMBOL = "SCENARIO_ARCH_FORBIDDEN_EXECUTION"
    FORBIDDEN_DIAGNOSTIC_ENGINE = "SCENARIO_ARCH_FORBIDDEN_DIAGNOSTIC"
    PRIVATE_NEXUS_ATTRIBUTE = "SCENARIO_ARCH_PRIVATE_NEXUS"
    BASELINE_RUNTIME_MISSING = "SCENARIO_ARCH_BASELINE_MISSING"
    CONFORMANCE_BYPASS = "SCENARIO_ARCH_CONFORMANCE_BYPASS"


_FORBIDDEN_EXECUTION_SYMBOLS = frozenset(
    {
        "GraphExecutor",
        "NexusLoop",
        "HarnessHostRuntime",
        "mint_run_id",
        "mint_attempt_id",
        "mint_execution_id",
        "bind_active_execution_identity",
        "reset_active_execution_identity",
    }
)

_FORBIDDEN_DIAGNOSTIC_ENGINE_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemLifecycleEngine",
        "ProblemGroupingEngine",
        "ExecutionReconstructor",
    }
)

_FORBIDDEN_PRIVATE_NEXUS_ATTRIBUTES = frozenset(
    {
        "_validation_engine",
        "_graph_executor",
        "_graph_runner",
        "_planner",
        "_graph_spec",
    }
)

_BASELINE_RUNTIME_MODULES = frozenset(
    {
        "intergrax.applications._shared.scenario_runtime_baseline",
        "intergrax.applications._shared.scenario_runtime_profiles",
    }
)

_BASELINE_RUNTIME_SYMBOLS = frozenset(
    {
        "ScenarioRuntimeComposition",
        "build_scenario_lab_runtime",
        "build_scenario_production_runtime",
        "build_scenario_runtime_from_environment",
    }
)

_IMPLEMENTATION_LIFECYCLES = frozenset(
    {
        ScenarioLifecycle.IMPLEMENTATION_INITIALIZED,
        ScenarioLifecycle.EXECUTABLE,
        ScenarioLifecycle.VERIFIED,
    }
)


@dataclass(frozen=True, slots=True)
class ScenarioArchitectureViolation:
    rule_id: ScenarioArchitectureRuleId
    scenario_slug: str
    relative_path: str
    line: int
    symbol: str
    message: str

    def format(self) -> str:
        return (
            f"{self.rule_id.value}\n"
            f"{self.relative_path}:{self.line}\n"
            f"{self.message}"
        )


@dataclass(frozen=True, slots=True)
class ScenarioArchitectureConformanceReport:
    scenario_slug: str
    violations: tuple[ScenarioArchitectureViolation, ...]
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def ok(self) -> bool:
        return not self.violations


class ScenarioArchitectureConformanceError(RuntimeError):
    """Scenario application architecture conformance failed."""

    def __init__(self, violations: tuple[ScenarioArchitectureViolation, ...]) -> None:
        self.violations = violations
        rendered = "\n\n".join(violation.format() for violation in violations)
        super().__init__(rendered or "scenario architecture conformance failed")


def scenario_requires_application_architecture_validation(
    metadata: ScenarioLifecycleMetadata,
    *,
    package_root: Path,
) -> bool:
    """Return True when initialized implementation warrants universal architecture checks."""
    if metadata.parse_status is not ScenarioLifecycleParseStatus.PARSED:
        return False
    application_dir = package_root / "application"
    if not application_dir.is_dir():
        return False
    if metadata.implementation_status is ScenarioImplementationStatus.INITIALIZED:
        return True
    return metadata.lifecycle in _IMPLEMENTATION_LIFECYCLES


def _relative_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _forbidden_scenario_subpackages(slug: str) -> frozenset[str]:
    base = f"platform_proofs.scenarios.{slug}"
    return frozenset({f"{base}.fixtures", f"{base}.proof"})


def _scenario_application_package_parts(
    *,
    package_root: Path,
    file_path: Path,
    slug: str,
) -> tuple[str, ...] | None:
    try:
        rel = file_path.relative_to(package_root)
    except ValueError:
        return None
    if not rel.parts or rel.parts[0] != "application":
        return None
    return ("platform_proofs", "scenarios", slug, *rel.parts[:-1])


def _resolve_import_module(
    *,
    package_root: Path,
    file_path: Path,
    slug: str,
    module: str | None,
    level: int,
) -> str | None:
    if level <= 0:
        return module
    package_parts = _scenario_application_package_parts(
        package_root=package_root,
        file_path=file_path,
        slug=slug,
    )
    if package_parts is None or len(package_parts) < level:
        return None
    base_parts = package_parts[: len(package_parts) - (level - 1)]
    if module is None:
        return ".".join(base_parts)
    return ".".join((*base_parts, *module.split(".")))


def _import_targets_forbidden_layer(
    *,
    file_path: Path,
    package_root: Path,
    slug: str,
    module: str | None,
    level: int,
) -> tuple[ScenarioArchitectureRuleId | None, str]:
    resolved_module = _resolve_import_module(
        package_root=package_root,
        file_path=file_path,
        slug=slug,
        module=module,
        level=level,
    )
    if resolved_module is None:
        return None, module or ""

    forbidden = _forbidden_scenario_subpackages(slug)
    for forbidden_prefix in forbidden:
        if resolved_module == forbidden_prefix or resolved_module.startswith(f"{forbidden_prefix}."):
            if forbidden_prefix.endswith(".fixtures"):
                return ScenarioArchitectureRuleId.APP_IMPORT_FIXTURES, resolved_module
            return ScenarioArchitectureRuleId.APP_IMPORT_PROOF, resolved_module
    return None, resolved_module


def _violation(
    *,
    rule_id: ScenarioArchitectureRuleId,
    scenario_slug: str,
    relative_path: str,
    line: int,
    symbol: str,
    message: str,
) -> ScenarioArchitectureViolation:
    return ScenarioArchitectureViolation(
        rule_id=rule_id,
        scenario_slug=scenario_slug,
        relative_path=relative_path,
        line=line,
        symbol=symbol,
        message=message,
    )


def _collect_application_violations(
    *,
    repo_root: Path,
    package_root: Path,
    scenario_slug: str,
) -> list[ScenarioArchitectureViolation]:
    application_dir = package_root / "application"
    violations: list[ScenarioArchitectureViolation] = []
    baseline_reference_found = False

    for path in sorted(application_dir.rglob("*.py")):
        rel = _relative_path(path, repo_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    rule_id, symbol = _import_targets_forbidden_layer(
                        file_path=path,
                        package_root=package_root,
                        slug=scenario_slug,
                        module=alias.name,
                        level=0,
                    )
                    if rule_id is not None:
                        violations.append(
                            _violation(
                                rule_id=rule_id,
                                scenario_slug=scenario_slug,
                                relative_path=rel,
                                line=node.lineno,
                                symbol=symbol,
                                message=f"forbidden import: {symbol}",
                            )
                        )

            if isinstance(node, ast.ImportFrom):
                rule_id, symbol = _import_targets_forbidden_layer(
                    file_path=path,
                    package_root=package_root,
                    slug=scenario_slug,
                    module=node.module,
                    level=node.level,
                )
                if rule_id is not None:
                    violations.append(
                        _violation(
                            rule_id=rule_id,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=symbol,
                            message=f"forbidden import: {symbol}",
                        )
                    )
                if node.module in _BASELINE_RUNTIME_MODULES:
                    baseline_reference_found = True
                for alias in node.names:
                    imported_name = alias.name
                    if imported_name in _BASELINE_RUNTIME_SYMBOLS:
                        baseline_reference_found = True
                    if imported_name in _FORBIDDEN_EXECUTION_SYMBOLS:
                        violations.append(
                            _violation(
                                rule_id=ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
                                scenario_slug=scenario_slug,
                                relative_path=rel,
                                line=node.lineno,
                                symbol=imported_name,
                                message=f"forbidden execution symbol import: {imported_name}",
                            )
                        )
                    if imported_name in _FORBIDDEN_DIAGNOSTIC_ENGINE_SYMBOLS:
                        violations.append(
                            _violation(
                                rule_id=ScenarioArchitectureRuleId.FORBIDDEN_DIAGNOSTIC_ENGINE,
                                scenario_slug=scenario_slug,
                                relative_path=rel,
                                line=node.lineno,
                                symbol=imported_name,
                                message=f"forbidden diagnostic engine import: {imported_name}",
                            )
                        )

            if isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_EXECUTION_SYMBOLS:
                    violations.append(
                        _violation(
                            rule_id=ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=node.id,
                            message=f"forbidden execution symbol: {node.id}",
                        )
                    )
                if node.id in _FORBIDDEN_DIAGNOSTIC_ENGINE_SYMBOLS:
                    violations.append(
                        _violation(
                            rule_id=ScenarioArchitectureRuleId.FORBIDDEN_DIAGNOSTIC_ENGINE,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=node.id,
                            message=f"forbidden diagnostic engine: {node.id}",
                        )
                    )
                if node.id in _BASELINE_RUNTIME_SYMBOLS:
                    baseline_reference_found = True

            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_PRIVATE_NEXUS_ATTRIBUTES:
                    violations.append(
                        _violation(
                            rule_id=ScenarioArchitectureRuleId.PRIVATE_NEXUS_ATTRIBUTE,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=node.attr,
                            message=f"forbidden private attribute: {node.attr}",
                        )
                    )
                if node.attr in _FORBIDDEN_EXECUTION_SYMBOLS:
                    violations.append(
                        _violation(
                            rule_id=ScenarioArchitectureRuleId.FORBIDDEN_EXECUTION_SYMBOL,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=node.attr,
                            message=f"forbidden execution symbol: {node.attr}",
                        )
                    )
                if node.attr in _FORBIDDEN_DIAGNOSTIC_ENGINE_SYMBOLS:
                    violations.append(
                        _violation(
                            rule_id=ScenarioArchitectureRuleId.FORBIDDEN_DIAGNOSTIC_ENGINE,
                            scenario_slug=scenario_slug,
                            relative_path=rel,
                            line=node.lineno,
                            symbol=node.attr,
                            message=f"forbidden diagnostic engine: {node.attr}",
                        )
                    )

            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg != "conformance_check":
                        continue
                    if isinstance(keyword.value, ast.Constant) and keyword.value.value is False:
                        violations.append(
                            _violation(
                                rule_id=ScenarioArchitectureRuleId.CONFORMANCE_BYPASS,
                                scenario_slug=scenario_slug,
                                relative_path=rel,
                                line=node.lineno,
                                symbol="conformance_check",
                                message="forbidden conformance bypass: conformance_check=False",
                            )
                        )

    if not baseline_reference_found:
        violations.append(
            _violation(
                rule_id=ScenarioArchitectureRuleId.BASELINE_RUNTIME_MISSING,
                scenario_slug=scenario_slug,
                relative_path=_relative_path(application_dir, repo_root),
                line=0,
                symbol="scenario_runtime_baseline",
                message=(
                    "application must reference shared scenario runtime baseline "
                    "(ScenarioRuntimeComposition or build_scenario_* runtime helpers)"
                ),
            )
        )

    return violations


def validate_scenario_application_architecture(
    *,
    repo_root: Path,
    scenario_slug: str,
    package_root: Path | None = None,
    metadata: ScenarioLifecycleMetadata | None = None,
    skip_lifecycle_check: bool = False,
) -> ScenarioArchitectureConformanceReport:
    """Validate universal architecture rules for a scenario application layer."""
    slug = validate_scenario_slug(scenario_slug).value
    resolved_package_root = package_root or scenario_package_root(
        repo_root,
        ScenarioSlug(slug),
    )
    resolved_metadata = metadata
    if resolved_metadata is None:
        spec_path = resolved_package_root / SCENARIO_SPEC_FILENAME
        if spec_path.is_file():
            resolved_metadata = load_scenario_lifecycle_metadata(
                spec_path,
                expected_slug=slug,
            )

    if not skip_lifecycle_check:
        if resolved_metadata is None:
            return ScenarioArchitectureConformanceReport(
                scenario_slug=slug,
                violations=(),
                skipped=True,
                skip_reason="missing lifecycle metadata",
            )
        if not scenario_requires_application_architecture_validation(
            resolved_metadata,
            package_root=resolved_package_root,
        ):
            return ScenarioArchitectureConformanceReport(
                scenario_slug=slug,
                violations=(),
                skipped=True,
                skip_reason="lifecycle does not require architecture validation",
            )

    application_dir = resolved_package_root / "application"
    if not application_dir.is_dir():
        return ScenarioArchitectureConformanceReport(
            scenario_slug=slug,
            violations=(),
            skipped=True,
            skip_reason="application layer not initialized",
        )

    violations = tuple(
        _collect_application_violations(
            repo_root=repo_root,
            package_root=resolved_package_root,
            scenario_slug=slug,
        )
    )
    return ScenarioArchitectureConformanceReport(
        scenario_slug=slug,
        violations=violations,
    )


def assert_scenario_application_architecture(
    *,
    repo_root: Path,
    scenario_slug: str,
    package_root: Path | None = None,
    metadata: ScenarioLifecycleMetadata | None = None,
    skip_lifecycle_check: bool = False,
) -> ScenarioArchitectureConformanceReport:
    """Assert universal scenario application architecture conformance."""
    report = validate_scenario_application_architecture(
        repo_root=repo_root,
        scenario_slug=scenario_slug,
        package_root=package_root,
        metadata=metadata,
        skip_lifecycle_check=skip_lifecycle_check,
    )
    if not report.ok:
        raise ScenarioArchitectureConformanceError(report.violations)
    return report


def discover_initialized_scenario_slugs(repo_root: Path) -> tuple[str, ...]:
    """Return slugs under platform_proofs/scenarios with an application layer."""
    scenarios_root = repo_root / CANONICAL_SCENARIOS_ROOT
    if not scenarios_root.is_dir():
        return ()
    slugs: list[str] = []
    for package_root in sorted(scenarios_root.iterdir()):
        if not package_root.is_dir():
            continue
        if (package_root / "application").is_dir():
            slugs.append(package_root.name)
    return tuple(slugs)
