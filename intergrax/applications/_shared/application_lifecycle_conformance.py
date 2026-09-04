# © Artur Czarnecki. All rights reserved.

"""Application-owned runtime lifecycle conformance (Tier-3 host/serving boundary)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from intergrax.runtime.architecture.agent_lifecycle_bypass_ast import (
    CANONICAL_AGENT_REGISTRY_MODULE,
    CANONICAL_AGENT_REGISTRY_SYMBOL,
    AgentRegistryImportBindings,
    LifecycleAstViolation,
    collect_agent_registry_import_bindings,
    collect_agent_registry_lifecycle_violations,
)

CANONICAL_BUILD_APPLICATION_REGISTRY_MODULE = "intergrax.applications._shared.wiring"
CANONICAL_BUILD_APPLICATION_REGISTRY_SYMBOL = "build_application_registry"

_APPLICATION_RUNTIME_SEGMENTS = ("host", "serving")


class ApplicationLifecycleRuleId(StrEnum):
    AGENT_LIFECYCLE_BYPASS = "APPLICATION_ARCH_AGENT_LIFECYCLE_BYPASS"
    BUILD_APPLICATION_REGISTRY_BYPASS = "APPLICATION_ARCH_BUILD_APPLICATION_REGISTRY_BYPASS"
    MUTABLE_REGISTRY_TYPE_EXPOSURE = "APPLICATION_ARCH_MUTABLE_REGISTRY_TYPE_EXPOSURE"


@dataclass(frozen=True, slots=True)
class BuildApplicationRegistryImportBindings:
    function_aliases: frozenset[str]
    module_aliases: frozenset[str]
    qualified_module_roots: frozenset[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class ApplicationLifecycleViolation:
    rule_id: ApplicationLifecycleRuleId
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
class ApplicationLifecycleConformanceReport:
    violations: tuple[ApplicationLifecycleViolation, ...]
    scanned_paths: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


class ApplicationLifecycleConformanceError(RuntimeError):
    """Application lifecycle conformance failed."""

    def __init__(self, violations: tuple[ApplicationLifecycleViolation, ...]) -> None:
        self.violations = violations
        rendered = "\n\n".join(violation.format() for violation in violations)
        super().__init__(rendered or "application lifecycle conformance failed")


def _relative_path(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _is_application_owned_runtime_path(path: Path, applications_root: Path) -> bool:
    try:
        rel = path.relative_to(applications_root)
    except ValueError:
        return False
    parts = rel.parts
    if len(parts) < 3:
        return False
    return parts[1] in _APPLICATION_RUNTIME_SEGMENTS


def iter_application_owned_runtime_files(
    repo_root: Path,
    *,
    applications_root: Path | None = None,
) -> tuple[Path, ...]:
    root = applications_root or (repo_root / "applications")
    if not root.is_dir():
        return ()
    paths: list[Path] = []
    for segment in _APPLICATION_RUNTIME_SEGMENTS:
        for path in sorted(root.glob(f"*/{segment}/**/*.py")):
            if _is_application_owned_runtime_path(path, root):
                paths.append(path)
    return tuple(paths)


def _collect_build_application_registry_import_bindings(
    tree: ast.AST,
) -> BuildApplicationRegistryImportBindings:
    function_aliases: set[str] = set()
    module_aliases: set[str] = set()
    qualified_module_roots: set[tuple[str, str]] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == CANONICAL_BUILD_APPLICATION_REGISTRY_MODULE
        ):
            for alias in node.names:
                if alias.name == CANONICAL_BUILD_APPLICATION_REGISTRY_SYMBOL:
                    function_aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == CANONICAL_BUILD_APPLICATION_REGISTRY_MODULE:
                    if alias.asname is not None:
                        module_aliases.add(alias.asname)
                    else:
                        qualified_module_roots.add(
                            (alias.name.split(".")[0], alias.name)
                        )

    return BuildApplicationRegistryImportBindings(
        function_aliases=frozenset(function_aliases),
        module_aliases=frozenset(module_aliases),
        qualified_module_roots=frozenset(qualified_module_roots),
    )


def _attribute_chain_parts(expr: ast.expr) -> tuple[str, tuple[str, ...]] | None:
    attrs: list[str] = []
    current = expr
    while isinstance(current, ast.Attribute):
        attrs.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        return current.id, tuple(reversed(attrs))
    return None


def _resolves_to_build_application_registry_module(
    expr: ast.expr,
    bindings: BuildApplicationRegistryImportBindings,
) -> bool:
    parts = _attribute_chain_parts(expr)
    if parts is None:
        return False
    root, attrs = parts
    if root in bindings.module_aliases and not attrs:
        return True
    for imported_root, module_path in bindings.qualified_module_roots:
        if root != imported_root or module_path != CANONICAL_BUILD_APPLICATION_REGISTRY_MODULE:
            continue
        suffix = module_path[len(imported_root) + 1 :]
        if suffix and attrs == tuple(suffix.split(".")):
            return True
    return False


def _is_build_application_registry_call(
    func: ast.expr,
    bindings: BuildApplicationRegistryImportBindings,
) -> bool:
    if isinstance(func, ast.Name) and func.id in bindings.function_aliases:
        return True
    if (
        isinstance(func, ast.Attribute)
        and func.attr == CANONICAL_BUILD_APPLICATION_REGISTRY_SYMBOL
        and _resolves_to_build_application_registry_module(func.value, bindings)
    ):
        return True
    return False


def _annotation_uses_canonical_agent_registry(
    annotation: ast.expr | None,
    bindings: AgentRegistryImportBindings,
) -> bool:
    if annotation is None:
        return False

    if isinstance(annotation, ast.Name) and annotation.id in bindings.class_aliases:
        return True

    if isinstance(annotation, ast.Attribute):
        if annotation.attr != CANONICAL_AGENT_REGISTRY_SYMBOL:
            return False
        return _resolves_to_agent_registry_module_for_annotation(
            annotation.value,
            bindings,
        )

    if isinstance(annotation, ast.Subscript):
        return _annotation_uses_canonical_agent_registry(annotation.slice, bindings)

    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return (
            _annotation_uses_canonical_agent_registry(annotation.left, bindings)
            or _annotation_uses_canonical_agent_registry(annotation.right, bindings)
        )

    return False


def _resolves_to_agent_registry_module_for_annotation(
    expr: ast.expr,
    bindings: AgentRegistryImportBindings,
) -> bool:
    parts = _attribute_chain_parts(expr)
    if parts is None:
        return False
    root, attrs = parts
    if root in bindings.module_aliases and not attrs:
        return True
    for imported_root, module_path in bindings.qualified_module_roots:
        if root != imported_root or module_path != CANONICAL_AGENT_REGISTRY_MODULE:
            continue
        suffix = module_path[len(imported_root) + 1 :]
        if suffix and attrs == tuple(suffix.split(".")):
            return True
    return False


def _collect_build_application_registry_violations(
    *,
    tree: ast.AST,
    bindings: BuildApplicationRegistryImportBindings,
    relative_path: str,
) -> list[ApplicationLifecycleViolation]:
    violations: list[ApplicationLifecycleViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_build_application_registry_call(node.func, bindings):
            continue
        violations.append(
            ApplicationLifecycleViolation(
                rule_id=ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS,
                relative_path=relative_path,
                line=node.lineno,
                symbol=CANONICAL_BUILD_APPLICATION_REGISTRY_SYMBOL,
                message=(
                    "application-owned production code must not call "
                    "build_application_registry; use registry projection/read-only runtime"
                ),
            )
        )
    return violations


def _collect_mutable_registry_type_violations(
    *,
    tree: ast.AST,
    bindings: AgentRegistryImportBindings,
    relative_path: str,
) -> list[ApplicationLifecycleViolation]:
    violations: list[ApplicationLifecycleViolation] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if _annotation_uses_canonical_agent_registry(arg.annotation, bindings):
                    violations.append(
                        ApplicationLifecycleViolation(
                            rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
                            relative_path=relative_path,
                            line=arg.lineno,
                            symbol=CANONICAL_AGENT_REGISTRY_SYMBOL,
                            message=(
                                "application runtime boundary must not expose mutable "
                                "AgentRegistry; use AgentRegistryRead or "
                                "MaterializedRegistryProjection"
                            ),
                        )
                    )

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if _annotation_uses_canonical_agent_registry(node.annotation, bindings):
                violations.append(
                    ApplicationLifecycleViolation(
                        rule_id=ApplicationLifecycleRuleId.MUTABLE_REGISTRY_TYPE_EXPOSURE,
                        relative_path=relative_path,
                        line=node.lineno,
                        symbol=CANONICAL_AGENT_REGISTRY_SYMBOL,
                        message=(
                            "application runtime boundary must not expose mutable "
                            "AgentRegistry; use AgentRegistryRead or "
                            "MaterializedRegistryProjection"
                        ),
                    )
                )

    return violations


def _map_lifecycle_ast_violation(
    *,
    ast_violation: LifecycleAstViolation,
    relative_path: str,
) -> ApplicationLifecycleViolation:
    return ApplicationLifecycleViolation(
        rule_id=ApplicationLifecycleRuleId.AGENT_LIFECYCLE_BYPASS,
        relative_path=relative_path,
        line=ast_violation.line,
        symbol=ast_violation.symbol,
        message=(
            "application-owned production code must not construct or mutate AgentRegistry; "
            "use canonical agent lifecycle"
        ),
    )


def _finalize_violations(
    violations: list[ApplicationLifecycleViolation],
) -> tuple[ApplicationLifecycleViolation, ...]:
    return tuple(
        sorted(
            violations,
            key=lambda item: (item.relative_path, item.rule_id.value, item.line, item.symbol),
        )
    )


def collect_application_lifecycle_violations_for_file(
    *,
    path: Path,
    repo_root: Path,
    source: str | None = None,
) -> tuple[ApplicationLifecycleViolation, ...]:
    relative_path = _relative_path(path, repo_root)
    try:
        tree = ast.parse(source or path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return ()
    agent_bindings = collect_agent_registry_import_bindings(tree)
    builder_bindings = _collect_build_application_registry_import_bindings(tree)

    violations: list[ApplicationLifecycleViolation] = []
    violations.extend(
        _map_lifecycle_ast_violation(
            ast_violation=ast_violation,
            relative_path=relative_path,
        )
        for ast_violation in collect_agent_registry_lifecycle_violations(
            tree,
            bindings=agent_bindings,
        )
    )
    violations.extend(
        _collect_build_application_registry_violations(
            tree=tree,
            bindings=builder_bindings,
            relative_path=relative_path,
        )
    )
    violations.extend(
        _collect_mutable_registry_type_violations(
            tree=tree,
            bindings=agent_bindings,
            relative_path=relative_path,
        )
    )

    return _finalize_violations(violations)


def validate_application_lifecycle_conformance(
    repo_root: Path,
    *,
    applications_root: Path | None = None,
    extra_paths: tuple[Path, ...] = (),
) -> ApplicationLifecycleConformanceReport:
    scanned_paths: list[str] = []
    violations: list[ApplicationLifecycleViolation] = []
    for path in (*iter_application_owned_runtime_files(repo_root, applications_root=applications_root), *extra_paths):
        scanned_paths.append(_relative_path(path, repo_root))
        violations.extend(
            collect_application_lifecycle_violations_for_file(
                path=path,
                repo_root=repo_root,
            )
        )
    return ApplicationLifecycleConformanceReport(
        violations=tuple(
            sorted(
                violations,
                key=lambda item: (item.relative_path, item.rule_id.value, item.line, item.symbol),
            )
        ),
        scanned_paths=tuple(scanned_paths),
    )


def assert_application_lifecycle_conformance(
    repo_root: Path,
    *,
    applications_root: Path | None = None,
    extra_paths: tuple[Path, ...] = (),
) -> ApplicationLifecycleConformanceReport:
    report = validate_application_lifecycle_conformance(
        repo_root,
        applications_root=applications_root,
        extra_paths=extra_paths,
    )
    if not report.ok:
        raise ApplicationLifecycleConformanceError(report.violations)
    return report
