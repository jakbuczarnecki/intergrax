# © Artur Czarnecki. All rights reserved.

"""Generic Tier-3 execution surface discovery (workers, harness CLI entries)."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from intergrax.applications._shared.application_runtime_graph import list_application_projects
from intergrax.applications.contracts.application_host import ApplicationProfile


class DiscoveredExecutionSurfaceKind(StrEnum):
    WORKER_BACKGROUND = "worker_background"
    HARNESS_CLI = "harness_cli"


@dataclass(frozen=True, slots=True)
class DiscoveredExecutionSurface:
    surface_id: str
    kind: DiscoveredExecutionSurfaceKind
    owner_application_id: str | None
    process_role: str
    entry_paths: tuple[Path, ...]


def _read_python_ast(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None


def _references_worker_background_surface(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "WORKER_BACKGROUND":
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "BootstrapSurfaceKind":
            return True
    return False


def _extract_process_role_constant(tree: ast.AST) -> str | None:
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or not target.id.endswith("PROCESS_ROLE"):
                continue
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _normalize_worker_role(role: str) -> str:
    if role.endswith("_worker"):
        return role[: -len("_worker")]
    return role


def _worker_surface_id(application_id: str, process_role: str) -> str:
    return f"worker:{application_id}.{_normalize_worker_role(process_role)}"


def _collect_worker_companion_paths(host_dir: Path, entry_path: Path, tree: ast.AST) -> tuple[Path, ...]:
    paths: set[Path] = {entry_path.resolve()}
    imported_stems: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        module = node.module
        if module.startswith("host.") or ".host." in module:
            imported_stems.add(module.rsplit(".", 1)[-1])
    for stem in imported_stems:
        candidate = host_dir / f"{stem}.py"
        if candidate.is_file():
            paths.add(candidate.resolve())
    for candidate in sorted(host_dir.glob("*.py")):
        if "worker" in candidate.stem.lower():
            paths.add(candidate.resolve())
    return tuple(sorted(paths))


def _iter_application_host_python(application_dir: Path) -> list[Path]:
    host_dir = application_dir / "host"
    if not host_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(host_dir.rglob("*.py")):
        if "tests" in path.parts or path.name.startswith("test_"):
            continue
        paths.append(path)
    return paths


def discover_worker_execution_surfaces(repo_root: Path) -> tuple[DiscoveredExecutionSurface, ...]:
    surfaces: list[DiscoveredExecutionSurface] = []
    apps_root = repo_root / "applications"
    if not apps_root.is_dir():
        return ()
    for application_id in list_application_projects(repo_root):
        application_dir = apps_root / application_id
        host_dir = application_dir / "host"
        if not host_dir.is_dir():
            continue
        for entry_path in _iter_application_host_python(application_dir):
            tree = _read_python_ast(entry_path)
            if tree is None or not _references_worker_background_surface(tree):
                continue
            process_role = _extract_process_role_constant(tree)
            if process_role is None:
                process_role = entry_path.stem
            companion_paths = _collect_worker_companion_paths(host_dir, entry_path, tree)
            surfaces.append(
                DiscoveredExecutionSurface(
                    surface_id=_worker_surface_id(application_id, process_role),
                    kind=DiscoveredExecutionSurfaceKind.WORKER_BACKGROUND,
                    owner_application_id=application_id,
                    process_role=process_role,
                    entry_paths=companion_paths,
                )
            )
    return tuple(surfaces)


def discover_worker_surface_ids(repo_root: Path) -> frozenset[str]:
    return frozenset(surface.surface_id for surface in discover_worker_execution_surfaces(repo_root))


def iter_worker_execution_surface_python_paths(repo_root: Path) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for surface in discover_worker_execution_surfaces(repo_root):
        paths.update(surface.entry_paths)
    return tuple(sorted(paths))


def _references_harness_host_runtime_entry(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "build_harness_host_runtime":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "build_harness_host_runtime":
            return True
    return False


def discover_harness_execution_surfaces(repo_root: Path) -> tuple[DiscoveredExecutionSurface, ...]:
    harness_root = repo_root / "intergrax" / "harness"
    if not harness_root.is_dir():
        return ()
    surfaces: list[DiscoveredExecutionSurface] = []
    for path in sorted(harness_root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = _read_python_ast(path)
        if tree is None or not _references_harness_host_runtime_entry(tree):
            continue
        module = f"intergrax.harness.{path.stem}"
        surfaces.append(
            DiscoveredExecutionSurface(
                surface_id=f"harness:{module}",
                kind=DiscoveredExecutionSurfaceKind.HARNESS_CLI,
                owner_application_id=None,
                process_role="harness_cli",
                entry_paths=(path.resolve(),),
            )
        )
    return tuple(surfaces)


def discover_harness_entry_surface_ids(repo_root: Path) -> frozenset[str]:
    return frozenset(surface.surface_id for surface in discover_harness_execution_surfaces(repo_root))


def application_manifest_profile_from_manifest_py(manifest_path: Path) -> ApplicationProfile | None:
    tree = _read_python_ast(manifest_path)
    if tree is None:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id != "ApplicationManifest":
                continue
            if func.attr == "product":
                return ApplicationProfile.PRODUCT
            if func.attr == "lab":
                return ApplicationProfile.LAB
        if isinstance(func, ast.Name) and func.id == "ApplicationManifest":
            for keyword in node.keywords:
                if keyword.arg != "profile":
                    continue
                value = keyword.value
                if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
                    if value.value.id == "ApplicationProfile":
                        if value.attr == "PRODUCT":
                            return ApplicationProfile.PRODUCT
                        if value.attr == "LAB":
                            return ApplicationProfile.LAB
    return None


def resolve_application_profile(repo_root: Path, application_id: str) -> ApplicationProfile | None:
    manifest_path = repo_root / "applications" / application_id / "manifest.py"
    if not manifest_path.is_file():
        return None
    return application_manifest_profile_from_manifest_py(manifest_path)


__all__ = [
    "DiscoveredExecutionSurface",
    "DiscoveredExecutionSurfaceKind",
    "application_manifest_profile_from_manifest_py",
    "discover_harness_entry_surface_ids",
    "discover_harness_execution_surfaces",
    "discover_worker_execution_surfaces",
    "discover_worker_surface_ids",
    "iter_worker_execution_surface_python_paths",
    "resolve_application_profile",
]
