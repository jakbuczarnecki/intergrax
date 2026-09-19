# © Artur Czarnecki. All rights reserved.

"""AST extraction of static import dependencies for public contract modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from testing_support.architecture.public_contract_boundary.discovery import path_to_module_name


@dataclass(frozen=True, slots=True)
class ExtractedImport:
    imported_module: str
    line: int


def is_public_contract_module(module: str) -> bool:
    if not module.startswith("intergrax."):
        return False
    parts = module.split(".")
    if len(parts) >= 2 and parts[1] == "contracts":
        return True
    for index, part in enumerate(parts[1:], start=1):
        if part != "contracts":
            continue
        prefix = parts[1:index]
        if "runtime" in prefix:
            return False
        return True
    return False


def _resolve_relative_module(source_module: str, node: ast.ImportFrom) -> str | None:
    if node.level <= 0:
        return node.module
    package_parts = source_module.split(".")
    if package_parts[-1] and not source_module.endswith(".__init__"):
        package_parts = package_parts[:-1]
    if node.level > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def collect_intergrax_imports(
    tree: ast.AST,
    *,
    source_module: str,
) -> list[ExtractedImport]:
    imports: list[ExtractedImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax."):
                    imports.append(
                        ExtractedImport(imported_module=alias.name, line=node.lineno),
                    )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        resolved = _resolve_relative_module(source_module, node)
        if resolved is None:
            continue
        if resolved.startswith("intergrax."):
            imports.append(ExtractedImport(imported_module=resolved, line=node.lineno))
    return imports


def extract_imports_from_file(
    path: Path,
    *,
    intergrax_root: Path,
) -> list[ExtractedImport]:
    source_module = path_to_module_name(path, intergrax_root=intergrax_root)
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text, filename=str(path))
    return collect_intergrax_imports(tree, source_module=source_module)
