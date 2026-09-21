# © Artur Czarnecki. All rights reserved.

"""Detect Nexus import-like usage in public authoring surfaces (HARNESS-01-R5-W1)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from tests.qualification.harness_01.nexus_boundary_detector import file_imports_nexus_module

_FENCED_PYTHON_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

# Explicit public authoring scope — not maintainer/ADR/architecture hubs.
PUBLIC_AUTHORING_DOC_RELATIVE_PATHS: tuple[str, ...] = (
    "docs/project/technical/guides/EXTENSION_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/TOOL_INVOCATION_PATTERN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/AGENT_AUTHOR_MINIMAL_PATH.md",
    "docs/project/technical/guides/AGENT_CREATION_GUIDE.md",
    "docs/project/technical/guides/CONTEXT_PLUGIN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/MEMORY_STORE_PLUGIN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/POLICY_RULE_PLUGIN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/SECURITY_DEFENSE_PLUGIN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/VENDOR_KNOWLEDGE_PLUGIN_AUTHOR_GUIDE.md",
    "docs/project/technical/guides/RAG_EXTENSION_GUIDE.md",
)

PUBLIC_AUTHORING_EXAMPLE_PREFIXES: tuple[str, ...] = (
    "examples/platform_plugins/",
)

# Extension scaffolds (tool/skill/integration/context) plus public new-agent golden path.
# Application host generators remain Wave 6 debt and are intentionally excluded here.
PUBLIC_EXTENSION_SCAFFOLD_RELATIVE_PATHS: tuple[str, ...] = (
    "intergrax/scaffold/new_tool_bundle.py",
    "intergrax/scaffold/new_skill.py",
    "intergrax/scaffold/new_integration.py",
    "intergrax/scaffold/new_context_bundle.py",
    "intergrax/scaffold/integration_templates.py",
    "intergrax/scaffold/new_agent.py",
)


@dataclass(frozen=True, slots=True)
class PublicAuthoringNexusHit:
    """One import-like Nexus reference in a public authoring surface."""

    relative_path: str
    kind: str
    detail: str


def extract_fenced_python_blocks(markdown: str) -> list[str]:
    """Return fenced ```python / ```py bodies (best-effort, not a full MD parser)."""
    return [match.group(1) for match in _FENCED_PYTHON_RE.finditer(markdown)]


def python_source_imports_nexus(source: str) -> bool:
    """True when AST shows an import of ``intergrax.runtime.nexus``."""
    return file_imports_nexus_module(source)


def markdown_fenced_python_imports_nexus(markdown: str) -> list[str]:
    """Return fenced Python snippets that AST-import Nexus (empty = clean)."""
    hits: list[str] = []
    for block in extract_fenced_python_blocks(markdown):
        try:
            if python_source_imports_nexus(block):
                hits.append(block.strip().splitlines()[0] if block.strip() else "<empty>")
        except SyntaxError:
            # Incomplete illustrative snippets are ignored unless they parse
            # as a Nexus import module statement via fallback line scan.
            for line in block.splitlines():
                stripped = line.strip()
                if stripped.startswith("from intergrax.runtime.nexus") or stripped.startswith(
                    "import intergrax.runtime.nexus"
                ):
                    hits.append(stripped)
                    break
    return hits


def scan_public_authoring_path(repo_root: Path, relative_path: str) -> list[PublicAuthoringNexusHit]:
    """Scan one relative path under ``repo_root`` for Nexus import-like usage."""
    path = repo_root / relative_path
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8-sig")
    hits: list[PublicAuthoringNexusHit] = []
    suffix = path.suffix.lower()
    if suffix == ".py":
        if python_source_imports_nexus(text):
            hits.append(
                PublicAuthoringNexusHit(
                    relative_path=relative_path.replace("\\", "/"),
                    kind="python_import",
                    detail="AST import of intergrax.runtime.nexus",
                )
            )
        return hits
    if suffix in {".md", ".markdown", ".rst"}:
        for snippet in markdown_fenced_python_imports_nexus(text):
            hits.append(
                PublicAuthoringNexusHit(
                    relative_path=relative_path.replace("\\", "/"),
                    kind="markdown_fenced_python_import",
                    detail=snippet[:200],
                )
            )
    return hits


def iter_public_authoring_relative_paths(repo_root: Path) -> list[str]:
    """Closed-world inventory of Wave-1 public authoring surfaces."""
    paths: list[str] = []
    for rel in PUBLIC_AUTHORING_DOC_RELATIVE_PATHS:
        if (repo_root / rel).is_file():
            paths.append(rel.replace("\\", "/"))
    for prefix in PUBLIC_AUTHORING_EXAMPLE_PREFIXES:
        root = repo_root / prefix
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".md", ".markdown"}:
                continue
            rel = path.relative_to(repo_root).as_posix()
            paths.append(rel)
    for rel in PUBLIC_EXTENSION_SCAFFOLD_RELATIVE_PATHS:
        if (repo_root / rel).is_file():
            paths.append(rel.replace("\\", "/"))
    return sorted(set(paths))


def scan_all_public_authoring_surfaces(repo_root: Path) -> list[PublicAuthoringNexusHit]:
    """Scan the Wave-1 public authoring inventory for Nexus import-like hits."""
    hits: list[PublicAuthoringNexusHit] = []
    for rel in iter_public_authoring_relative_paths(repo_root):
        hits.extend(scan_public_authoring_path(repo_root, rel))
    return hits


def scan_generated_agent_package(agent_dir: Path) -> list[PublicAuthoringNexusHit]:
    """AST/markdown scan of a scaffolded agent tree (``.py`` / ``.md`` / notebook cells)."""
    hits: list[PublicAuthoringNexusHit] = []
    if not agent_dir.is_dir():
        return hits
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(agent_dir).as_posix()
        suffix = path.suffix.lower()
        text = path.read_text(encoding="utf-8-sig")
        if suffix == ".py":
            if python_source_imports_nexus(text):
                hits.append(
                    PublicAuthoringNexusHit(
                        relative_path=rel,
                        kind="python_import",
                        detail="AST import of intergrax.runtime.nexus",
                    )
                )
            continue
        if suffix in {".md", ".markdown", ".rst"}:
            for snippet in markdown_fenced_python_imports_nexus(text):
                hits.append(
                    PublicAuthoringNexusHit(
                        relative_path=rel,
                        kind="markdown_fenced_python_import",
                        detail=snippet[:200],
                    )
                )
            continue
        if suffix == ".ipynb":
            hits.extend(_scan_notebook_nexus_hits(rel, text))
    return hits


def _scan_notebook_nexus_hits(relative_path: str, notebook_text: str) -> list[PublicAuthoringNexusHit]:
    """Scan notebook code-cell sources for Nexus imports (no OCR)."""
    import json

    hits: list[PublicAuthoringNexusHit] = []
    try:
        notebook = json.loads(notebook_text)
    except json.JSONDecodeError:
        return hits
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return hits
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            source = "".join(str(part) for part in raw_source)
        else:
            source = str(raw_source)
        try:
            if python_source_imports_nexus(source):
                hits.append(
                    PublicAuthoringNexusHit(
                        relative_path=relative_path,
                        kind="notebook_code_cell_import",
                        detail=f"cell[{index}] AST import of intergrax.runtime.nexus",
                    )
                )
        except SyntaxError:
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("from intergrax.runtime.nexus") or stripped.startswith(
                    "import intergrax.runtime.nexus"
                ):
                    hits.append(
                        PublicAuthoringNexusHit(
                            relative_path=relative_path,
                            kind="notebook_code_cell_import",
                            detail=f"cell[{index}] {stripped[:200]}",
                        )
                    )
                    break
    return hits
