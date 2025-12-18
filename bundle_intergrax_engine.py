# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "node_modules",
    ".eggs",
}

PY_EXT = ".py"


@dataclass(frozen=True)
class FileMeta:
    rel_path: str
    module_name: str
    module_group: str  # first segment after "intergrax/"
    sha256: str
    lines: int
    chars: int
    symbols: List[str]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def count_lines(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def is_excluded_path(path: Path) -> bool:
    parts = set(path.parts)
    return any(p in EXCLUDE_DIRS for p in parts)


def collect_python_files(package_dir: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirpath_p = Path(dirpath)

        # Prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        if is_excluded_path(dirpath_p):
            continue

        for fn in filenames:
            if not fn.endswith(PY_EXT):
                continue
            p = dirpath_p / fn
            if is_excluded_path(p):
                continue
            files.append(p)

    files.sort(key=lambda p: str(p).lower())
    return files


def to_rel(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root)).replace("\\", "/")


def to_module_name(rel_path: str) -> str:
    # rel_path starts with "intergrax/..."
    if not rel_path.endswith(".py"):
        return rel_path.replace("/", ".")
    p = rel_path[:-3].replace("/", ".")  # strip .py
    if p.endswith(".__init__"):
        p = p[: -len(".__init__")]
    return p


def module_group_from_rel(rel_path: str) -> str:
    """
    Grouping rules:
    - intergrax/<subfolder>/...   -> module_group = <subfolder>
    - intergrax/<file>.py         -> module_group = "root"
    """
    parts = rel_path.split("/")
    if not parts or parts[0] != "intergrax":
        return "root"

    # Example: ["intergrax", "chat_agent.py"]
    if len(parts) == 2 and parts[1].endswith(".py"):
        return "root"

    # Example: ["intergrax", "runtime", "engine.py"]
    if len(parts) >= 2:
        return parts[1] or "root"

    return "root"



def extract_symbols(py_text: str) -> List[str]:
    try:
        tree = ast.parse(py_text)
    except SyntaxError:
        return ["<syntax-error: unable to parse symbols>"]

    symbols: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(f"class {node.name}")
        elif isinstance(node, ast.AsyncFunctionDef):
            symbols.append(f"async def {node.name}()")
        elif isinstance(node, ast.FunctionDef):
            symbols.append(f"def {node.name}()")
    return symbols


def build_llm_header(project_root: Path, package_dir_name: str, metas: List[FileMeta]) -> str:
    """
    Build a top-of-file instruction block for LLMs.
    This goes into the generated bundle file.
    """
    module_groups = sorted({m.module_group for m in metas}, key=lambda s: s.lower())
    lines_total = sum(m.lines for m in metas)

    header: List[str] = []
    header.append("# ======================================================================\n")
    header.append("# LLM INSTRUCTIONS\n")
    header.append("# ======================================================================\n")
    header.append("# This file is an auto-generated, complete source code bundle of the Intergrax framework.\n")
    header.append("# It contains all Python modules from the package directory:\n")
    header.append(f"#   - Package root: {package_dir_name}/\n")
    header.append(f"#   - Project root: {project_root}\n")
    header.append("#\n")
    header.append("# IMPORTANT RULES FOR THE MODEL:\n")
    header.append("# 1) Treat THIS file as the single source of truth for Intergrax.\n")
    header.append("# 2) Do NOT assume any missing code exists elsewhere.\n")
    header.append("# 3) When proposing changes, always reference the exact FILE and MODULE headers below.\n")
    header.append("# 4) Prefer edits that preserve existing architecture, naming, and conventions.\n")
    header.append("#\n")
    header.append("# How to navigate this bundle:\n")
    header.append("# - Use the MODULE MAP and INDEX to find modules.\n")
    header.append("# - Each original file is included below with a header:\n")
    header.append("#     FILE: <relative path>\n")
    header.append("#     MODULE: <python import path>\n")
    header.append("#     MODULE_GROUP: <first folder under intergrax/>\n")
    header.append("#     SYMBOLS: <top-level classes/functions>\n")
    header.append("#\n")
    header.append("# Included module groups (dynamic):\n")
    for g in module_groups:
        header.append(f"# - {g}/\n")
    header.append("#\n")
    header.append(f"# Files included: {len(metas)}\n")
    header.append(f"# Total lines: {lines_total}\n")
    header.append("# ======================================================================\n\n")
    return "".join(header)


def build_bundle(
    project_root: Path,
    package_dir_name: str,
    out_path: Path,
    max_mb: int = 25,
    include_symbols: bool = True,
) -> List[FileMeta]:
    project_root = project_root.resolve()
    package_dir = (project_root / package_dir_name).resolve()

    if not package_dir.exists() or not package_dir.is_dir():
        raise SystemExit(f"Package dir not found: {package_dir}")

    paths = collect_python_files(package_dir)

    metas: List[FileMeta] = []
    for p in paths:
        rel = to_rel(project_root, p)
        txt = read_text(p)
        metas.append(
            FileMeta(
                rel_path=rel,
                module_name=to_module_name(rel),
                module_group=module_group_from_rel(rel),
                sha256=sha256_text(txt),
                lines=count_lines(txt),
                chars=len(txt),
                symbols=extract_symbols(txt) if include_symbols else [],
            )
        )

    # Automatic ordering (dynamic):
    # - group by first-level folder under intergrax/
    # - then sort by relative path
    metas.sort(key=lambda m: (m.module_group.lower(), m.rel_path.lower()))

    max_chars = None if max_mb <= 0 else max_mb * 1024 * 1024
    parts: List[str] = []

    # LLM instruction header (top of file)
    parts.append(build_llm_header(project_root, package_dir_name, metas))

    # Bundle header + maps
    parts.append("# INTERGRAX ENGINE BUNDLE (auto-generated)\n")
    parts.append(f"# ROOT: {project_root}\n")
    parts.append(f"# PACKAGE: {package_dir_name}\n")
    parts.append(f"# FILES: {len(metas)}\n")
    parts.append("#\n")

    # Module map (dynamic)
    module_map: Dict[str, List[FileMeta]] = {}
    for m in metas:
        module_map.setdefault(m.module_group, []).append(m)

    parts.append("# MODULE MAP (dynamic):\n")
    for group in sorted(module_map.keys(), key=lambda s: s.lower()):
        parts.append(f"# - {group}/ ({len(module_map[group])} files)\n")
    parts.append("#\n")

    # Index
    parts.append("# INDEX (path | module | module_group | lines | sha256[0:12]):\n")
    total_lines = 0
    for m in metas:
        total_lines += m.lines
        parts.append(
            f"# - {m.rel_path} | {m.module_name} | {m.module_group} | {m.lines} | {m.sha256[:12]}\n"
        )
    parts.append(f"#\n# TOTAL LINES: {total_lines}\n")
    parts.append("# ======================================================================\n\n")

    total_chars = sum(len(s) for s in parts)

    # File blocks
    for m in metas:
        header: List[str] = []
        header.append("# ======================================================================\n")
        header.append(f"# FILE: {m.rel_path}\n")
        header.append(f"# MODULE: {m.module_name}\n")
        header.append(f"# MODULE_GROUP: {m.module_group}\n")
        header.append("# TAGS:\n")
        header.append(f"#   - package={package_dir_name}\n")
        header.append(f"#   - module_group={m.module_group}\n")
        header.append(f"#   - file={Path(m.rel_path).name}\n")
        header.append(f"# LINES: {m.lines}\n")
        header.append(f"# SHA256: {m.sha256}\n")
        if include_symbols:
            header.append("# SYMBOLS:\n")
            if m.symbols:
                for s in m.symbols:
                    header.append(f"#   - {s}\n")
            else:
                header.append("#   - <none>\n")
        header.append("# ======================================================================\n")

        body = read_text(project_root / m.rel_path)
        if not body.endswith("\n"):
            body += "\n"

        chunk = "".join(header) + body + "\n"

        if max_chars is not None and (total_chars + len(chunk)) > max_chars:
            parts.append("# ======================================================================\n")
            parts.append(f"# TRUNCATED: bundle reached max_mb={max_mb}\n")
            parts.append("# ======================================================================\n")
            break

        parts.append(chunk)
        total_chars += len(chunk)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")

    return metas


def main() -> None:
    # No-args version:
    # - script is placed in project root (same directory as LICENSE.txt)
    # - output is created next to this script (project root)
    script_dir = Path(__file__).resolve().parent

    package_dir_name = "intergrax"
    out_file_name = "INTERGRAX_ENGINE_BUNDLE._py_"

    metas = build_bundle(
        project_root=script_dir,
        package_dir_name=package_dir_name,
        out_path=script_dir / out_file_name,
        max_mb=25,
        include_symbols=True,
    )

    print(f"Bundle created: {out_file_name}")
    print(f"Files included: {len(metas)}")


if __name__ == "__main__":
    main()
