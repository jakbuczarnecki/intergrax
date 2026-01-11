# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BUNDLES_DIR_NAME = "bundles"

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
IPYNB_EXT = ".ipynb"
SOURCE_EXTS = (PY_EXT, IPYNB_EXT)


# ----------------------------------------------------------------------
# define extra module bundles here
# keys  -> output file name (without extension handling; we'll append ". _py" style exactly)
# values-> folder path relative to project root
# ----------------------------------------------------------------------
EXTRA_BUNDLES: Dict[str, str] = {
    "ENGINE_RUNTIME_BUNDLE": r"intergrax\runtime\nexus",
    "ENGINE_RUNTIME_TRACING_BUNDLE": r"intergrax\runtime\nexus\tracing",
    "LLM": r"intergrax\llm",
    "LLM_ADAPTERS": r"intergrax\llm_adapters",
    "MEMORY": r"intergrax\memory",
    "MULTIMEDIA": r"intergrax\multimedia",
    "RAG": r"intergrax\rag",
    "SUPERVISOR": r"intergrax\supervisor",
    "TOOLS": r"intergrax\tools",
    "WEBSEARCH": r"intergrax\websearch",
    "FULL_BUNDLE": r"intergrax",
    "NOTEBOOKS": r"notebooks",
    "TESTS": r"tests",
}


@dataclass(frozen=True)
class FileMeta:
    rel_path: str
    module_name: str
    module_group: str  # first segment after "intergrax/" or a known top folder like "notebooks"
    sha256: str
    lines: int
    chars: int
    symbols: List[str]


# =============================================================================
# Reading / normalization
# =============================================================================

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


def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {}


def _as_list(x: object) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            if isinstance(item, str):
                out.append(item)
            else:
                out.append(str(item))
        return out
    if isinstance(x, str):
        return [x]
    return [str(x)]


def _normalize_newlines(s: str) -> str:
    # Keep output deterministic
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _render_ipynb_to_text(path: Path) -> str:
    """
    Convert .ipynb JSON into a readable, LLM-friendly plain text form.
    - code cells: keep source
    - markdown cells: keep as commented text
    - include cell indices + types
    """
    raw = read_text(path)
    data = _safe_json_loads(raw)

    nb_cells = data.get("cells", [])
    if not isinstance(nb_cells, list):
        nb_cells = []

    parts: List[str] = []
    parts.append(f"# NOTEBOOK: {path.name}\n")
    parts.append("# FORMAT: rendered from .ipynb to plain text for LLM analysis\n\n")

    for idx, cell in enumerate(nb_cells, start=1):
        if not isinstance(cell, dict):
            continue

        cell_type = str(cell.get("cell_type", "unknown"))
        src = cell.get("source", [])
        src_lines = _as_list(src)
        src_text = _normalize_newlines("".join(src_lines))

        parts.append("# ----------------------------------------------------------------------\n")
        parts.append(f"# CELL {idx:03d} | type={cell_type}\n")
        parts.append("# ----------------------------------------------------------------------\n")

        if cell_type == "markdown":
            # Preserve as comments so it doesn't break python parsing elsewhere.
            if src_text.strip():
                for line in src_text.split("\n"):
                    parts.append(f"# {line}\n")
            else:
                parts.append("# <empty markdown>\n")
            parts.append("\n")
        elif cell_type == "code":
            if src_text.strip():
                # Ensure code ends with newline
                if not src_text.endswith("\n"):
                    src_text += "\n"
                parts.append(src_text)
            else:
                parts.append("# <empty code>\n")
            parts.append("\n")
        else:
            # Unknown cell type: still preserve content safely
            if src_text.strip():
                for line in src_text.split("\n"):
                    parts.append(f"# {line}\n")
            else:
                parts.append("# <empty cell>\n")
            parts.append("\n")

    out = "".join(parts)
    if not out.endswith("\n"):
        out += "\n"
    return out


def read_source_for_bundle(path: Path) -> str:
    """
    Unified reader:
    - .py -> raw text
    - .ipynb -> rendered text
    """
    if path.suffix.lower() == IPYNB_EXT:
        return _render_ipynb_to_text(path)
    return _normalize_newlines(read_text(path))


# =============================================================================
# Scanning / selection
# =============================================================================

def is_excluded_path(path: Path) -> bool:
    parts = set(path.parts)
    return any(p in EXCLUDE_DIRS for p in parts)


def _is_source_filename(fn: str) -> bool:
    fn_l = fn.lower()
    return fn_l.endswith(PY_EXT) or fn_l.endswith(IPYNB_EXT)


def collect_source_files(root_dir: Path) -> List[Path]:
    """
    Collect all .py and .ipynb files under root_dir (recursive) while respecting EXCLUDE_DIRS.
    """
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirpath_p = Path(dirpath)

        # Prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        if is_excluded_path(dirpath_p):
            continue

        for fn in filenames:
            if not _is_source_filename(fn):
                continue
            p = dirpath_p / fn
            if is_excluded_path(p):
                continue
            files.append(p)

    files.sort(key=lambda p: str(p).lower())
    return files


# =============================================================================
# Path -> rel/module/group
# =============================================================================

def to_rel(project_root: Path, path: Path) -> str:
    return str(path.relative_to(project_root)).replace("\\", "/")


def to_module_name(rel_path: str) -> str:
    # For .py: strip extension and map to import path.
    if rel_path.endswith(PY_EXT):
        p = rel_path[: -len(PY_EXT)].replace("/", ".")
        if p.endswith(".__init__"):
            p = p[: -len(".__init__")]
        return p

    # For .ipynb (or other): keep extension in name by default (stable + unambiguous).
    # If you want notebook "imports" later, you can change this mapping.
    return rel_path.replace("/", ".")


def module_group_from_rel(rel_path: str) -> str:
    """
    Grouping rules:
    - intergrax/<subfolder>/...   -> module_group = <subfolder>
    - intergrax/<file>.py         -> module_group = "root"
    - notebooks/...               -> module_group = "notebooks"
    - tests/...                   -> module_group = "tests"
    - otherwise                   -> first top-level folder or "root"
    """
    parts = rel_path.split("/")
    if not parts:
        return "root"

    top = parts[0]
    if top == "intergrax":
        # Example: ["intergrax", "chat_agent.py"]
        if len(parts) == 2 and parts[1].endswith(PY_EXT):
            return "root"
        # Example: ["intergrax", "runtime", "engine.py"]
        if len(parts) >= 2:
            return parts[1] or "root"
        return "root"

    if top in {"notebooks", "tests"}:
        return top

    # Default: group by top-level folder if present, else root
    return top or "root"


# =============================================================================
# Symbol extraction (py + ipynb(code))
# =============================================================================

def _strip_ipython_magics(code: str) -> str:
    """
    Best-effort cleanup so ast.parse doesn't fail on notebook magics.
    - Remove lines starting with %, !, or containing cell magics like %%time.
    """
    cleaned: List[str] = []
    for line in _normalize_newlines(code).split("\n"):
        s = line.lstrip()
        if s.startswith("%%"):
            continue
        if s.startswith("%"):
            continue
        if s.startswith("!"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_symbols(py_like_text: str) -> List[str]:
    # Attempt to parse; if fails, return marker.
    try:
        tree = ast.parse(py_like_text)
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


def extract_symbols_for_file(path: Path, rendered_text: str) -> List[str]:
    """
    - .py: parse raw file text
    - .ipynb: parse only code segments (rendered output already contains markdown as comments,
      but it may still include non-python headers; we do best-effort by stripping magics and
      ignoring comment-only lines is fine for AST.
    """
    txt = rendered_text
    if path.suffix.lower() == IPYNB_EXT:
        txt = _strip_ipython_magics(txt)
    return extract_symbols(txt)


# =============================================================================
# Bundle building
# =============================================================================

def build_llm_header(project_root: Path, bundle_scope: str, metas: List[FileMeta]) -> str:
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
    header.append("#\n")
    header.append("# Bundle scope:\n")
    header.append(f"#   - {bundle_scope}\n")
    header.append(f"#   - Project root: {project_root}\n")
    header.append("#\n")
    header.append("# IMPORTANT RULES FOR THE MODEL:\n")
    header.append("# 1) Treat THIS file as the single source of truth for the included scope.\n")
    header.append("# 2) Do NOT assume any missing code exists elsewhere.\n")
    header.append("# 3) When proposing changes, always reference the exact FILE and MODULE headers below.\n")
    header.append("# 4) Prefer edits that preserve existing architecture, naming, and conventions.\n")
    header.append("#\n")
    header.append("# How to navigate this bundle:\n")
    header.append("# - Use the MODULE MAP and INDEX to find modules.\n")
    header.append("# - Each original file is included below with a header:\n")
    header.append("#     FILE: <relative path>\n")
    header.append("#     MODULE: <python import path or stable pseudo-module for non-.py>\n")
    header.append("#     MODULE_GROUP: <first folder under intergrax/ or a known top-level folder>\n")
    header.append("#     SYMBOLS: <top-level classes/functions (best-effort)>\n")
    header.append("#\n")
    header.append("# Included module groups (dynamic):\n")
    for g in module_groups:
        header.append(f"# - {g}/\n")
    header.append("#\n")
    header.append(f"# Files included: {len(metas)}\n")
    header.append(f"# Total lines: {lines_total}\n")
    header.append("# ======================================================================\n\n")
    return "".join(header)


def _build_metas_from_paths(
    *,
    project_root: Path,
    paths: List[Path],
    include_symbols: bool,
) -> List[FileMeta]:
    metas: List[FileMeta] = []
    for p in paths:
        rel = to_rel(project_root, p)
        txt = read_source_for_bundle(p)

        metas.append(
            FileMeta(
                rel_path=rel,
                module_name=to_module_name(rel),
                module_group=module_group_from_rel(rel),
                sha256=sha256_text(txt),
                lines=count_lines(txt),
                chars=len(txt),
                symbols=extract_symbols_for_file(p, txt) if include_symbols else [],
            )
        )

    # Automatic ordering (dynamic):
    # - group by module_group
    # - then sort by relative path
    metas.sort(key=lambda m: (m.module_group.lower(), m.rel_path.lower()))
    return metas


def build_bundle_from_paths(
    *,
    project_root: Path,
    out_path: Path,
    paths: List[Path],
    bundle_title: str,
    bundle_scope: str,
    max_mb: int = 25,
    include_symbols: bool = True,
) -> List[FileMeta]:
    """
    Generate a bundle from a pre-selected list of source file paths.
    Supported: .py, .ipynb
    Paths MUST be absolute or project_root-relative (we resolve anyway).
    """
    project_root = project_root.resolve()

    resolved_paths: List[Path] = []
    for p in paths:
        rp = p
        if not rp.is_absolute():
            rp = (project_root / rp).resolve()
        resolved_paths.append(rp)

    # Filter out non-existing (fail fast for safety)
    for p in resolved_paths:
        if not p.exists() or not p.is_file():
            raise SystemExit(f"File not found: {p}")
        if p.suffix.lower() not in {PY_EXT, IPYNB_EXT}:
            raise SystemExit(f"Unsupported file type: {p}")

    metas = _build_metas_from_paths(project_root=project_root, paths=resolved_paths, include_symbols=include_symbols)

    max_chars: Optional[int] = None if max_mb <= 0 else max_mb * 1024 * 1024
    parts: List[str] = []

    # LLM instruction header (top of file)
    parts.append(build_llm_header(project_root, bundle_scope, metas))

    # Bundle header + maps
    parts.append(f"# {bundle_title} (auto-generated)\n")
    parts.append(f"# ROOT: {project_root}\n")
    parts.append(f"# SCOPE: {bundle_scope}\n")
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
        parts.append(f"# - {m.rel_path} | {m.module_name} | {m.module_group} | {m.lines} | {m.sha256[:12]}\n")
    parts.append(f"#\n# TOTAL LINES: {total_lines}\n")
    parts.append("# ======================================================================\n\n")

    total_chars = sum(len(s) for s in parts)

    # File blocks
    for m in metas:
        abs_path = (project_root / Path(m.rel_path)).resolve()

        header: List[str] = []
        header.append("# ======================================================================\n")
        header.append(f"# FILE: {m.rel_path}\n")
        header.append(f"# MODULE: {m.module_name}\n")
        header.append(f"# MODULE_GROUP: {m.module_group}\n")
        header.append("# TAGS:\n")
        header.append(f"#   - scope={bundle_scope}\n")
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

        body = read_source_for_bundle(abs_path)
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


def build_extra_bundles(
    *,
    project_root: Path,
    bundles: Dict[str, str],
    bundles_dir: Path,
    max_mb: int = 25,
    include_symbols: bool = True,
) -> None:
    """
    Generate additional bundles based on a dict:
      key   -> output filename stem (we will generate: <key>._py)
      value -> folder path relative to project_root
    Supported files: .py, .ipynb
    """
    project_root = project_root.resolve()

    for out_name, rel_folder in bundles.items():
        folder = (project_root / Path(rel_folder)).resolve()
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"Extra bundle folder not found: {folder} (key={out_name})")

        paths = collect_source_files(folder)

        # Output file exactly as requested: "<name>._py"
        out_path = bundles_dir / f"{out_name}._py"

        build_bundle_from_paths(
            project_root=project_root,
            out_path=out_path,
            paths=paths,
            bundle_title=f"INTERGRAX MODULE BUNDLE: {out_name}",
            bundle_scope=f"folder={to_rel(project_root, folder)}/",
            max_mb=max_mb,
            include_symbols=include_symbols,
        )

        print(f"Module bundle created: {out_path.name}  (files={len(paths)})")


def main() -> None:
    # No-args version:
    # - script is placed in project root (same directory as LICENSE.txt)
    # - output is created next to this script (project root)
    script_dir = Path(__file__).resolve().parent

    bundles_dir = script_dir / BUNDLES_DIR_NAME
    bundles_dir.mkdir(parents=True, exist_ok=True)

    if EXTRA_BUNDLES:
        build_extra_bundles(
            project_root=script_dir,
            bundles=EXTRA_BUNDLES,
            bundles_dir=bundles_dir,
            max_mb=25,
            include_symbols=True,
        )


if __name__ == "__main__":
    main()
