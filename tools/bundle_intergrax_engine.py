# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

# Schema v4: STRUCTURE.json is now a true lightweight index (no graphs, no per-file code_objects duplication).
BUNDLE_SCHEMA_VERSION = 4

# =============================================================================
# Source type registry (edit here)
# =============================================================================
# Register extensions (lowercase, including the dot) and optional renderers.
# - Renderer takes a file Path and returns normalized text to embed into bundle.
# - Use None for plain text files (read_text + newline normalization).
SourceRenderer = Callable[[Path], str]

# Keep symbol extraction only for "python-like" sources.
SYMBOL_EXTS = {".py", ".ipynb"}

# You can extend this freely: add ".yaml", ".yml", ".toml", ".md", ".txt", etc.
# For most textual formats, renderer=None is enough.
SOURCE_HANDLERS: Dict[str, Optional[SourceRenderer]] = {
    ".py": None,       # plain text
    ".ipynb": None,    # will be set to _render_ipynb_to_text after function is defined
    ".yaml": None,
    ".yml": None,
}

# Derived set used by scanning/validation (do not edit)
SOURCE_EXTS = tuple(sorted(SOURCE_HANDLERS.keys()))

# ----------------------------------------------------------------------
# define extra module bundles here
# keys  -> output file name stem (we append "._py" exactly)
# values-> folder path relative to project root
# ----------------------------------------------------------------------
EXTRA_BUNDLES: Dict[str, str] = {
    "NEXUS_FULL_BUNDLE": r"intergrax\runtime\nexus",
    "NEXUS_TRACING_BUNDLE": r"intergrax\runtime\nexus\tracing",
    "NEXUS_RUNTIME_STEPS": r"intergrax\runtime\nexus\runtime_steps",

    "LLM": r"intergrax\llm",
    "LLM_ADAPTERS": r"intergrax\llm_adapters",

    "MEMORY": r"intergrax\memory",

    "MULTIMEDIA": r"intergrax\multimedia",

    "RAG": r"intergrax\rag",

    "PROMPTS_ENGINE": r"intergrax\prompts",

    "SUPERVISOR": r"intergrax\supervisor",

    "TOOLS": r"intergrax\tools",

    "WEBSEARCH": r"intergrax\websearch",

    "INTERGRAX_FULL_BUNDLE": r"intergrax",
    
    "NOTEBOOKS": r"notebooks",

    "TESTS": r"tests",
    "TESTS_UNITS": r"tests\unit",

    "PROMPTS": r"prompts",

    "RUNTIME": r"intergrax\runtime",

    "FASTAPI_CORE": r"intergrax\fastapi_core",
}

# =============================================================================
# Typed metadata
# =============================================================================

@dataclass(frozen=True)
class CodeObjectMeta:
    """
    Code object metadata for stable reference linking.
    The 'coid' is stable per object content (AST-based), independent from line numbers.
    """
    symbol: str                 # e.g. "pkg.mod:Class.method" or "pkg.mod:function"
    kind: str                   # "class" | "function" | "method"
    coid: str                   # sha256(...)
    lineno: Optional[int]       # best-effort (may be None)
    end_lineno: Optional[int]   # best-effort (may be None)


@dataclass(frozen=True)
class FileMeta:
    rel_path: str
    module_name: str
    module_group: str  # first segment after "intergrax/" or a known top folder like "notebooks"
    sha256: str
    lines: int
    chars: int

    # python-like only (best-effort)
    symbols: List[str]                  # class/function symbols only (module:Name)
    imports: List[str]                  # import modules
    code_objects: List[CodeObjectMeta]  # COID per class/function/method


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
            if src_text.strip():
                for line in src_text.split("\n"):
                    parts.append(f"# {line}\n")
            else:
                parts.append("# <empty markdown>\n")
            parts.append("\n")
        elif cell_type == "code":
            if src_text.strip():
                if not src_text.endswith("\n"):
                    src_text += "\n"
                parts.append(src_text)
            else:
                parts.append("# <empty code>\n")
            parts.append("\n")
        else:
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


# Bind ipynb renderer in the registry
SOURCE_HANDLERS[".ipynb"] = _render_ipynb_to_text


def read_source_for_bundle(path: Path) -> str:
    """
    Unified reader based on SOURCE_HANDLERS registry.
    """
    suffix = path.suffix.lower()
    renderer = SOURCE_HANDLERS.get(suffix)
    if renderer is not None:
        return _normalize_newlines(renderer(path))
    return _normalize_newlines(read_text(path))


# =============================================================================
# Scanning / selection
# =============================================================================

def is_excluded_path(path: Path) -> bool:
    parts = set(path.parts)
    return any(p in EXCLUDE_DIRS for p in parts)


def _is_source_filename(fn: str) -> bool:
    fn_l = fn.lower()
    return any(fn_l.endswith(ext) for ext in SOURCE_EXTS)


def collect_source_files(root_dir: Path) -> List[Path]:
    """
    Collect all registered source files under root_dir (recursive) while respecting EXCLUDE_DIRS.
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
    """
    For .py: strip extension and map to python import path.
    For everything else: use a stable pseudo-module that includes extension.
    """
    if rel_path.endswith(".py"):
        p = rel_path[: -len(".py")].replace("/", ".")
        if p.endswith(".__init__"):
            p = p[: -len(".__init__")]
        return p
    return rel_path.replace("/", ".")


def module_group_from_rel(rel_path: str) -> str:
    """
    Grouping rules (structure-only, no architectural meaning).
    """
    parts = rel_path.split("/")
    if not parts:
        return "root"

    top = parts[0]
    if top == "intergrax":
        if len(parts) == 2 and parts[1].endswith(".py"):
            return "root"
        if len(parts) >= 2:
            return parts[1] or "root"
        return "root"

    if top in {"notebooks", "tests"}:
        return top

    return top or "root"


# =============================================================================
# Python-like analysis: symbols, imports, COID (single AST pass)
# =============================================================================

def _strip_ipython_magics(code: str) -> str:
    """
    Best-effort cleanup so ast.parse doesn't fail on notebook magics.
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


def _ast_parse_best_effort(py_like_text: str) -> Optional[ast.AST]:
    try:
        return ast.parse(py_like_text)
    except SyntaxError:
        return None


def _ast_normalized_dump(node: ast.AST) -> str:
    """
    Stable representation for hashing.
    No attributes = independent from line numbers and formatting.
    """
    return ast.dump(node, include_attributes=False)


def _compute_coid(module_name: str, qualified_name: str, kind: str, node: ast.AST) -> str:
    raw = f"{module_name}|{qualified_name}|{kind}|{_ast_normalized_dump(node)}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def _extract_python_like_metadata(path: Path, rendered_text: str, module_name: str) -> Tuple[List[str], List[str], List[CodeObjectMeta]]:
    """
    Single AST pass:
    - symbols: class/function (module:Name)
    - imports: best-effort import modules list
    - code_objects: class/function/method with COID
    """
    suffix = path.suffix.lower()
    if suffix not in SYMBOL_EXTS:
        return ([], [], [])

    txt = rendered_text
    if suffix == ".ipynb":
        txt = _strip_ipython_magics(txt)

    tree = _ast_parse_best_effort(txt)
    if tree is None:
        # Keep deterministic fallback for symbols
        return ([f"{module_name}:<syntax-error>"], [], [])

    symbols: List[str] = []
    imports: List[str] = []
    code_objects: List[CodeObjectMeta] = []

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name:
                    imports.append(str(alias.name))
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                imports.append(str(node.module))
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            # class object
            sym = f"{module_name}:{node.name}"
            symbols.append(sym)

            coid = _compute_coid(module_name, node.name, "class", node)
            code_objects.append(
                CodeObjectMeta(
                    symbol=sym,
                    kind="class",
                    coid=coid,
                    lineno=getattr(node, "lineno", None),
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

            # methods
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = f"{node.name}.{sub.name}"
                    sym_m = f"{module_name}:{qn}"
                    coid_m = _compute_coid(module_name, qn, "method", sub)
                    code_objects.append(
                        CodeObjectMeta(
                            symbol=sym_m,
                            kind="method",
                            coid=coid_m,
                            lineno=getattr(sub, "lineno", None),
                            end_lineno=getattr(sub, "end_lineno", None),
                        )
                    )

            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            sym = f"{module_name}:{node.name}"
            symbols.append(sym)

            coid = _compute_coid(module_name, node.name, "function", node)
            code_objects.append(
                CodeObjectMeta(
                    symbol=sym,
                    kind="function",
                    coid=coid,
                    lineno=getattr(node, "lineno", None),
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            sym = f"{module_name}:{node.name}"
            symbols.append(sym)

            coid = _compute_coid(module_name, node.name, "function", node)
            code_objects.append(
                CodeObjectMeta(
                    symbol=sym,
                    kind="function",
                    coid=coid,
                    lineno=getattr(node, "lineno", None),
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

            self.generic_visit(node)

    Visitor().visit(tree)

    # Deduplicate + deterministic order
    symbols = sorted(set(symbols), key=lambda s: s.lower())
    imports = sorted(set(imports), key=lambda s: s.lower())
    code_objects.sort(key=lambda o: (o.symbol.lower(), o.kind.lower(), o.coid))
    return (symbols, imports, code_objects)


# =============================================================================
# STRUCTURE.json (true lightweight index)
# =============================================================================

def build_structure_index(
    *,
    bundle_filename: str,
    bundle_scope: str,
    project_root: Path,
    metas: List[FileMeta],
) -> dict:
    """
    ULTRA-LIGHT STRUCTURE INDEX.

    Purpose:
    - Navigation only
    - No symbols
    - No COID
    - No stats
    - No duplication of bundle semantics
    """

    modules_index: Dict[str, str] = {}

    for m in metas:
        modules_index[m.module_name] = m.rel_path

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle": {
            "filename": bundle_filename,
            "scope": bundle_scope,
            "project_root": str(project_root),
        },
        "modules": modules_index,
    }



# =============================================================================
# LLM Instructions content (separate file)
# =============================================================================

def build_llm_instructions_text(*, bundle_filename: str, bundle_scope: str, structure_filename: str) -> str:
    """
    Generate an English instruction file that the user can paste into chat,
    while requiring responses in Polish.
    """
    text = f"""# Intergrax Bundle Reading Instructions (LLM)

Bundle file: {bundle_filename}
Structure file: {structure_filename}
Scope: {bundle_scope}

These instructions define a deterministic, safety-critical procedure for analyzing and modifying Intergrax code.
Deviation from this procedure is considered a failure.

The bundle is the single source of truth for code content.
The structure file is the single source of truth for indexing.

Response language requirement:
- Provide all answers in Polish (PL).
- Keep code comments in English (EN).

=====================================================================
0) OPERATING MODEL (MANDATORY)
=====================================================================

You operate strictly as a code-referencing system.

Allowed knowledge sources:
1) Structure index ({structure_filename})
2) Bundle ({bundle_filename})

Anything not explicitly present there is UNKNOWN.

If any step cannot be completed using only these two files → STOP.

=====================================================================
1) SCOPE GATE (HARD STOP)
=====================================================================

Before proposing any patch you MUST list:
- ALL files to be modified
- ALL files to be read for context

If ANY file is outside bundle scope or missing:
→ STOP immediately
→ Request missing bundle

No assumptions allowed.

=====================================================================
2) ZERO-HALLUCINATION RULE (HARD STOP)
=====================================================================

You MUST NOT:
- invent variable names
- invent parameters
- invent method signatures
- rewrite signatures in your own format
- "simplify" code for explanation

If exact code cannot be located verbatim → STOP.

=====================================================================
3) DETERMINISTIC WORKFLOW (FIXED ORDER)
=====================================================================

You MUST follow these steps in order:

STEP 1 — STRUCTURE LOOKUP
Use structure index ONLY to locate:
- file
- module
- COID(s)

STEP 2 — VERBATIM SOURCE EXTRACTION
Open bundle and copy exact source fragment.

STEP 3 — VERIFICATION PASS
List:
- FILE
- MODULE
- COID
- VERBATIM SOURCE BLOCK

STEP 4 — ONLY AFTER THAT → PATCH

Skipping any step = invalid response.

=====================================================================
4) VERBATIM SOURCE GATE (CRITICAL)
=====================================================================

Before ANY modification proposal you MUST include VERBATIM SOURCE copied from bundle (not paraphrased).

Must include:
- full function/method signature
- exact parameter names
- local variable names

If signature differs by one parameter → INVALID.

If VERBATIM block is missing → STOP.

=====================================================================
5) SIGNATURE LOCK RULE
=====================================================================

Signatures are immutable references.

You may not:
- reorder parameters
- rename parameters
- change type annotations
- omit default values

Unless task explicitly says so.

=====================================================================
6) STRUCTURE INDEX USAGE (MANDATORY)
=====================================================================

Use {structure_filename} for:
- files[module]
- symbols[symbol]
- coid_index[coid]

COID is the primary reference key.

=====================================================================
7) BUNDLE NAVIGATION
=====================================================================

Bundle headers define truth:
- FILE
- MODULE
- MODULE_GROUP
- SHA256
- CODE_OBJECTS (COID)

Bundle text overrides any assumptions.

=====================================================================
8) PATCH FORMAT
=====================================================================

For each file:
- FILE path
- VERBATIM SOURCE (before)
- PATCH BLOCK (after)
- short justification

=====================================================================
ENFORCEMENT RULE
=====================================================================

If ANY rule above cannot be satisfied:
→ STOP
→ Explain which rule blocks continuation
→ Ask for missing bundle scope

End of instructions.
"""
    return text


# =============================================================================
# Bundle building
# =============================================================================

def build_llm_header(project_root: Path, bundle_scope: str, metas: List[FileMeta]) -> str:
    module_groups = sorted({m.module_group for m in metas}, key=lambda s: s.lower())
    lines_total = sum(m.lines for m in metas)

    header: List[str] = []
    header.append("# ======================================================================\n")
    header.append("# LLM INSTRUCTIONS (embedded)\n")
    header.append("# ======================================================================\n")
    header.append("# This file is an auto-generated, complete source code bundle of the Intergrax framework.\n")
    header.append("#\n")
    header.append("# BUNDLE_SCHEMA_VERSION:\n")
    header.append(f"#   - {BUNDLE_SCHEMA_VERSION}\n")
    header.append("#\n")
    header.append("# Bundle scope:\n")
    header.append(f"#   - {bundle_scope}\n")
    header.append(f"#   - Project root: {project_root}\n")
    header.append("#\n")
    header.append("# CODE_OBJECTS are AST-based and stable via COID.\n")
    header.append("# IMPORTANT RULES FOR THE MODEL:\n")
    header.append("# 1) Treat THIS file as the single source of truth for the included scope.\n")
    header.append("# 2) Do NOT assume any missing code exists elsewhere.\n")
    header.append("# 3) Do NOT invent paths/classes/methods. Verify everything against the bundle.\n")
    header.append("# 4) When proposing changes, always reference the exact FILE and MODULE headers below.\n")
    header.append("# 5) Prefer minimal, backward-compatible edits.\n")
    header.append("#\n")
    header.append("# Included module groups (structural):\n")
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
        module_name = to_module_name(rel)
        txt = read_source_for_bundle(p)

        if include_symbols:
            syms, imps, code_objects = _extract_python_like_metadata(p, txt, module_name=module_name)
        else:
            syms, imps, code_objects = ([], [], [])

        metas.append(
            FileMeta(
                rel_path=rel,
                module_name=module_name,
                module_group=module_group_from_rel(rel),
                sha256=sha256_text(txt),
                lines=count_lines(txt),
                chars=len(txt),
                symbols=syms,
                imports=imps,
                code_objects=code_objects,
            )
        )

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
    write_instructions_file: bool = True,
    write_structure_file: bool = True,
) -> List[FileMeta]:
    """
    Generate a bundle from a pre-selected list of source file paths.
    Also generates:
    - <bundle_stem>_STRUCTURE.json (lightweight index)
    - <bundle_stem>_INSTRUCTIONS.md (LLM procedure)
    """
    project_root = project_root.resolve()

    resolved_paths: List[Path] = []
    for p in paths:
        rp = p if p.is_absolute() else (project_root / p).resolve()
        resolved_paths.append(rp)

    for p in resolved_paths:
        if not p.exists() or not p.is_file():
            raise SystemExit(f"File not found: {p}")
        if p.suffix.lower() not in SOURCE_HANDLERS:
            raise SystemExit(f"Unsupported file type: {p.suffix} (file={p})")

    metas = _build_metas_from_paths(project_root=project_root, paths=resolved_paths, include_symbols=include_symbols)
    write_llm_system_model_for_bundle(out_path=out_path, metas=metas)

    max_chars: Optional[int] = None if max_mb <= 0 else max_mb * 1024 * 1024
    parts: List[str] = []

    parts.append(build_llm_header(project_root, bundle_scope, metas))

    parts.append(f"# {bundle_title} (auto-generated)\n")
    parts.append(f"# ROOT: {project_root}\n")
    parts.append(f"# SCOPE: {bundle_scope}\n")
    parts.append(f"# FILES: {len(metas)}\n")
    parts.append("#\n")

    module_map: Dict[str, List[FileMeta]] = {}
    for m in metas:
        module_map.setdefault(m.module_group, []).append(m)

    parts.append("# MODULE MAP (structural):\n")
    for group in sorted(module_map.keys(), key=lambda s: s.lower()):
        parts.append(f"# - {group}/ ({len(module_map[group])} files)\n")
    parts.append("#\n")

    parts.append("# INDEX (path | module | module_group | lines | sha256[0:12]):\n")
    total_lines = 0
    for m in metas:
        total_lines += m.lines
        parts.append(f"# - {m.rel_path} | {m.module_name} | {m.module_group} | {m.lines} | {m.sha256[:12]}\n")
    parts.append(f"#\n# TOTAL LINES: {total_lines}\n")
    parts.append("# ======================================================================\n\n")

    total_chars = sum(len(s) for s in parts)

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
        header.append(f"# CHARS: {m.chars}\n")
        header.append(f"# SHA256: {m.sha256}\n")

        if include_symbols:
            header.append("# SYMBOLS (class/function only, best-effort):\n")
            if m.symbols:
                for s in m.symbols:
                    header.append(f"#   - {s}\n")
            else:
                header.append("#   - <none>\n")

            header.append("# IMPORTS (best-effort):\n")
            if m.imports:
                for imp in m.imports:
                    header.append(f"#   - {imp}\n")
            else:
                header.append("#   - <none>\n")

            header.append("# CODE_OBJECTS (COID) (best-effort, python-like only):\n")
            if m.code_objects:
                for o in m.code_objects:
                    header.append(
                        f"#   - COID={o.coid} | kind={o.kind} | symbol={o.symbol} | lineno={o.lineno} | end_lineno={o.end_lineno}\n"
                    )
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

    structure_path = out_path.parent / f"{out_path.stem}_STRUCTURE.json"
    if write_structure_file:
        structure = build_structure_index(
            bundle_filename=out_path.name,
            bundle_scope=bundle_scope,
            project_root=project_root,
            metas=metas,
        )
        structure_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")

    if write_instructions_file:
        instr_path = out_path.parent / f"{out_path.stem}_INSTRUCTIONS.md"
        instr_text = build_llm_instructions_text(
            bundle_filename=out_path.name,
            bundle_scope=bundle_scope,
            structure_filename=structure_path.name,
        )
        instr_path.write_text(instr_text, encoding="utf-8")

    return metas


def build_extra_bundles(
    *,
    project_root: Path,
    bundles: Dict[str, str],
    bundles_dir: Path,
    max_mb: int = 25,
    include_symbols: bool = True,
    write_instructions_file: bool = True,
    write_structure_file: bool = True,
) -> None:
    """
    Generate additional bundles based on a dict:
      key   -> output filename stem (we will generate: <key>._py)
      value -> folder path relative to project_root
    """
    project_root = project_root.resolve()

    for out_name, rel_folder in bundles.items():
        folder = (project_root / Path(rel_folder)).resolve()
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"Extra bundle folder not found: {folder} (key={out_name})")

        paths = collect_source_files(folder)

        out_path = bundles_dir / f"{out_name}._py"

        build_bundle_from_paths(
            project_root=project_root,
            out_path=out_path,
            paths=paths,
            bundle_title=f"INTERGRAX MODULE BUNDLE: {out_name}",
            bundle_scope=f"folder={to_rel(project_root, folder)}/",
            max_mb=max_mb,
            include_symbols=include_symbols,
            write_instructions_file=write_instructions_file,
            write_structure_file=write_structure_file,
        )

        instr_name = f"{out_path.stem}_INSTRUCTIONS.md"
        struct_name = f"{out_path.stem}_STRUCTURE.json"

        print(f"Module bundle created: {out_path.name}  (files={len(paths)})")
        if write_structure_file:
            print(f"Structure created:    {struct_name}")
        if write_instructions_file:
            print(f"Instructions created: {instr_name}")


def find_project_root(start: Path) -> Path:
    """
    Walk up the directory tree to find project root (identified by pyproject.toml).
    """
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Project root not found (pyproject.toml missing)")



# =============================================================================
# LLM System Model files (per-bundle) — ADDED
# =============================================================================

def write_arch_graph_file_for_bundle(*, out_path: Path, metas: List[FileMeta]) -> None:
    """
    ARCH_GRAPH for a bundle = structural grouping, not semantic architecture.
    No hardcoded knowledge about Intergrax domains.
    """

    stem = out_path.stem

    layers: Dict[str, List[str]] = {}

    for m in metas:
        group = m.module_group or "root"
        layers.setdefault(group, []).append(m.module_name)

    for k in layers:
        layers[k] = sorted(set(layers[k]))

    data = {
        "schema": "intergrax_arch_graph_v1",
        "bundle": {
            "stem": stem,
            "bundle_file": out_path.name,
        },
        "layers": layers,
    }

    (out_path.parent / f"{stem}_ARCH_GRAPH.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )



def write_dep_graph_file_for_bundle(*, out_path: Path, metas: List[FileMeta]) -> None:
    stem = out_path.stem

    forward: Dict[str, List[str]] = {}
    reverse: Dict[str, List[str]] = {}

    all_modules = {m.module_name for m in metas}

    for m in metas:
        if not m.imports:
            continue

        # tylko importy, które wskazują na moduły z bundla
        internal_imports = sorted({imp for imp in m.imports if imp in all_modules})
        if not internal_imports:
            continue

        forward[m.module_name] = internal_imports

        for dep in internal_imports:
            reverse.setdefault(dep, []).append(m.module_name)

    # sort deterministically
    for k in reverse:
        reverse[k] = sorted(set(reverse[k]))

    data = {
        "schema": "intergrax_dep_graph_v2",
        "bundle": {
            "stem": stem,
            "bundle_file": out_path.name,
        },

        "semantics": {
            "type": "static_import_graph",
            "meaning": "module A lists modules it imports",
            "NOT": [
                "runtime_call_graph",
                "execution_flow",
                "architecture_layers",
                "data_flow_graph",
            ],
            "direction": "forward = importer → imported",
        },

        "internal_modules_only": True,

        "forward_dependencies": forward,
        "reverse_dependencies": reverse,
    }

    (out_path.parent / f"{stem}_DEP_GRAPH.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )



def write_contracts_file_for_bundle(*, out_path: Path, metas: List[FileMeta]) -> None:
    """
    Generates mechanical symbol lock information.
    This is NOT an architectural contract.
    It only restricts modification operations for the LLM.
    """

    stem = out_path.stem

    locked: Dict[str, dict] = {}

    for m in metas:
        mn = m.module_name.lower()

        # core runtime surface
        if ".runtime." in mn or ".fastapi_core." in mn:
            for obj in m.code_objects:
                locked[obj.coid] = {
                    "symbol": obj.symbol,
                    "kind": obj.kind,
                    "lock_type": "soft_lock",
                    "reason": "core_runtime_surface",
                }

    data = {
        "schema": "intergrax_contracts_v2",
        "bundle": {
            "stem": stem,
            "bundle_file": out_path.name,
        },

        "semantics": {
            "meaning": "mechanical_modification_constraints",
            "NOT": [
                "public_api_definition",
                "architectural_boundary",
                "layer_definition",
            ],
        },

        "lock_policy": {
            "hard_lock": "signature and symbol existence cannot change",
            "soft_lock": "signature locked, body modifiable",
        },

        "locked_code_objects": locked,
    }

    (out_path.parent / f"{stem}_CONTRACTS.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )



def write_patch_zones_file_for_bundle(*, out_path: Path, metas: List[FileMeta]) -> None:
    """
    Patch zones define edit-risk classification for LLM operations.
    They DO NOT define architecture, ownership, or importance.
    """

    stem = out_path.stem

    safe: List[str] = []
    restricted: List[str] = []
    locked: List[str] = []

    for m in metas:
        mn = m.module_name.lower()
        mg = m.module_group.lower()

        # low-risk areas (feature logic, extensions, tests)
        if (
            ".tools." in mn
            or mg == "tests"
            or mn.startswith("tests.")
            or ".prompts." in mn
            or mn.startswith("prompts.")
        ):
            safe.append(m.rel_path)

        # core system areas (editable but high impact)
        elif ".runtime." in mn or ".fastapi_core." in mn:
            restricted.append(m.rel_path)

        # everything else is default locked unless explicitly requested
        else:
            locked.append(m.rel_path)

    data = {
        "schema": "intergrax_patch_zones_v2",
        "bundle": {
            "stem": stem,
            "bundle_file": out_path.name,
        },

        "semantics": {
            "meaning": "edit_risk_classification",
            "NOT": [
                "architecture_definition",
                "ownership",
                "importance_level",
                "permission_model",
            ],
        },

        "zones": {
            "safe": sorted(set(safe)),
            "restricted": sorted(set(restricted)),
            "locked": sorted(set(locked)),
        },

        "zone_policy": {
            "safe": "LLM may modify with standard verification",
            "restricted": "LLM must minimize edits and verify dependencies",
            "locked": "LLM must not modify unless explicitly instructed",
        },
    }

    (out_path.parent / f"{stem}_PATCH_ZONES.json").write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )



def write_llm_protocol_file_for_bundle(*, out_path: Path) -> None:
    stem = out_path.stem

    text = f"""# Intergrax LLM Operating Protocol (Per-Bundle)

Bundle stem: {stem}
Bundle file: {out_path.name}

====================================================================
SYSTEM MODEL
====================================================================

You are operating on a MACHINE-GENERATED CODE MODEL.
This is NOT documentation.
This is NOT architecture description.
This is a deterministic mechanical representation of source code.

You are an IMPLEMENTATION ENGINE, not a reasoning agent.

====================================================================
TRUTH HIERARCHY (CRITICAL)
====================================================================

1) {stem}._py
   → The ONLY source of truth about code.
   → If something is not present here verbatim, it does not exist.

2) {stem}_CONTRACTS.json
   → Defines modification constraints.
   → Overrides any assumption about editability.

3) {stem}_STRUCTURE.json
   → Maps modules → files.
   → Navigation only. No semantics.

4) {stem}_DEP_GRAPH.json
   → Symbol reference graph.
   → Static relationships only.
   → NOT runtime flow.

5) {stem}_ARCH_GRAPH.json
   → Structural grouping by module_group.
   → NOT system architecture.

6) {stem}_PATCH_ZONES.json
   → Safe vs restricted regions inside files.

If two files appear to conflict → bundle ._py wins.

====================================================================
ROLE
====================================================================

You are NOT:
- an architect
- a refactoring agent
- a documentation system

You ARE:
a deterministic code patch engine operating strictly inside the bundle.

====================================================================
HARD RULES
====================================================================

- Do NOT invent symbols, methods, modules, paths.
- Do NOT reformat or "simplify" code for explanation.
- Do NOT change signatures unless explicitly instructed.
- Any referenced code must exist VERBATIM in bundle.

If a required symbol is not found verbatim → STOP.

====================================================================
WORKFLOW (MANDATORY ORDER)
====================================================================

STEP 1 — Locate file via STRUCTURE.json  
STEP 2 — Extract VERBATIM source from bundle  
STEP 3 — Verify against CONTRACTS.json (edit permissions)  
STEP 4 — Check DEP_GRAPH.json for impact scope  
STEP 5 — Check PATCH_ZONES.json for safe regions  
STEP 6 — Only then propose MINIMAL patch  

Skipping steps = invalid operation.

====================================================================
OUTPUT FORMAT
====================================================================

FILE:
VERBATIM SOURCE (before):
PATCH (after):
Justification (1–2 sentences max)

====================================================================
FAIL CONDITIONS
====================================================================

STOP if:
- symbol not found verbatim
- file not in STRUCTURE.json
- edit violates CONTRACTS.json
- edit outside allowed PATCH_ZONES
- scope incomplete

Ask for missing bundle scope instead of guessing.

End of protocol.
"""

    (out_path.parent / f"{stem}_LLM_PROTOCOL.md").write_text(text, encoding="utf-8")



def write_llm_system_model_for_bundle(*, out_path: Path, metas: List[FileMeta]) -> None:
    """
    Generates per-bundle LLM navigation/guard artifacts next to the bundle.
    """
    write_arch_graph_file_for_bundle(out_path=out_path, metas=metas)
    write_dep_graph_file_for_bundle(out_path=out_path, metas=metas)
    write_contracts_file_for_bundle(out_path=out_path, metas=metas)
    write_patch_zones_file_for_bundle(out_path=out_path, metas=metas)
    write_llm_protocol_file_for_bundle(out_path=out_path)





def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    bundles_dir = project_root / BUNDLES_DIR_NAME
    bundles_dir.mkdir(parents=True, exist_ok=True)

    if EXTRA_BUNDLES:
        build_extra_bundles(
            project_root=project_root,
            bundles=EXTRA_BUNDLES,
            bundles_dir=bundles_dir,
            max_mb=100,
            include_symbols=True,
            write_instructions_file=False,
            write_structure_file=True,
        )


if __name__ == "__main__":
    main()
