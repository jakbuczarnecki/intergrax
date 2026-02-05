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

# Bump schema version because we add STRUCTURE.json + COID index
BUNDLE_SCHEMA_VERSION = 3

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
    "PROMPTS": r"prompts",
    "FASTAPI_CORE": r"intergrax\fastapi_core",
}


# =============================================================================
# Typed metadata (extended)
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
    symbols: List[str]                 # global-unique ids, best-effort (python-like only)
    imports: List[str]                 # import modules, best-effort (python-like only)

    # New: execution/usage indices (python-like only)
    creates: List[str]                 # called names (best-effort)
    calls: List[str]                   # attribute calls + name calls (best-effort)
    type_usages: List[str]             # ast.Name identifiers (best-effort)

    # New: stable code object references (python-like only)
    code_objects: List[CodeObjectMeta]  # contains COID per class/function/method


@dataclass
class FileAstIndex:
    rel_path: str
    module: str
    imports: List[str]
    creates: List[str]
    calls: List[str]
    type_usages: List[str]
    code_objects: List[CodeObjectMeta]


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


# Bind ipynb renderer in the registry (no hardcoded ifs elsewhere)
SOURCE_HANDLERS[".ipynb"] = _render_ipynb_to_text


def read_source_for_bundle(path: Path) -> str:
    """
    Unified reader based on SOURCE_HANDLERS registry.
    - If a renderer is registered for a suffix -> use it.
    - Otherwise -> read_text + normalize newlines.
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
    Grouping rules (structure-only, no architectural meaning):
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
        if len(parts) == 2 and parts[1].endswith(".py"):
            return "root"
        if len(parts) >= 2:
            return parts[1] or "root"
        return "root"

    if top in {"notebooks", "tests"}:
        return top

    return top or "root"


# =============================================================================
# Python-like analysis: symbols, imports, call graph, COID
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


def extract_symbols(py_like_text: str, module_name: str) -> List[str]:
    tree = _ast_parse_best_effort(py_like_text)
    if tree is None:
        return [f"{module_name}:<syntax-error>"]

    symbols: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(f"{module_name}:{node.name}")
        elif isinstance(node, ast.AsyncFunctionDef):
            symbols.append(f"{module_name}:{node.name}")
        elif isinstance(node, ast.FunctionDef):
            symbols.append(f"{module_name}:{node.name}")
    return symbols


def extract_imports(py_like_text: str) -> List[str]:
    tree = _ast_parse_best_effort(py_like_text)
    if tree is None:
        return []

    imports: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.append(str(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(str(node.module))
    return sorted(set(imports), key=lambda s: s.lower())


def extract_call_and_usage_indices(py_like_text: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Best-effort indices to locate behavior without searching raw text.
    - creates: calls where func is ast.Name (often constructors or factory functions)
    - calls: both name calls and attribute calls
    - type_usages: all ast.Name identifiers (noisy but useful as an index)
    """
    tree = _ast_parse_best_effort(py_like_text)
    if tree is None:
        return ([], [], [])

    creates: List[str] = []
    calls: List[str] = []
    type_usages: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                creates.append(node.func.id)
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
        elif isinstance(node, ast.Name):
            type_usages.append(node.id)

    return (
        sorted(set(creates), key=lambda s: s.lower()),
        sorted(set(calls), key=lambda s: s.lower()),
        sorted(set(type_usages), key=lambda s: s.lower()),
    )


def extract_code_objects(py_like_text: str, module_name: str) -> List[CodeObjectMeta]:
    """
    Extract classes, functions and methods with stable COID.
    Also includes lineno/end_lineno if available (best-effort).
    """
    tree = _ast_parse_best_effort(py_like_text)
    if tree is None:
        return []

    objs: List[CodeObjectMeta] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            sym = f"{module_name}:{node.name}"
            coid = _compute_coid(module_name, node.name, "class", node)
            objs.append(
                CodeObjectMeta(
                    symbol=sym,
                    kind="class",
                    coid=coid,
                    lineno=getattr(node, "lineno", None),
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qn = f"{node.name}.{sub.name}"
                    sym_m = f"{module_name}:{qn}"
                    coid_m = _compute_coid(module_name, qn, "method", sub)
                    objs.append(
                        CodeObjectMeta(
                            symbol=sym_m,
                            kind="method",
                            coid=coid_m,
                            lineno=getattr(sub, "lineno", None),
                            end_lineno=getattr(sub, "end_lineno", None),
                        )
                    )

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = f"{module_name}:{node.name}"
            coid = _compute_coid(module_name, node.name, "function", node)
            objs.append(
                CodeObjectMeta(
                    symbol=sym,
                    kind="function",
                    coid=coid,
                    lineno=getattr(node, "lineno", None),
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

    # Deterministic ordering
    objs.sort(key=lambda o: (o.symbol.lower(), o.kind.lower(), o.coid))
    return objs


def extract_symbols_and_imports_for_file(path: Path, rendered_text: str, module_name: str) -> Tuple[List[str], List[str]]:
    """
    Extract symbols/imports only for extensions listed in SYMBOL_EXTS.
    """
    suffix = path.suffix.lower()
    if suffix not in SYMBOL_EXTS:
        return ([], [])

    txt = rendered_text
    if suffix == ".ipynb":
        txt = _strip_ipython_magics(txt)

    syms = extract_symbols(txt, module_name=module_name)
    imps = extract_imports(txt)
    return (syms, imps)


def extract_extended_python_like_metadata(path: Path, rendered_text: str, module_name: str):
    """
    Single AST pass → all advanced indices.
    This is the *only* source of truth for code_objects and COID.
    """
    suffix = path.suffix.lower()
    if suffix not in SYMBOL_EXTS:
        return ([], [], [], [])

    txt = rendered_text
    if suffix == ".ipynb":
        txt = _strip_ipython_magics(txt)

    tree = _ast_parse_best_effort(txt)
    if tree is None:
        return ([], [], [], [])

    creates: List[str] = []
    calls: List[str] = []
    type_usages: List[str] = []
    code_objects: List[CodeObjectMeta] = []

    class Visitor(ast.NodeVisitor):

        def visit_ClassDef(self, node: ast.ClassDef):
            sym = f"{module_name}:{node.name}"
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

            # detect constructor calls inside this class
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

        def visit_FunctionDef(self, node: ast.FunctionDef):
            sym = f"{module_name}:{node.name}"
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

        def visit_Call(self, node: ast.Call):
            # CALLS
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)

            # CREATES = ClassName(...)
            if isinstance(node.func, ast.Name) and node.func.id[0].isupper():
                creates.append(node.func.id)

            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            type_usages.append(node.id)

    Visitor().visit(tree)

    return (
        sorted(set(creates), key=lambda s: s.lower()),
        sorted(set(calls), key=lambda s: s.lower()),
        sorted(set(type_usages), key=lambda s: s.lower()),
        code_objects,
    )


# =============================================================================
# Structural map builders (no hardcoded architecture)
# =============================================================================

def build_folder_tree(metas: List[FileMeta]) -> str:
    """
    Build a structural folder tree as JSON:
    - dict for directories
    - null for files (leaf)
    """
    tree: dict = {}
    for m in metas:
        parts = m.rel_path.split("/")
        cursor = tree
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor.setdefault(parts[-1], None)
    return json.dumps(tree, indent=2)


def build_global_symbol_table(metas: List[FileMeta]) -> str:
    """
    Global symbol table for quick 'definition' lookup.
    Includes:
    - class/function symbols from `symbols`
    - plus method symbols from `code_objects` (kind=method)
    """
    table: List[dict] = []
    for m in metas:
        for s in m.symbols:
            table.append(
                {
                    "symbol": s,
                    "file": m.rel_path,
                    "module": m.module_name,
                    "module_group": m.module_group,
                }
            )

        for obj in m.code_objects:
            # methods were not included in old `symbols` list; add them here
            if obj.kind == "method":
                table.append(
                    {
                        "symbol": obj.symbol,
                        "file": m.rel_path,
                        "module": m.module_name,
                        "module_group": m.module_group,
                    }
                )

    return json.dumps(table, indent=2)


def build_dependency_graph(metas: List[FileMeta]) -> str:
    graph: Dict[str, List[str]] = {}
    for m in metas:
        graph[m.module_name] = m.imports
    return json.dumps(graph, indent=2)


def build_call_graph(metas: List[FileMeta]) -> str:
    """
    module -> {creates,calls}
    """
    graph: Dict[str, Dict[str, List[str]]] = {}
    for m in metas:
        graph[m.module_name] = {
            "creates": m.creates,
            "calls": m.calls,
        }
    return json.dumps(graph, indent=2)


def build_type_usage_graph(metas: List[FileMeta]) -> str:
    """
    module -> type_usages (noisy but useful)
    """
    graph: Dict[str, List[str]] = {}
    for m in metas:
        graph[m.module_name] = m.type_usages
    return json.dumps(graph, indent=2)


# =============================================================================
# STRUCTURE.json (separate file)
# =============================================================================

def build_structure_index(*, bundle_filename: str, bundle_scope: str, project_root: Path, metas: List[FileMeta]) -> dict:
    """
    Structure index is the machine-readable index to avoid raw-text searching.
    Contains:
    - bundle metadata
    - files table
    - global COID registry: coid -> symbol/file/module/kind/lineno/end_lineno
    - per-module indices: creates/calls/type_usages and code_objects list
    """
    coid_index: Dict[str, dict] = {}
    files_index: Dict[str, dict] = {}

    for m in metas:
        files_index[m.module_name] = {
            "file": m.rel_path,
            "module_group": m.module_group,
            "sha256": m.sha256,
            "lines": m.lines,
            "chars": m.chars,
            "imports": m.imports,
            "creates": m.creates,
            "calls": m.calls,
            "type_usages": m.type_usages,
            "code_objects": [
                {
                    "symbol": o.symbol,
                    "kind": o.kind,
                    "coid": o.coid,
                    "lineno": o.lineno,
                    "end_lineno": o.end_lineno,
                }
                for o in m.code_objects
            ],
        }

        for o in m.code_objects:
            coid_index[o.coid] = {
                "symbol": o.symbol,
                "kind": o.kind,
                "file": m.rel_path,
                "module": m.module_name,
                "module_group": m.module_group,
                "lineno": o.lineno,
                "end_lineno": o.end_lineno,
                "file_sha256": m.sha256,
            }

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle": {
            "filename": bundle_filename,
            "scope": bundle_scope,
            "project_root": str(project_root),
        },
        "files": files_index,
        "coid_index": coid_index,
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

These instructions are a strict operating procedure for analyzing and modifying Intergrax code using the attached files.
The bundle is the single source of truth for code content. The structure file is the single source of truth for indexing.

Response language requirement:
- Provide all answers in Polish (PL).
- Keep code comments in English (EN).

## 0) Inputs and their roles (mandatory)
You receive three files:
1) Bundle (code): {bundle_filename}
2) Structure index (machine-readable): {structure_filename}
3) Instructions (this file)

Rules:
- Use the structure index FIRST to locate symbols, call sites, and code object IDs (COID).
- Use the bundle SECOND to read the exact source and confirm signatures.
- Never search by “guessing” paths or symbols.

## 1) Scope Gate (mandatory)
- Before proposing any patch, list ALL files that would be edited/added/removed.
- If any required file is OUTSIDE the attached bundle scope (or not present in the bundle), STOP.
- Request the missing file(s) or a wider-scope bundle. Do not guess.

## 2) Zero-hallucination rule (mandatory)
- Do not invent paths, modules, classes, functions, or signatures.
- If a symbol is not found in the structure index or bundle, explicitly say so and request the missing file or a wider-scope bundle.
- Never rely on memory or generic patterns; always verify against the provided files.

## 3) Verification-first workflow (mandatory)
Before proposing any integration or patch, do a 'Verification Pass' where you list:
- Exact FILE path(s) to edit (from bundle headers or structure index).
- Exact MODULE name(s) (from bundle headers or structure index).
- Existing class/protocol/function signatures copied from the bundle (verbatim, short).
- Relevant COID(s) from the structure index (stable references).
- The minimal delta you will apply.

Only after the Verification Pass, provide the patch.

## 4) Exact Signature Quote (mandatory)
- When quoting a class/protocol/function signature, copy it verbatim from the bundle:
  - include parameter names, defaults, and types as shown in the code
  - do not paraphrase signatures

## 5) How to use the structure index (mandatory)
The structure index contains:
- files[module] with: file path, imports, creates, calls, type_usages, and code_objects
- coid_index[coid] with: symbol, kind, file, module, and optional lineno/end_lineno

Use it for:
- locating "where is X created?" via files[*].creates
- locating "who calls method Y?" via files[*].calls
- locating "where is type Z used?" via files[*].type_usages
- stable code references via COID (coid_index)

Important:
- Prefer COID references in your analysis and patch plan.
- COID is stable across formatting and line changes (AST-based).

## 6) How to navigate the bundle (mandatory)
Each file in the bundle is preceded by a header:
- FILE: <relative path>
- MODULE: <import path>
- MODULE_GROUP: <structural grouping>
- LINES, SHA256
- SYMBOLS (best-effort)
- IMPORTS (best-effort)
- CODE_OBJECTS (COID) (best-effort, python-like only)

Use:
1) Structure index to locate file/module/COID
2) Bundle header to confirm exact file/module and read real code

## 7) Few-shot examples

### Example A: locate where TraceEvent is created
Task: "Where is TraceEvent constructed in runtime?"

Procedure:
1) In {structure_filename}, search files[*].creates for "TraceEvent".
2) List matching modules and their file paths.
3) Pick the most relevant runtime module based on scope.
4) Open that file section in the bundle and confirm the call site context.

Output style (Polish):
- "TraceEvent jest tworzony w module X (plik Y). Potwierdzone w kodzie w sekcji FILE: Y."

### Example B: use COID for stable reference
Task: "Patch TraceEvent.to_dict"

Procedure:
1) Find COID in coid_index where symbol ends with ":TraceEvent.to_dict".
2) Use that COID in your Verification Pass.
3) Open the exact file section in the bundle and apply the minimal patch.

Output style (Polish):
- "Modyfikujemy COID=<...> (symbol=..., plik=...)."

## 8) No-contract-drift rule (mandatory)
- Do not change runtime semantics unless explicitly requested.
- Prefer backward-compatible additions:
  - default values
  - optional fields
  - preserving existing call sites

## 9) Serialization Gate (mandatory)
If the change impacts a persisted/serialized model:
- You must identify ALL code paths that serialize/deserialize it (e.g., SerializedXxx mappers, stores, codecs).
- You must update them in the same patch.
- If you cannot find those paths within the current bundle scope, STOP and request a wider-scope bundle.

## 10) Patch format requirement
When proposing a patch:
- Group by file.
- For each file:
  - show the exact file path
  - show only the changed blocks (not the entire file)
  - explain briefly why the change is necessary

## 11) Reasoned Decision Summary (mandatory, no chain-of-thought disclosure)
- Do NOT provide chain-of-thought or hidden reasoning.
- Provide a short "Reasoned Decision Summary" (2-6 bullet points) explaining the key evidence from the bundle/structure index
  that justifies the change (paths, signatures, COIDs, contracts).

### Important:
 - Do not introduce new dynamic structures or loose dict contracts unless they already exist in the bundle. Prefer existing typed models.
 - Before implementing logic, confirm which module group/file is responsible (runtime / storage / adapters / etc.) based on existing patterns in the bundle.

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
    header.append("# How to navigate this bundle:\n")
    header.append("# - Use the STRUCTURAL MAP, GLOBAL SYMBOL TABLE, and CODE_OBJECTS (COID).\n")
    header.append("# - Prefer the separate STRUCTURE.json for indexing call sites and COID lookup.\n")
    header.append("# - Each original file is included below with a header.\n")
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
            syms, imps = extract_symbols_and_imports_for_file(p, txt, module_name=module_name)
            creates, calls, type_usages, code_objects = extract_extended_python_like_metadata(p, txt, module_name=module_name)
        else:
            syms, imps = ([], [])
            creates, calls, type_usages, code_objects = ([], [], [], [])

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
                creates=creates,
                calls=calls,
                type_usages=type_usages,
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
    Supported: extensions registered in SOURCE_HANDLERS.
    Paths MUST be absolute or project_root-relative (we resolve anyway).

    Also generates:
    - <bundle_stem>_STRUCTURE.json (machine index)
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

    # Global machine-readable sections inside bundle (still useful)
    parts.append("# ======================================================================\n")
    parts.append("# STRUCTURAL MAP (folder tree JSON)\n")
    parts.append("# ======================================================================\n")
    parts.append(build_folder_tree(metas))
    parts.append("\n\n")

    parts.append("# ======================================================================\n")
    parts.append("# GLOBAL SYMBOL TABLE (JSON)\n")
    parts.append("# ======================================================================\n")
    parts.append(build_global_symbol_table(metas))
    parts.append("\n\n")

    parts.append("# ======================================================================\n")
    parts.append("# DEPENDENCY GRAPH (module -> imports, JSON)\n")
    parts.append("# ======================================================================\n")
    parts.append(build_dependency_graph(metas))
    parts.append("\n\n")

    parts.append("# ======================================================================\n")
    parts.append("# CALL GRAPH (module -> {creates,calls}, JSON)\n")
    parts.append("# ======================================================================\n")
    parts.append(build_call_graph(metas))
    parts.append("\n\n")

    parts.append("# ======================================================================\n")
    parts.append("# TYPE USAGE GRAPH (module -> identifiers, JSON)\n")
    parts.append("# ======================================================================\n")
    parts.append(build_type_usage_graph(metas))
    parts.append("\n\n")

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
            header.append("# SYMBOLS (global-unique, best-effort):\n")
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
                    # keep it extremely grep-friendly
                    header.append(f"#   - COID={o.coid} | kind={o.kind} | symbol={o.symbol} | lineno={o.lineno} | end_lineno={o.end_lineno}\n")
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
    Supported files: extensions registered in SOURCE_HANDLERS.
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
    Walk up the directory tree to find project root
    (identified by pyproject.toml).
    """
    current = start
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Project root not found (pyproject.toml missing)")


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
            max_mb=25,
            include_symbols=True,
            write_instructions_file=True,
            write_structure_file=True,
        )


if __name__ == "__main__":
    main()
