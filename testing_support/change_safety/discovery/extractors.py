# © Artur Czarnecki. All rights reserved.

"""D2/D3 discovery: AST fact scan + 19 REL extractors (E-01..E-19).

Pipeline: ``scan_source`` (single AST pass, JSON-serialisable facts, content-addressed cache) ->
extractor registry (``EXTRACTORS``) -> typed edges. Dynamic/unresolvable constructs become
``DEPENDS_UNKNOWN`` edges to ``unknown:`` nodes — never silently dropped.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.discovery.git_paths import git_ls_files

_REPO = Path(__file__).resolve().parents[3]

EXTRACTOR_VERSION = "evol-x-extractors-4"

# Code roots (module nodes). ``tests/`` is scanned only for TESTED_BY / GATED_BY / test ids.
DISCOVERY_ROOTS: tuple[str, ...] = (
    "intergrax/",
    "agents/",
    "applications/",
    "platform_proofs/",
    "testing_support/change_safety/",
)
TEST_ROOT = "tests/"
IN_UNIVERSE_TOPS: frozenset[str] = frozenset(
    {"intergrax", "agents", "applications", "platform_proofs", "testing_support"}
)

MANDATORY_RELS: tuple[str, ...] = (
    "IMPORTS",
    "CONSUMES_CONTRACT",
    "IMPLEMENTS",
    "COMPOSES",
    "REGISTERED_IN",
    "SHARES",
    "PERSISTS",
    "READS_PERSISTED",
    "EMITS_EVENT",
    "HANDLES_EVENT",
    "BINDS",
    "OWNED_BY",
    "TESTED_BY",
    "GATED_BY",
    "DEPENDS_UNKNOWN",
)

_CONTRACT_REF_RE = re.compile(
    r"\b(?:intergrax|agents|applications)(?:\.[A-Za-z_][A-Za-z0-9_]*)*?\.contracts(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
)
_DOTTED_STR_RE = re.compile(
    r"^(?:intergrax|agents|applications|platform_proofs|testing_support)(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)
_TRIPLE_RE = re.compile(r'("""|\'\'\')(.*?)\1', re.S)
_PERSIST_WRITE = frozenset(
    {"write_text", "write_bytes", "dump", "save", "persist", "commit", "executemany", "put_item"}
)
_PERSIST_READ = frozenset({"read_text", "read_bytes", "load", "loads", "fetchall", "fetchone", "get_item"})
_PERSIST_HINTS = ("store", "persist", "repository", "sqlite", "checkpoint", "snapshot")
_EMIT_NAMES = frozenset({"emit", "publish", "dispatch", "emit_event", "publish_event"})


# --------------------------------------------------------------------------- nodes / edges


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    category: str
    path: str = ""


@dataclass(frozen=True)
class GraphEdge:
    source_id: str
    target_id: str
    rel: str


def module_id_for_path(path: str) -> str:
    rel = path.replace("\\", "/")
    if not rel.endswith(".py"):
        return f"path:{rel}"
    body = rel[:-3]
    if body.endswith("/__init__"):
        body = body[: -len("/__init__")]
    return "module:" + body.replace("/", ".")


def _module_id(path: str) -> str:  # backwards-compatible alias
    return module_id_for_path(path)


def suite_id_for_path(path: str) -> str:
    return f"suite:{path.replace(chr(92), '/')}"


# --------------------------------------------------------------------------- AST fact scan


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    ids.add(id(body[0].value))
    return ids


def strip_comments_and_docstrings_text(source: str) -> str:
    """Second-method text normaliser (deliberately not AST based)."""
    without_triple = _TRIPLE_RE.sub('""', source)
    lines = []
    for line in without_triple.splitlines():
        if "#" in line:
            line = line.split("#", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def scan_source(source: str, *, filename: str = "<memory>") -> dict[str, Any]:
    facts: dict[str, Any] = {
        "syntax_error": False,
        "imports": [],
        "dyn_literal": [],
        "dyn_unknown": [],
        "symbols": [],
        "bases": [],
        "registers": [],
        "emits": [],
        "handles": [],
        "persist_write": 0,
        "persist_read": 0,
        "config_literals": [],
        "env_vars": [],
        "markers": [],
        "str_modules": [],
        "str_contract_refs": [],
        "test_ids": [],
        "text_contract_refs": sorted(set(_CONTRACT_REF_RE.findall(strip_comments_and_docstrings_text(source)))),
    }
    try:
        tree = ast.parse(source, filename=filename)
    except (SyntaxError, ValueError):
        facts["syntax_error"] = True
        return facts
    docs = _docstring_nodes(tree)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                facts["imports"].append([alias.name, 0, [], node.lineno])
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names = [a.name for a in node.names]
            facts["imports"].append([mod, node.level, names, node.lineno])
            for a in node.names:
                aliases[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
        elif isinstance(node, ast.Call):
            callee = _dotted(node.func)
            tail = callee.rsplit(".", 1)[-1]
            if callee in {"importlib.import_module", "import_module", "__import__"}:
                arg = node.args[0] if node.args else None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    facts["dyn_literal"].append(arg.value)
                else:
                    facts["dyn_unknown"].append(node.lineno)
            elif callee in {"eval", "exec"}:
                facts["dyn_unknown"].append(node.lineno)
            if tail.startswith("register"):
                args = [_dotted(a) for a in node.args if _dotted(a)]
                facts["registers"].append([callee, args])
            if tail in _EMIT_NAMES:
                arg = node.args[0] if node.args else None
                name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else ""
                facts["emits"].append(name)
            if tail in _PERSIST_WRITE or (callee == "open" and _open_is_write(node)):
                facts["persist_write"] += 1
            elif tail in _PERSIST_READ or callee == "open":
                facts["persist_read"] += 1
            if callee in {"os.getenv", "os.environ.get", "getenv"} and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    facts["env_vars"].append(a0.value)
            if callee.endswith("pytest.mark.gate") or callee == "pytest.mark.gate":
                facts["markers"].append("gate")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            val = node.value
            if len(val) < 200_000 and "contracts" in val:
                facts["str_contract_refs"].extend(_CONTRACT_REF_RE.findall(val))
            if len(val) < 200 and _DOTTED_STR_RE.match(val):
                facts["str_modules"].append(val)
            elif len(val) < 200 and val.endswith((".json", ".yaml", ".yml", ".toml")) and "\n" not in val:
                facts["config_literals"].append(val)
        elif isinstance(node, ast.Attribute):
            d = _dotted(node)
            if d == "pytest.mark.gate":
                facts["markers"].append("gate")
        elif isinstance(node, ast.Subscript):
            if _dotted(node.value) == "os.environ":
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    facts["env_vars"].append(sl.value)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            facts["symbols"].append(["class", node.name])
            for base in node.bases:
                d = _dotted(base)
                if d:
                    facts["bases"].append([node.name, d, aliases.get(d.split(".")[0], "")])
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _collect_handlers(item, facts)
                    if node.name.startswith("Test") and item.name.startswith("test"):
                        facts["test_ids"].append(f"{node.name}::{item.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            facts["symbols"].append(["function", node.name])
            _collect_handlers(node, facts)
            if node.name.startswith("test"):
                facts["test_ids"].append(node.name)
    facts["imports"].sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    for key in (
        "dyn_literal",
        "env_vars",
        "config_literals",
        "str_modules",
        "str_contract_refs",
        "markers",
        "emits",
        "handles",
    ):
        facts[key] = sorted(set(facts[key]))
    facts["dyn_unknown"] = sorted(set(facts["dyn_unknown"]))
    return facts


def _open_is_write(node: ast.Call) -> bool:
    mode: Any = None
    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        mode = node.args[1].value
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            mode = kw.value.value
    return isinstance(mode, str) and any(c in mode for c in "wax+")


def _collect_handlers(fn: ast.AST, facts: dict[str, Any]) -> None:
    name = getattr(fn, "name", "")
    decorators = [_dotted(d.func if isinstance(d, ast.Call) else d) for d in getattr(fn, "decorator_list", [])]
    if name.startswith(("handle_", "on_")) or any(
        d.rsplit(".", 1)[-1] in {"on_event", "subscribe", "handler"} for d in decorators
    ):
        facts["handles"].append(name)


# --------------------------------------------------------------------------- content-addressed cache (D9, non-evidence)

_FACT_CACHE: dict[str, dict[str, Any]] | None = None


def _cache_path() -> Path:
    return _REPO / ".tmp" / "session" / "evol-x-full-plan" / "facts_cache.v2.json"


def _load_cache() -> dict[str, dict[str, Any]]:
    global _FACT_CACHE
    if _FACT_CACHE is None:
        _FACT_CACHE = {}
        path = _cache_path()
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("extractor_version") == EXTRACTOR_VERSION:
                    _FACT_CACHE = payload["facts"]
            except (OSError, ValueError, KeyError):
                _FACT_CACHE = {}
    return _FACT_CACHE


def flush_fact_cache() -> None:
    cache = _load_cache()
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"extractor_version": EXTRACTOR_VERSION, "facts": cache}, separators=(",", ":")),
        encoding="utf-8",
    )


def reset_fact_cache(*, drop_disk: bool = False) -> None:
    global _FACT_CACHE
    _FACT_CACHE = {}
    if drop_disk and _cache_path().is_file():
        _cache_path().unlink()


@dataclass(frozen=True)
class TrackedFile:
    path: str
    blob_sha: str


def git_ls_files_with_blobs(prefixes: tuple[str, ...]) -> tuple[TrackedFile, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-s", *[p.rstrip("/") for p in prefixes]],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    out: list[TrackedFile] = []
    for line in result.stdout.splitlines():
        meta, _, rel = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 2 and rel.endswith(".py"):
            out.append(TrackedFile(rel.strip().replace("\\", "/"), parts[1]))
    return tuple(sorted(out, key=lambda t: t.path))


def facts_for_file(tracked: TrackedFile, *, use_cache: bool) -> dict[str, Any]:
    cache = _load_cache() if use_cache else None
    key = tracked.blob_sha
    if cache is not None and key in cache:
        return cache[key]
    try:
        source = (_REPO / tracked.path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        facts = scan_source("", filename=tracked.path)
        facts["syntax_error"] = True
    else:
        facts = scan_source(source, filename=tracked.path)
    if cache is not None:
        cache[key] = facts
    return facts


# --------------------------------------------------------------------------- import resolution


def _package_of(path: str) -> list[str]:
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def resolve_import_targets(path: str, record: list[Any], known_modules: frozenset[str]) -> tuple[list[str], list[str]]:
    """Return (resolved module ids, unresolved in-universe dotted names)."""
    module, level, names, _lineno = record
    if level:
        pkg = _package_of(path)
        base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
        dotted = ".".join(base + ([module] if module else []))
    else:
        dotted = module
    if not dotted:
        return [], []
    top = dotted.split(".")[0]
    if top not in IN_UNIVERSE_TOPS:
        return [], []
    if top == "testing_support" and not (
        dotted == "testing_support.change_safety" or dotted.startswith("testing_support.change_safety.")
    ):
        return [], []
    candidates = [dotted] + [f"{dotted}.{n}" for n in names if n != "*"]
    resolved: list[str] = []
    unresolved: list[str] = []
    for cand in candidates:
        parts = cand.split(".")
        hit = None
        for i in range(len(parts), 0, -1):
            mid = "module:" + ".".join(parts[:i])
            if mid in known_modules:
                hit = mid
                break
        if hit is None:
            if cand == dotted:
                unresolved.append(cand)
        elif hit != "module:" + cand and cand != dotted:
            # ``from pkg import name`` where name is a symbol, not a submodule: edge is on pkg only
            continue
        else:
            resolved.append(hit)
    if not resolved and dotted not in unresolved:
        unresolved.append(dotted)
    return sorted(set(resolved)), sorted(set(unresolved))


# --------------------------------------------------------------------------- extractor registry (E-01..E-19)


@dataclass(frozen=True)
class ExtractionContext:
    path: str
    module_id: str
    facts: dict[str, Any]
    known_modules: frozenset[str]
    known_suites: frozenset[str]
    owner_for_path: Callable[[str], tuple[str, ...]]
    is_test: bool
    resolved: tuple[tuple[list[Any], list[str], list[str]], ...] = ()


@dataclass(frozen=True)
class ExtractionResult:
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    unknowns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Extractor:
    extractor_id: str
    rel: str
    mandatory: bool
    fn: Callable[[ExtractionContext], ExtractionResult]


def _is_contract_module(module_id: str) -> bool:
    return (
        module_id.startswith("module:intergrax.contracts")
        or ".contracts." in module_id
        or module_id.endswith(".contracts")
    )


def _e01_imports(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    unknowns: list[str] = []
    for rec, targets, unresolved in ctx.resolved:
        if rec[1]:
            continue
        edges.extend(GraphEdge(ctx.module_id, t, "IMPORTS") for t in targets if t != ctx.module_id)
        unknowns.extend(unresolved)
    return ExtractionResult(edges=tuple(edges), unknowns=tuple(unknowns))


def _e02_relative(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    unknowns: list[str] = []
    for rec, targets, unresolved in ctx.resolved:
        if not rec[1]:
            continue
        edges.extend(GraphEdge(ctx.module_id, t, "IMPORTS") for t in targets if t != ctx.module_id)
        unknowns.extend(unresolved)
    return ExtractionResult(edges=tuple(edges), unknowns=tuple(unknowns))


def _e03_contract(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    for _rec, targets, _unresolved in ctx.resolved:
        edges.extend(
            GraphEdge(ctx.module_id, t, "CONSUMES_CONTRACT")
            for t in targets
            if _is_contract_module(t) and t != ctx.module_id
        )
    return ExtractionResult(edges=tuple(edges))


def _e04_symbols(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    nodes = [
        GraphNode(f"symbol:{ctx.module_id[len('module:') :]}.{name}", f"symbol_{kind}", ctx.path)
        for kind, name in ctx.facts["symbols"]
    ]
    edges = [GraphEdge(ctx.module_id, n.node_id, "DEFINES") for n in nodes]
    return ExtractionResult(nodes=tuple(nodes), edges=tuple(edges))


def _e05_implements(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    for _cls, _dotted_base, origin in ctx.facts["bases"]:
        if not origin:
            continue
        parts = origin.split(".")
        for i in range(len(parts), 0, -1):
            mid = "module:" + ".".join(parts[:i])
            if mid in ctx.known_modules:
                if mid != ctx.module_id:
                    edges.append(GraphEdge(ctx.module_id, mid, "IMPLEMENTS"))
                break
    return ExtractionResult(edges=tuple(edges))


def _e06_registered(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: set[GraphEdge] = set()
    for callee, args in ctx.facts["registers"]:
        tops = {callee.split(".")[0], *(a.split(".")[0] for a in args)}
        for rec, targets, _unresolved in ctx.resolved:
            if tops & set(rec[2]) or rec[0].split(".")[-1] in tops:
                edges.update(GraphEdge(ctx.module_id, t, "REGISTERED_IN") for t in targets if t != ctx.module_id)
    return ExtractionResult(edges=tuple(sorted(edges, key=lambda e: (e.target_id,))))


def _e07_composes(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not any(h in ctx.path for h in ("composition", "compose", "bootstrap", "wiring")):
        return ExtractionResult()
    edges = [e for e in _e01_imports(ctx).edges]
    return ExtractionResult(edges=tuple(GraphEdge(e.source_id, e.target_id, "COMPOSES") for e in edges))


def _e08_shares(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges = [
        GraphEdge(e.source_id, e.target_id, "SHARES")
        for e in _e01_imports(ctx).edges
        if {"shared", "_shared", "common"} & set(e.target_id[len("module:") :].split("."))
    ]
    return ExtractionResult(edges=tuple(edges))


def _persist_targets(ctx: ExtractionContext) -> list[str]:
    out = []
    for e in _e01_imports(ctx).edges:
        if any(h in e.target_id.rsplit(".", 1)[-1] for h in _PERSIST_HINTS):
            out.append(e.target_id)
    return out


def _e09_persists(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not ctx.facts["persist_write"]:
        return ExtractionResult()
    targets = _persist_targets(ctx)
    return ExtractionResult(
        edges=tuple(GraphEdge(ctx.module_id, f"store:{t[len('module:') :]}", "PERSISTS") for t in targets),
        nodes=tuple(GraphNode(f"store:{t[len('module:') :]}", "store") for t in targets),
    )


def _e10_reads(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not ctx.facts["persist_read"]:
        return ExtractionResult()
    targets = _persist_targets(ctx)
    return ExtractionResult(
        edges=tuple(GraphEdge(ctx.module_id, f"store:{t[len('module:') :]}", "READS_PERSISTED") for t in targets),
        nodes=tuple(GraphNode(f"store:{t[len('module:') :]}", "store") for t in targets),
    )


def _e11_emits(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not ctx.facts["emits"]:
        return ExtractionResult()
    names = [n or f"anonymous:{ctx.module_id[len('module:') :]}" for n in ctx.facts["emits"]]
    return ExtractionResult(
        nodes=tuple(GraphNode(f"event:{n}", "event") for n in names),
        edges=tuple(GraphEdge(ctx.module_id, f"event:{n}", "EMITS_EVENT") for n in names),
    )


def _e12_handles(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not ctx.facts["handles"]:
        return ExtractionResult()
    names = [f"{ctx.module_id[len('module:') :]}.{h}" for h in ctx.facts["handles"]]
    return ExtractionResult(
        nodes=tuple(GraphNode(f"event:{n}", "event") for n in names),
        edges=tuple(GraphEdge(ctx.module_id, f"event:{n}", "HANDLES_EVENT") for n in names),
    )


def _e13_binds(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    cfg = [f"config:{c}" for c in ctx.facts["config_literals"]] + [f"config:env:{v}" for v in ctx.facts["env_vars"]]
    return ExtractionResult(
        nodes=tuple(GraphNode(c, "config") for c in cfg), edges=tuple(GraphEdge(ctx.module_id, c, "BINDS") for c in cfg)
    )


def _e14_owned_by(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    owners = ctx.owner_for_path(ctx.path)
    return ExtractionResult(edges=tuple(GraphEdge(ctx.module_id, f"owner:{o}", "OWNED_BY") for o in owners))


def _e15_tested_by(ctx: ExtractionContext) -> ExtractionResult:
    if not ctx.is_test:
        return ExtractionResult()
    suite = suite_id_for_path(ctx.path)
    edges: list[GraphEdge] = []
    for _rec, targets, _unresolved in ctx.resolved:
        edges.extend(GraphEdge(t, suite, "TESTED_BY") for t in targets)
    return ExtractionResult(
        nodes=(GraphNode(suite, "suite", ctx.path),), edges=tuple(sorted(set(edges), key=lambda e: (e.source_id,)))
    )


def _e16_gated_by(ctx: ExtractionContext) -> ExtractionResult:
    if not ctx.is_test or "gate" not in ctx.facts["markers"]:
        return ExtractionResult()
    return ExtractionResult(
        nodes=(GraphNode("ci:gate", "ci"),), edges=(GraphEdge(suite_id_for_path(ctx.path), "ci:gate", "GATED_BY"),)
    )


def _e17_dynamic_unknown(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test or not ctx.facts["dyn_unknown"]:
        return ExtractionResult()
    ids = [f"unknown:dynamic:{ctx.path}:{ln}" for ln in ctx.facts["dyn_unknown"]]
    return ExtractionResult(
        nodes=tuple(GraphNode(i, "unknown", ctx.path) for i in ids),
        edges=tuple(GraphEdge(ctx.module_id, i, "DEPENDS_UNKNOWN") for i in ids),
        unknowns=tuple(ids),
    )


def _e18_dynamic_literal(ctx: ExtractionContext) -> ExtractionResult:
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    for dotted in ctx.facts["dyn_literal"]:
        targets, _ = resolve_import_targets(ctx.path, [dotted, 0, [], 0], ctx.known_modules)
        edges.extend(GraphEdge(ctx.module_id, t, "IMPORTS") for t in targets if t != ctx.module_id)
    return ExtractionResult(edges=tuple(edges))


def _e19_string_refs(ctx: ExtractionContext) -> ExtractionResult:
    """String-literal module references (registry-by-name) — BINDS edges; also app/agent composition nodes."""
    if ctx.is_test:
        return ExtractionResult()
    edges: list[GraphEdge] = []
    for dotted in [*ctx.facts["str_modules"], *ctx.facts["str_contract_refs"]]:
        targets, _ = resolve_import_targets(ctx.path, [dotted, 0, [], 0], ctx.known_modules)
        edges.extend(GraphEdge(ctx.module_id, t, "BINDS") for t in targets if t != ctx.module_id)
    nodes: list[GraphNode] = []
    parts = ctx.path.split("/")
    if parts[0] in {"applications", "agents"} and len(parts) > 2:
        kind = "app" if parts[0] == "applications" else "agent"
        aid = f"{kind}:{parts[1]}"
        nodes.append(GraphNode(aid, kind))
        edges.append(GraphEdge(aid, ctx.module_id, "COMPOSES"))
    return ExtractionResult(nodes=tuple(nodes), edges=tuple(edges))


EXTRACTORS: tuple[Extractor, ...] = (
    Extractor("E-01", "IMPORTS", True, _e01_imports),
    Extractor("E-02", "IMPORTS", True, _e02_relative),
    Extractor("E-03", "CONSUMES_CONTRACT", True, _e03_contract),
    Extractor("E-04", "DEFINES", True, _e04_symbols),
    Extractor("E-05", "IMPLEMENTS", True, _e05_implements),
    Extractor("E-06", "REGISTERED_IN", False, _e06_registered),
    Extractor("E-07", "COMPOSES", False, _e07_composes),
    Extractor("E-08", "SHARES", False, _e08_shares),
    Extractor("E-09", "PERSISTS", False, _e09_persists),
    Extractor("E-10", "READS_PERSISTED", False, _e10_reads),
    Extractor("E-11", "EMITS_EVENT", False, _e11_emits),
    Extractor("E-12", "HANDLES_EVENT", False, _e12_handles),
    Extractor("E-13", "BINDS", True, _e13_binds),
    Extractor("E-14", "OWNED_BY", True, _e14_owned_by),
    Extractor("E-15", "TESTED_BY", True, _e15_tested_by),
    Extractor("E-16", "GATED_BY", False, _e16_gated_by),
    Extractor("E-17", "DEPENDS_UNKNOWN", True, _e17_dynamic_unknown),
    Extractor("E-18", "IMPORTS", True, _e18_dynamic_literal),
    Extractor("E-19", "BINDS", False, _e19_string_refs),
)


def extractor_manifest() -> dict[str, Any]:
    rows = [{"id": e.extractor_id, "rel": e.rel, "mandatory": e.mandatory} for e in EXTRACTORS]
    return {"extractor_version": EXTRACTOR_VERSION, "extractors": rows, "digest": stable_digest_hex(rows)}


def covered_mandatory_rels() -> frozenset[str]:
    return frozenset(e.rel for e in EXTRACTORS if e.mandatory) | frozenset(e.rel for e in EXTRACTORS)


# --------------------------------------------------------------------------- legacy convenience API


def build_inventory_nodes() -> tuple[GraphNode, ...]:
    nodes: list[GraphNode] = []
    for root in DISCOVERY_ROOTS:
        for rel in git_ls_files(root.rstrip("/")):
            if rel.endswith(".py"):
                nodes.append(GraphNode(module_id_for_path(rel), "module", rel))
    return tuple(sorted(set(nodes), key=lambda n: n.node_id))


def build_import_edges() -> tuple[GraphEdge, ...]:
    from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory

    inv = build_closed_world_inventory(source_commit_sha="")
    return tuple(e for e in inv.edges if e.rel == "IMPORTS")
