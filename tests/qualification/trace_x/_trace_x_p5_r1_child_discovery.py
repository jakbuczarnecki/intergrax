# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q7: lexical binding authority + Q6 usage-context closure."""

from __future__ import annotations

import ast
import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from tests.qualification.trace_x._trace_x_p5_discovery import repo_root

_INTERGRAX_SCAN_ROOT: Final[Path] = repo_root() / "intergrax"
_EXCLUDE_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {"__pycache__", "tests", "docs", "examples", "benchmarks", "model_runtime_proof", "proofs", "legacy"},
)

CANONICAL_CHILD_MODULE: Final[str] = "intergrax.runtime.execution.child"
CANONICAL_CHILD_CLASS: Final[str] = "ChildExecutionRunner"
EXECUTION_PACKAGE_MODULE: Final[str] = "intergrax.runtime.execution"


class CanonicalConstructorUsageKind(enum.Enum):
    DIRECT_CONSTRUCTOR_CALL = "DIRECT_CONSTRUCTOR_CALL"
    TYPE_ANNOTATION = "TYPE_ANNOTATION"
    IMPORT_BINDING = "IMPORT_BINDING"
    FORBIDDEN_RUNTIME_ESCAPE = "FORBIDDEN_RUNTIME_ESCAPE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CanonicalConstructorUsageRecord:
    rel_path: str
    enclosing: str
    lineno: int
    kind: CanonicalConstructorUsageKind


def _intergrax_path_excluded(rel_path: Path) -> bool:
    parts = rel_path.parts
    if _EXCLUDE_DIR_NAMES.intersection(parts):
        return True
    if "docker" in parts and "runtime-context" in parts:
        return True
    return False


def _iter_intergrax_production_py_files() -> list[Path]:
    paths: list[Path] = []
    if not _INTERGRAX_SCAN_ROOT.is_dir():
        return paths
    for py_path in _INTERGRAX_SCAN_ROOT.rglob("*.py"):
        rel = py_path.relative_to(repo_root())
        if _intergrax_path_excluded(rel):
            continue
        paths.append(py_path)
    return paths


def _surface_key(rel_path: str, enclosing: str) -> str:
    normalized = rel_path.replace("\\", "/")
    return f"{normalized}::{enclosing}"


def _enclosing_symbol(class_stack: list[str], function_stack: list[str]) -> str:
    if function_stack:
        fn = function_stack[-1]
        if class_stack:
            return f"{class_stack[-1]}.{fn}"
        return fn
    if class_stack:
        return class_stack[-1]
    return "<module>"


def _call_callee_root(node: ast.Call) -> ast.expr:
    func = node.func
    if isinstance(func, ast.Subscript):
        return func.value
    return func


def _expr_contains(container: ast.AST, target: ast.expr) -> bool:
    for node in ast.walk(container):
        if node is target:
            return True
    return False


@dataclass
class _ImportProvenance:
    """Lexical-scope bindings for canonical child constructor resolution."""

    constructor_names: set[str] = field(default_factory=set)
    child_module_aliases: set[str] = field(default_factory=set)
    intergrax_roots: set[str] = field(default_factory=set)
    shadowed_constructor_names: set[str] = field(default_factory=set)
    local_child_execution_runner_class: bool = False

    def copy(self) -> _ImportProvenance:
        return _ImportProvenance(
            constructor_names=set(self.constructor_names),
            child_module_aliases=set(self.child_module_aliases),
            intergrax_roots=set(self.intergrax_roots),
            shadowed_constructor_names=set(self.shadowed_constructor_names),
            local_child_execution_runner_class=self.local_child_execution_runner_class,
        )


def _apply_import_from(node: ast.ImportFrom, prov: _ImportProvenance) -> None:
    if node.module == CANONICAL_CHILD_MODULE and node.level == 0:
        for alias in node.names:
            if alias.name != CANONICAL_CHILD_CLASS:
                continue
            bound = alias.asname or alias.name
            prov.constructor_names.add(bound)
            prov.shadowed_constructor_names.discard(bound)
        return
    if node.module == EXECUTION_PACKAGE_MODULE and node.level == 0:
        for alias in node.names:
            if alias.name != "child":
                continue
            bound = alias.asname or alias.name
            prov.child_module_aliases.add(bound)


def _apply_import(node: ast.Import, prov: _ImportProvenance) -> None:
    for alias in node.names:
        if alias.name == CANONICAL_CHILD_MODULE:
            if alias.asname:
                prov.child_module_aliases.add(alias.asname)
            else:
                prov.intergrax_roots.add("intergrax")
            continue
        if alias.name == "intergrax" or alias.name.startswith("intergrax."):
            bound = alias.asname or "intergrax"
            prov.intergrax_roots.add(bound)


def _import_from_binds_canonical_child(node: ast.ImportFrom) -> bool:
    if node.module == CANONICAL_CHILD_MODULE and node.level == 0:
        return any(alias.name == CANONICAL_CHILD_CLASS for alias in node.names)
    if node.module == EXECUTION_PACKAGE_MODULE and node.level == 0:
        return any(alias.name == "child" for alias in node.names)
    return False


def _import_binds_canonical_child(node: ast.Import) -> bool:
    return any(alias.name == CANONICAL_CHILD_MODULE for alias in node.names)


def _dotted_expr_canonical_constructor(expr: ast.expr, prov: _ImportProvenance) -> bool:
    parts: list[str] = []
    cur: ast.expr = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return False
    parts.append(cur.id)
    parts.reverse()
    if len(parts) < 2 or parts[-1] != CANONICAL_CHILD_CLASS:
        return False
    module_path = ".".join(parts[:-1])
    if module_path != CANONICAL_CHILD_MODULE:
        return False
    root = parts[0]
    return root in prov.intergrax_roots


def _collect_module_import_provenance(tree: ast.Module) -> _ImportProvenance:
    prov = _ImportProvenance()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            _apply_import_from(node, prov)
        elif isinstance(node, ast.Import):
            _apply_import(node, prov)
        elif isinstance(node, ast.ClassDef) and node.name == CANONICAL_CHILD_CLASS:
            prov.local_child_execution_runner_class = True
            prov.constructor_names.discard(CANONICAL_CHILD_CLASS)
    return prov


def _name_refers_to_canonical_constructor(name: str, prov: _ImportProvenance) -> bool:
    if prov.local_child_execution_runner_class and name == CANONICAL_CHILD_CLASS:
        return False
    if name in prov.shadowed_constructor_names:
        return False
    return name in prov.constructor_names


def _attribute_is_canonical_constructor(value: ast.expr, attr: str, prov: _ImportProvenance) -> bool:
    if attr != CANONICAL_CHILD_CLASS:
        return False
    if isinstance(value, ast.Name) and value.id in prov.child_module_aliases:
        return True
    return False


def _canonical_constructor_reference_root(expr: ast.expr) -> ast.expr:
    if isinstance(expr, ast.Subscript):
        return _canonical_constructor_reference_root(expr.value)
    return expr


def _expr_is_canonical_constructor_ref(expr: ast.expr, prov: _ImportProvenance) -> bool:
    """Authoritative canonical ChildExecutionRunner reference (call sites and alias-escape)."""
    root = _canonical_constructor_reference_root(expr)
    if isinstance(root, ast.Name):
        return _name_refers_to_canonical_constructor(root.id, prov)
    if isinstance(root, ast.Attribute):
        if _dotted_expr_canonical_constructor(root, prov):
            return True
        return _attribute_is_canonical_constructor(root.value, root.attr, prov)
    return False


def _is_canonical_constructor_call(node: ast.Call, prov: _ImportProvenance) -> bool:
    return _expr_is_canonical_constructor_ref(_call_callee_root(node), prov)


def _is_direct_constructor_callee_expr(
    expr: ast.expr,
    parent_stack: tuple[ast.AST, ...],
    prov: _ImportProvenance,
) -> bool:
    for ancestor in reversed(parent_stack):
        if isinstance(ancestor, ast.Call):
            callee = ancestor.func
            if _expr_is_canonical_constructor_ref(_call_callee_root(ancestor), prov):
                return callee is not None and _expr_contains(callee, expr)
            return False
    return False


def classify_canonical_constructor_usage(
    expr: ast.expr,
    parent: ast.AST | None,
    *,
    in_annotation: bool,
    parent_stack: tuple[ast.AST, ...],
    prov: _ImportProvenance,
) -> CanonicalConstructorUsageKind:
    """Single usage-context authority for every canonical constructor reference."""
    if not _expr_is_canonical_constructor_ref(expr, prov):
        raise ValueError("classify_canonical_constructor_usage requires a canonical reference expression")

    if in_annotation:
        return CanonicalConstructorUsageKind.TYPE_ANNOTATION

    if _is_direct_constructor_callee_expr(expr, parent_stack, prov):
        return CanonicalConstructorUsageKind.DIRECT_CONSTRUCTOR_CALL

    if parent is None:
        return CanonicalConstructorUsageKind.UNKNOWN

    if isinstance(
        parent,
        (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.NamedExpr,
            ast.Return,
            ast.Yield,
            ast.YieldFrom,
            ast.arguments,
            ast.ClassDef,
            ast.List,
            ast.Tuple,
            ast.Set,
            ast.Dict,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.comprehension,
            ast.IfExp,
            ast.BoolOp,
            ast.UnaryOp,
            ast.BinOp,
            ast.Compare,
            ast.keyword,
            ast.Starred,
            ast.withitem,
            ast.Match,
            ast.Expr,
            ast.If,
            ast.For,
            ast.While,
            ast.Assert,
            ast.Raise,
            ast.Delete,
            ast.FormattedValue,
        ),
    ):
        return CanonicalConstructorUsageKind.FORBIDDEN_RUNTIME_ESCAPE

    if isinstance(parent, ast.Subscript):
        return CanonicalConstructorUsageKind.FORBIDDEN_RUNTIME_ESCAPE

    if isinstance(parent, ast.Call):
        return CanonicalConstructorUsageKind.FORBIDDEN_RUNTIME_ESCAPE

    if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = parent.args
        if expr in args.defaults or expr in args.kw_defaults:
            return CanonicalConstructorUsageKind.FORBIDDEN_RUNTIME_ESCAPE

    if isinstance(parent, ast.ExceptHandler):
        return CanonicalConstructorUsageKind.UNKNOWN

    return CanonicalConstructorUsageKind.UNKNOWN


class LexicalBindingFormError(ValueError):
    """Unsupported binding-site AST shape — qualification fails closed."""


def lexical_bound_names(node: ast.AST) -> set[str]:
    """Single authority for Python lexical binders (parameters, imports, defs, …)."""
    if isinstance(node, ast.arguments):
        names: set[str] = set()
        for arg in (
            *node.posonlyargs,
            *node.args,
            *node.kwonlyargs,
        ):
            names.add(arg.arg)
        if node.vararg is not None:
            names.add(node.vararg.arg)
        if node.kwarg is not None:
            names.add(node.kwarg.arg)
        return names
    if isinstance(node, ast.Import):
        bound: set[str] = set()
        for alias in node.names:
            bound.add(alias.asname or alias.name.split(".")[0])
        return bound
    if isinstance(node, ast.ImportFrom):
        bound = set()
        for alias in node.names:
            bound.add(alias.asname or alias.name)
        return bound
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.ExceptHandler) and node.name is not None:
        return {node.name}
    if isinstance(node, ast.withitem) and node.optional_vars is not None:
        return lexical_bound_names_from_target(node.optional_vars)
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return lexical_bound_names_from_target(node.target)
    if isinstance(node, ast.NamedExpr):
        return lexical_bound_names_from_target(node.target)
    raise LexicalBindingFormError(f"unsupported lexical binding site: {type(node).__name__}")


def lexical_bound_names_from_target(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for elt in target.elts:
            names |= lexical_bound_names_from_target(elt)
        return names
    if isinstance(target, ast.Starred):
        return lexical_bound_names_from_target(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return set()
    raise LexicalBindingFormError(f"unsupported binding target: {type(target).__name__}")


def lexical_bound_names_from_match_pattern(pattern: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(pattern, ast.MatchAs):
        if pattern.name is not None:
            names.add(pattern.name)
        if pattern.pattern is not None:
            names |= lexical_bound_names_from_match_pattern(pattern.pattern)
        return names
    if isinstance(pattern, ast.MatchStar):
        if pattern.name is not None:
            names.add(pattern.name)
        return names
    if isinstance(pattern, ast.MatchSequence):
        for sub in pattern.patterns:
            names |= lexical_bound_names_from_match_pattern(sub)
        return names
    if isinstance(pattern, ast.MatchMapping):
        for key in pattern.keys:
            if isinstance(key, ast.MatchAs) or isinstance(key, ast.MatchStar):
                names |= lexical_bound_names_from_match_pattern(key)
        if pattern.rest is not None:
            names.add(pattern.rest)
        return names
    if isinstance(pattern, ast.MatchClass):
        for sub in pattern.patterns:
            names |= lexical_bound_names_from_match_pattern(sub)
        for name in pattern.kwd_attrs:
            names.add(name)
        if pattern.kwd_patterns:
            for sub in pattern.kwd_patterns:
                names |= lexical_bound_names_from_match_pattern(sub)
        return names
    if isinstance(pattern, (ast.MatchValue, ast.MatchSingleton)):
        return set()
    raise LexicalBindingFormError(f"unsupported match pattern: {type(pattern).__name__}")


def _iter_assignment_target_names(target: ast.expr) -> list[str]:
    return list(lexical_bound_names_from_target(target))


def _parallel_unpack_rebind_names(
    target: ast.expr,
    value: ast.expr,
    prov: _ImportProvenance,
) -> list[str]:
    if isinstance(target, ast.Starred):
        return []
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        names: list[str] = []
        for target_elt, value_elt in zip(target.elts, value.elts, strict=False):
            names.extend(_parallel_unpack_rebind_names(target_elt, value_elt, prov))
        return names
    if isinstance(target, ast.Name) and _expr_is_canonical_constructor_ref(value, prov):
        return [target.id]
    return []


def _assignment_targets_canonical_rebind(
    targets: list[ast.expr],
    value: ast.expr,
    prov: _ImportProvenance,
) -> list[str]:
    violations: list[str] = []
    if _expr_is_canonical_constructor_ref(value, prov):
        for target in targets:
            violations.extend(_iter_assignment_target_names(target))
        return violations
    for target in targets:
        violations.extend(_parallel_unpack_rebind_names(target, value, prov))
    return violations


@dataclass
class ChildRunnerDiscoveryResult:
    surfaces: set[str] = field(default_factory=set)
    rebind_violations: list[str] = field(default_factory=list)
    alias_escape_violations: list[str] = field(default_factory=list)
    class_body_import_violations: list[str] = field(default_factory=list)
    unknown_usage_violations: list[str] = field(default_factory=list)
    lexical_binding_violations: list[str] = field(default_factory=list)
    usage_records: list[CanonicalConstructorUsageRecord] = field(default_factory=list)


class _ChildRunnerDiscoveryVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str, module_prov: _ImportProvenance) -> None:
        self._rel_path = rel_path
        self._class_stack: list[str] = []
        self._function_stack: list[str] = []
        self._scope_stack: list[_ImportProvenance] = [module_prov.copy()]
        self._parent_stack: list[ast.AST] = []
        self._annotation_depth: int = 0
        self.result = ChildRunnerDiscoveryResult()

    def _current_prov(self) -> _ImportProvenance:
        return self._scope_stack[-1]

    def _parent_tuple(self) -> tuple[ast.AST, ...]:
        return tuple(self._parent_stack)

    def _in_annotation(self) -> bool:
        return self._annotation_depth > 0

    def _record_usage(self, expr: ast.expr, kind: CanonicalConstructorUsageKind) -> None:
        enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
        record = CanonicalConstructorUsageRecord(
            rel_path=self._rel_path,
            enclosing=enclosing,
            lineno=getattr(expr, "lineno", 0),
            kind=kind,
        )
        self.result.usage_records.append(record)
        if kind == CanonicalConstructorUsageKind.FORBIDDEN_RUNTIME_ESCAPE:
            self.result.alias_escape_violations.append(
                f"{self._rel_path}::{enclosing}: alias-escape",
            )
        elif kind == CanonicalConstructorUsageKind.UNKNOWN:
            self.result.unknown_usage_violations.append(
                f"{self._rel_path}::{enclosing}: unknown-usage-context",
            )

    def _visit_with_parent(self, parent: ast.AST, child: ast.AST | None) -> None:
        if child is None:
            return
        self._parent_stack.append(parent)
        try:
            self.visit(child)
        finally:
            self._parent_stack.pop()

    def _maybe_classify_expr(self, node: ast.expr) -> None:
        if not _expr_is_canonical_constructor_ref(node, self._current_prov()):
            return
        parent = self._parent_stack[-2] if len(self._parent_stack) >= 2 else None
        kind = classify_canonical_constructor_usage(
            node,
            parent,
            in_annotation=self._in_annotation(),
            parent_stack=self._parent_tuple(),
            prov=self._current_prov(),
        )
        self._record_usage(node, kind)

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.expr):
            self._maybe_classify_expr(node)
        self._parent_stack.append(node)
        try:
            super().visit(node)
        finally:
            self._parent_stack.pop()

    def _visit_annotation_subtree(self, node: ast.AST | None) -> None:
        if node is None:
            return
        self._annotation_depth += 1
        try:
            self.visit(node)
        finally:
            self._annotation_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.name == CANONICAL_CHILD_CLASS:
            scoped = self._current_prov().copy()
            scoped.local_child_execution_runner_class = True
            scoped.constructor_names.discard(CANONICAL_CHILD_CLASS)
            self._scope_stack.append(scoped)
        else:
            self._scope_stack.append(self._current_prov().copy())
        self._class_stack.append(node.name)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)
        for stmt in node.body:
            self.visit(stmt)
        self._class_stack.pop()
        self._scope_stack.pop()
        self._record_constructor_shadow(node.name)

    def _visit_callable_definition_time_expressions(
        self,
        parent: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
        args: ast.arguments,
    ) -> None:
        """Defaults/kw-only defaults evaluate in the enclosing scope (before parameters bind)."""
        for default in (*args.defaults, *args.kw_defaults):
            if default is not None:
                self._visit_with_parent(parent, default)

    def _enter_callable_lexical_scope(
        self,
        function_name: str | None,
        args: ast.arguments,
    ) -> None:
        self._scope_stack.append(self._current_prov().copy())
        if function_name is not None:
            self._function_stack.append(function_name)
        self._apply_lexical_binds_from_site(args)

    def _exit_callable_lexical_scope(self, function_name: str | None) -> None:
        if function_name is not None:
            self._function_stack.pop()
        self._scope_stack.pop()
        if function_name is not None:
            self._record_constructor_shadow(function_name)

    def _visit_function_lexical_body(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._visit_annotation_subtree(node.returns)
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            self._visit_annotation_subtree(arg.annotation)
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_callable_definition_time_expressions(node, node.args)
        self._enter_callable_lexical_scope(node.name, node.args)
        self._visit_function_lexical_body(node)
        self._exit_callable_lexical_scope(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_callable_definition_time_expressions(node, node.args)
        self._enter_callable_lexical_scope(node.name, node.args)
        self._visit_function_lexical_body(node)
        self._exit_callable_lexical_scope(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._visit_callable_definition_time_expressions(node, node.args)
        self._enter_callable_lexical_scope(None, node.args)
        self._visit_with_parent(node, node.body)
        self._exit_callable_lexical_scope(None)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._apply_lexical_binds_from_target(node.target)
        self._visit_with_parent(node, node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)  # type: ignore[arg-type]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._apply_lexical_binds_from_target(item.optional_vars)
                self._visit_with_parent(node, item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)  # type: ignore[arg-type]

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name is not None:
                self._record_constructor_shadow(handler.name)
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_Match(self, node: ast.Match) -> None:
        self.visit(node.subject)
        for case in node.cases:
            self._apply_lexical_binds_from_match_pattern(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        prov = self._current_prov()
        for name in _assignment_targets_canonical_rebind([node.target], node.value, prov):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: rebind {name}")
        self._visit_with_parent(node, node.value)
        self._apply_lexical_binds_from_target(node.target)
        self._visit_with_parent(node, node.target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_annotation_subtree(node.annotation)
        if node.value is not None:
            prov = self._current_prov()
            for name in _assignment_targets_canonical_rebind([node.target], node.value, prov):
                enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
                self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: rebind {name}")
            self._visit_with_parent(node, node.value)
        if isinstance(node.target, ast.Name):
            self._record_constructor_shadow(node.target.id)
        self._visit_with_parent(node, node.target)

    def visit_Return(self, node: ast.Return) -> None:
        self._visit_with_parent(node, node.value)

    def visit_Yield(self, node: ast.Yield) -> None:
        self._visit_with_parent(node, node.value)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self._visit_with_parent(node, node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        self._visit_with_parent(node, node.value)

    def _in_class_body_scope(self) -> bool:
        return bool(self._class_stack) and not self._function_stack

    def _record_constructor_shadow(self, name: str) -> None:
        prov = self._current_prov()
        if name not in prov.constructor_names:
            return
        enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
        self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: shadow {name}")
        prov.shadowed_constructor_names.add(name)

    def _apply_lexical_binds(self, names: set[str]) -> None:
        for name in names:
            self._record_constructor_shadow(name)

    def _apply_lexical_binds_from_site(self, site: ast.AST) -> None:
        try:
            self._apply_lexical_binds(lexical_bound_names(site))
        except LexicalBindingFormError as exc:
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.lexical_binding_violations.append(
                f"{self._rel_path}::{enclosing}: {exc}",
            )

    def _apply_lexical_binds_from_target(self, target: ast.expr) -> None:
        try:
            self._apply_lexical_binds(lexical_bound_names_from_target(target))
        except LexicalBindingFormError as exc:
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.lexical_binding_violations.append(
                f"{self._rel_path}::{enclosing}: {exc}",
            )

    def _apply_lexical_binds_from_match_pattern(self, pattern: ast.AST) -> None:
        try:
            self._apply_lexical_binds(lexical_bound_names_from_match_pattern(pattern))
        except LexicalBindingFormError as exc:
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.lexical_binding_violations.append(
                f"{self._rel_path}::{enclosing}: {exc}",
            )

    def visit_Import(self, node: ast.Import) -> None:
        if self._in_class_body_scope() and _import_binds_canonical_child(node):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.class_body_import_violations.append(
                f"{self._rel_path}::{enclosing}: class-body canonical import",
            )
            return
        _apply_import(node, self._current_prov())
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._in_class_body_scope() and _import_from_binds_canonical_child(node):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.class_body_import_violations.append(
                f"{self._rel_path}::{enclosing}: class-body canonical import",
            )
            return
        _apply_import_from(node, self._current_prov())
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        prov = self._current_prov()
        for name in _assignment_targets_canonical_rebind(node.targets, node.value, prov):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.rebind_violations.append(f"{self._rel_path}::{enclosing}: rebind {name}")
        self._visit_with_parent(node, node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record_constructor_shadow(target.id)
            else:
                for name in _iter_assignment_target_names(target):
                    self._record_constructor_shadow(name)
            self._visit_with_parent(node, target)

    def visit_Call(self, node: ast.Call) -> None:
        self._visit_with_parent(node, node.func)
        for arg in node.args:
            self._visit_with_parent(node, arg)
        for keyword in node.keywords:
            self._visit_with_parent(node, keyword.value)
        if _is_canonical_constructor_call(node, self._current_prov()):
            enclosing = _enclosing_symbol(self._class_stack, self._function_stack)
            self.result.surfaces.add(_surface_key(self._rel_path, enclosing))


def analyze_child_execution_runner_discovery(rel_path: str, source: str) -> ChildRunnerDiscoveryResult:
    normalized = rel_path.replace("\\", "/")
    tree = ast.parse(source, filename=normalized)
    if not isinstance(tree, ast.Module):
        return ChildRunnerDiscoveryResult()
    module_prov = _collect_module_import_provenance(tree)
    visitor = _ChildRunnerDiscoveryVisitor(normalized, module_prov)
    for stmt in tree.body:
        visitor.visit(stmt)
    return visitor.result


def discover_child_execution_runner_constructor_surfaces() -> frozenset[str]:
    """Registry-independent canonical ``ChildExecutionRunner(...)`` surfaces under ``intergrax/``."""
    discovered: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        discovered |= analyze_child_execution_runner_discovery(rel, text).surfaces
    return frozenset(discovered)


def discover_child_execution_runner_surfaces_in_source(rel_path: str, source: str) -> frozenset[str]:
    return frozenset(analyze_child_execution_runner_discovery(rel_path, source).surfaces)


def discover_canonical_constructor_rebinding_violations() -> list[str]:
    violations: list[str] = []
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        result = analyze_child_execution_runner_discovery(rel, text)
        violations.extend(result.rebind_violations)
        violations.extend(result.alias_escape_violations)
        violations.extend(result.class_body_import_violations)
        violations.extend(result.unknown_usage_violations)
        violations.extend(result.lexical_binding_violations)
    return violations


def discover_canonical_constructor_rebindings_in_source(rel_path: str, source: str) -> list[str]:
    result = analyze_child_execution_runner_discovery(rel_path, source)
    return [
        *result.rebind_violations,
        *result.alias_escape_violations,
        *result.class_body_import_violations,
        *result.unknown_usage_violations,
        *result.lexical_binding_violations,
    ]


def discover_canonical_constructor_usage_records_in_source(
    rel_path: str,
    source: str,
) -> list[CanonicalConstructorUsageRecord]:
    return list(analyze_child_execution_runner_discovery(rel_path, source).usage_records)


def production_canonical_constructor_usage_inventory() -> tuple[
    list[CanonicalConstructorUsageRecord],
    frozenset[str],
]:
    records: list[CanonicalConstructorUsageRecord] = []
    surfaces: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        result = analyze_child_execution_runner_discovery(rel, text)
        records.extend(result.usage_records)
        surfaces |= result.surfaces
    return records, frozenset(surfaces)


class _WireHostProfileVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self._rel_path = rel_path
        self._function_stack: list[str] = []
        self.roots: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "wire_host_effective_profile_execution":
            fn = self._function_stack[-1] if self._function_stack else "<module>"
            self.roots.add(_surface_key(self._rel_path, fn))
        self.generic_visit(node)


def discover_wire_host_effective_profile_execution_roots() -> frozenset[str]:
    """Production call sites of ``wire_host_effective_profile_execution(...)`` under ``intergrax/``."""
    discovered: set[str] = set()
    for py_path in _iter_intergrax_production_py_files():
        rel = str(py_path.relative_to(repo_root())).replace("\\", "/")
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(py_path))
        visitor = _WireHostProfileVisitor(rel)
        visitor.visit(tree)
        discovered |= visitor.roots
    return frozenset(discovered)


def discover_wire_host_roots_in_source(rel_path: str, source: str) -> frozenset[str]:
    normalized = rel_path.replace("\\", "/")
    tree = ast.parse(source, filename=normalized)
    visitor = _WireHostProfileVisitor(normalized)
    visitor.visit(tree)
    return frozenset(visitor.roots)


def profile_aware_root_forwards_child_context_inheritance(source: str, root_surface_key: str) -> bool:
    """
    Mechanical check: enclosing function wires ``host_profile.child_context_inheritance``
    into host orchestration spec construction.
    """
    _path, enclosing = root_surface_key.split("::", 1)
    tree = ast.parse(source)
    target_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == enclosing:
            target_fn = node
            break
    if target_fn is None:
        return False
    for node in ast.walk(target_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        call_name = None
        if isinstance(func, ast.Name):
            call_name = func.id
        elif isinstance(func, ast.Attribute):
            call_name = func.attr
        if call_name != "build_host_orchestration_loop_init_spec_from_environment":
            continue
        for keyword in node.keywords:
            if keyword.arg != "child_context_inheritance":
                continue
            value = keyword.value
            if isinstance(value, ast.Attribute) and value.attr == "child_context_inheritance":
                if isinstance(value.value, ast.Name) and value.value.id == "host_profile":
                    return True
    return False


__all__ = [
    "CANONICAL_CHILD_CLASS",
    "CANONICAL_CHILD_MODULE",
    "CanonicalConstructorUsageKind",
    "CanonicalConstructorUsageRecord",
    "ChildRunnerDiscoveryResult",
    "analyze_child_execution_runner_discovery",
    "LexicalBindingFormError",
    "classify_canonical_constructor_usage",
    "lexical_bound_names",
    "lexical_bound_names_from_match_pattern",
    "lexical_bound_names_from_target",
    "discover_canonical_constructor_rebinding_violations",
    "discover_canonical_constructor_rebindings_in_source",
    "discover_canonical_constructor_usage_records_in_source",
    "discover_child_execution_runner_constructor_surfaces",
    "discover_child_execution_runner_surfaces_in_source",
    "discover_wire_host_effective_profile_execution_roots",
    "discover_wire_host_roots_in_source",
    "production_canonical_constructor_usage_inventory",
    "profile_aware_root_forwards_child_context_inheritance",
]
