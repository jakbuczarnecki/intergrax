# © Artur Czarnecki. All rights reserved.

"""Static detection of governed RuntimeToolInvoker.invoke callsites (HARNESS-01)."""

from __future__ import annotations

import ast
from dataclasses import dataclass

RTI_SYMBOL = "RuntimeToolInvoker"

# Closed-world: same-file helpers that return ``state.context.config.tool_invoker``.
_CLOSED_WORLD_RTI_RESOLVER_CALLEES: frozenset[str] = frozenset({"_resolve_invoker"})


@dataclass(frozen=True, slots=True)
class GovernedInvokerCallsite:
    line: int
    receiver: str


def receiver_expression(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{receiver_expression(node.value)}.{node.attr}"
    return "<expr>"


def annotation_references_runtime_tool_invoker(node: ast.expr | None) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name) and node.id == RTI_SYMBOL:
        return True
    if isinstance(node, ast.Attribute) and node.attr == RTI_SYMBOL:
        return True
    if isinstance(node, ast.Subscript):
        return annotation_references_runtime_tool_invoker(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (
            annotation_references_runtime_tool_invoker(node.left)
            or annotation_references_runtime_tool_invoker(node.right)
        )
    return False


def _is_runtime_tool_invoker_constructor(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name) and func.id == RTI_SYMBOL:
        return True
    if isinstance(func, ast.Attribute) and func.attr == RTI_SYMBOL:
        return True
    return False


def _is_config_tool_invoker_source(node: ast.expr) -> bool:
    text = receiver_expression(node)
    return text.endswith(".context.config.tool_invoker")


def _expression_proves_runtime_tool_invoker(
    node: ast.expr,
    *,
    proven_names: frozenset[str],
) -> bool:
    if isinstance(node, ast.Name) and node.id in proven_names:
        return True
    if isinstance(node, ast.Call) and _is_runtime_tool_invoker_constructor(node):
        return True
    if _is_config_tool_invoker_source(node):
        return True
    if isinstance(node, ast.Call):
        callee = node.func
        if isinstance(callee, ast.Name) and callee.id in _CLOSED_WORLD_RTI_RESOLVER_CALLEES:
            return True
    return False


def _receiver_proves_runtime_tool_invoker(
    receiver: ast.expr,
    *,
    proven_names: frozenset[str],
    proven_self_fields: frozenset[str],
    class_rti_fields: frozenset[str],
) -> bool:
    if isinstance(receiver, ast.Name):
        return receiver.id in proven_names
    if isinstance(receiver, ast.Attribute):
        if isinstance(receiver.value, ast.Name) and receiver.value.id == "self":
            if receiver.attr in proven_self_fields or receiver.attr in class_rti_fields:
                return True
        return _expression_proves_runtime_tool_invoker(
            receiver,
            proven_names=proven_names,
        )
    return False


def is_governed_runtime_tool_invoker_invoke_call(
    node: ast.Call,
    *,
    proven_names: frozenset[str] | None = None,
    proven_self_fields: frozenset[str] | None = None,
    class_rti_fields: frozenset[str] | None = None,
) -> bool:
    """
    Governed physical invoke shape:

    ``<proven RuntimeToolInvoker>.invoke(state=..., request=..., ...)``.
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "invoke":
        return False
    names = proven_names if proven_names is not None else frozenset()
    self_fields = proven_self_fields if proven_self_fields is not None else frozenset()
    fields = class_rti_fields if class_rti_fields is not None else frozenset()
    if not _receiver_proves_runtime_tool_invoker(
        func.value,
        proven_names=names,
        proven_self_fields=self_fields,
        class_rti_fields=fields,
    ):
        return False
    keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
    return "state" in keyword_names and "request" in keyword_names


def _collect_class_runtime_tool_invoker_fields(tree: ast.Module) -> dict[str, frozenset[str]]:
    by_class: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        fields: set[str] = set()
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if annotation_references_runtime_tool_invoker(item.annotation):
                    fields.add(item.target.id)
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_params = list(item.args.args) + list(item.args.kwonlyargs)
                for stmt in item.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Attribute):
                        continue
                    target = stmt.targets[0]
                    if not (
                        isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        continue
                    value = stmt.value
                    if isinstance(value, ast.Name):
                        param = next(
                            (
                                arg
                                for arg in init_params
                                if arg.arg == value.id
                                and annotation_references_runtime_tool_invoker(arg.annotation)
                            ),
                            None,
                        )
                        if param is not None:
                            fields.add(target.attr)
        if fields:
            by_class[node.name] = frozenset(fields)
    return by_class


def _proven_bindings_for_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    class_rti_fields: frozenset[str],
) -> tuple[frozenset[str], frozenset[str]]:
    proven_names: set[str] = set()
    proven_self_fields: set[str] = set()
    for arg in fn.args.args:
        if annotation_references_runtime_tool_invoker(arg.annotation):
            proven_names.add(arg.arg)
    for arg in fn.args.kwonlyargs:
        if annotation_references_runtime_tool_invoker(arg.annotation):
            proven_names.add(arg.arg)
    for stmt in fn.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if annotation_references_runtime_tool_invoker(stmt.annotation):
                proven_names.add(stmt.target.id)
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if isinstance(target, ast.Name) and _expression_proves_runtime_tool_invoker(
                stmt.value,
                proven_names=frozenset(proven_names),
            ):
                proven_names.add(target.id)
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and _expression_proves_runtime_tool_invoker(
                    stmt.value,
                    proven_names=frozenset(proven_names),
                )
            ):
                proven_self_fields.add(target.attr)
    proven_self_fields.update(class_rti_fields)
    return frozenset(proven_names), frozenset(proven_self_fields)


def _callsites_in_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    class_rti_fields: frozenset[str],
) -> list[GovernedInvokerCallsite]:
    proven_names, proven_self_fields = _proven_bindings_for_function(
        fn,
        class_rti_fields=class_rti_fields,
    )
    hits: list[GovernedInvokerCallsite] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and is_governed_runtime_tool_invoker_invoke_call(
            node,
            proven_names=proven_names,
            proven_self_fields=proven_self_fields,
            class_rti_fields=class_rti_fields,
        ):
            func_attr = node.func
            assert isinstance(func_attr, ast.Attribute)
            hits.append(
                GovernedInvokerCallsite(
                    line=node.lineno,
                    receiver=receiver_expression(func_attr.value),
                )
            )
    return hits


def collect_governed_invoker_callsites(source: str, *, filename: str = "<memory>") -> list[GovernedInvokerCallsite]:
    tree = ast.parse(source, filename=filename)
    assert isinstance(tree, ast.Module)
    class_fields = _collect_class_runtime_tool_invoker_fields(tree)
    hits: list[GovernedInvokerCallsite] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = class_fields.get(node.name, frozenset())
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    hits.extend(_callsites_in_function(item, class_rti_fields=fields))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            hits.extend(_callsites_in_function(node, class_rti_fields=frozenset()))
    return hits


def file_references_runtime_tool_invoker(source: str) -> bool:
    return RTI_SYMBOL in source
