# © Artur Czarnecki. All rights reserved.

"""Static detection of governed RuntimeToolInvoker.invoke callsites (HARNESS-01)."""

from __future__ import annotations

import ast
from dataclasses import dataclass


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


def receiver_is_runtime_tool_invoker_target(receiver: ast.expr) -> bool:
    """Physical ``RuntimeToolInvoker.invoke`` receivers use invoker-shaped attribute names."""
    leaf = receiver_expression(receiver).rsplit(".", 1)[-1]
    return leaf in {"invoker", "tool_invoker", "_invoker", "_tool_invoker"} or leaf.endswith(
        "_invoker"
    )


def is_governed_runtime_tool_invoker_invoke_call(node: ast.Call) -> bool:
    """
    Governed physical invoke shape:

    ``<invoker>.invoke(state=..., request=..., ...)`` — excludes catalog host / port delegates.
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "invoke":
        return False
    if not receiver_is_runtime_tool_invoker_target(func.value):
        return False
    keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
    return "state" in keyword_names and "request" in keyword_names


def collect_governed_invoker_callsites(source: str, *, filename: str = "<memory>") -> list[GovernedInvokerCallsite]:
    tree = ast.parse(source, filename=filename)
    hits: list[GovernedInvokerCallsite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_governed_runtime_tool_invoker_invoke_call(node):
            func_attr = node.func
            assert isinstance(func_attr, ast.Attribute)
            hits.append(
                GovernedInvokerCallsite(
                    line=node.lineno,
                    receiver=receiver_expression(func_attr.value),
                )
            )
    return hits


def file_references_runtime_tool_invoker(source: str) -> bool:
    return "RuntimeToolInvoker" in source
