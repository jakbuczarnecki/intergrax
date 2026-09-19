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


def is_governed_runtime_tool_invoker_invoke_call(node: ast.Call) -> bool:
    """
    Receiver-agnostic governed invoke shape:

    ``*.invoke(state=..., request=..., ...)`` — distinct from declarative async ports.
    """
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "invoke"):
        return False
    keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
    return "state" in keyword_names and "request" in keyword_names


def collect_governed_invoker_callsites(source: str, *, filename: str = "<memory>") -> list[GovernedInvokerCallsite]:
    tree = ast.parse(source, filename=filename)
    hits: list[GovernedInvokerCallsite] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and is_governed_runtime_tool_invoker_invoke_call(node):
            hits.append(
                GovernedInvokerCallsite(
                    line=node.lineno,
                    receiver=receiver_expression(node.func.value),
                )
            )
    return hits


def file_references_runtime_tool_invoker(source: str) -> bool:
    return "RuntimeToolInvoker" in source
