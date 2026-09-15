# © Artur Czarnecki. All rights reserved.

"""Static detection of embedded qualification harness patterns in pytest modules."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from testing_support.execution_qualification.embedded_harness_kexpr import (
    embedded_harness_call_names,
    embedded_harness_test_names,
)


@dataclass(frozen=True, slots=True)
class EmbeddedHarnessInventoryEntry:
    module_path: str
    test_function: str
    pattern: str


def inventory_embedded_harness_in_module(
    module_path: Path,
) -> tuple[EmbeddedHarnessInventoryEntry, ...]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    entries: list[EmbeddedHarnessInventoryEntry] = []
    rel = module_path.as_posix()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name in embedded_harness_test_names():
            entries.append(
                EmbeddedHarnessInventoryEntry(
                    module_path=rel,
                    test_function=node.name,
                    pattern="embedded_mandatory_test",
                ),
            )
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in embedded_harness_call_names():
                entries.append(
                    EmbeddedHarnessInventoryEntry(
                        module_path=rel,
                        test_function=node.name,
                        pattern=f"call:{name}",
                    ),
                )
    return tuple(entries)


def module_contains_embedded_harness_calls(module_path: Path) -> bool:
    return bool(inventory_embedded_harness_in_module(module_path))


def semantic_test_function_names_in_module(module_path: Path) -> frozenset[str]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }
    return frozenset(names - embedded_harness_test_names())
