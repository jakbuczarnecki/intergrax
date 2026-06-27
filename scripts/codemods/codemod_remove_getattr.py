#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Replace getattr(...) call sites with intergrax.utils.attribute_access.optional."""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP = frozenset(
    {
        "intergrax/utils/attribute_access.py",
        "intergrax/utils/lazy_export.py",
    }
)
IMPORT = "from intergrax.utils import attribute_access\n"
GETATTR_RE = re.compile(r"\bgetattr\s*\(")


class GetattrReplacer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "getattr":
            return node
        args = node.args
        if len(args) < 2:
            return node
        obj = args[0]
        name_arg = args[1]
        if not isinstance(name_arg, ast.Constant) or not isinstance(name_arg.value, str):
            return node
        if len(args) == 2:
            call = ast.Call(
                func=ast.Attribute(value=ast.Name(id="attribute_access", ctx=ast.Load()), attr="optional", ctx=ast.Load()),
                args=[obj, name_arg],
                keywords=[],
            )
        else:
            call = ast.Call(
                func=ast.Attribute(value=ast.Name(id="attribute_access", ctx=ast.Load()), attr="optional", ctx=ast.Load()),
                args=[obj, name_arg, args[2]],
                keywords=[],
            )
        return ast.copy_location(call, node)


def patch_file(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    if rel in SKIP:
        return False
    source = path.read_text(encoding="utf-8")
    if not GETATTR_RE.search(source):
        return False
    tree = ast.parse(source)
    new_tree = GetattrReplacer().visit(tree)
    ast.fix_missing_locations(new_tree)
    updated = ast.unparse(new_tree)
    if "attribute_access" not in updated:
        lines = updated.splitlines(keepends=True)
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index
                break
        lines.insert(insert_at, IMPORT)
        updated = "".join(lines)
    elif "from intergrax.utils import attribute_access" not in updated and "import attribute_access" not in updated:
        lines = updated.splitlines(keepends=True)
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = index
                break
        lines.insert(insert_at, IMPORT)
        updated = "".join(lines)
    if updated != source:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def main() -> int:
    changed = 0
    for root_name in ("intergrax", "agents"):
        root = REPO / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts:
                continue
            if patch_file(path):
                changed += 1
                print(path.relative_to(REPO).as_posix())
    print(f"Patched {changed} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
