#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""TYP-05: reflection and string-key dispatch signals on the frozen contract surface."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from scripts.maintenance.measure_frozen_contract_typ import DEFAULT_SURFACE, REPO_ROOT, measure_contract_surface

_DISPATCH_HINTS = ("dispatch", "registry", "plugin", "handler_map", "by_name")


def _reflection_dispatch_hits(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    source = ast.parse(text)
    string_keys = 0
    getattr_dyn = 0
    for node in ast.walk(source):
        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                string_keys += 1
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "getattr":
                if node.args and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    getattr_dyn += 1
    lowered = text.lower()
    hint_hits = sum(1 for h in _DISPATCH_HINTS if h in lowered)
    return {"string_key_subscripts": string_keys, "getattr_string_key": getattr_dyn, "dispatch_hint_tokens": hint_hits}


def measure_typ05(surface_root: Path = DEFAULT_SURFACE) -> dict[str, Any]:
    report: dict[str, Any] = {
        "files": {},
        "totals": {
            "string_key_subscripts": 0,
            "getattr_string_key": 0,
            "dispatch_hint_tokens": 0,
        },
    }
    for path in sorted(surface_root.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        hits = _reflection_dispatch_hits(path)
        report["files"][rel] = hits
        for k, v in hits.items():
            report["totals"][k] += v
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--surface", type=Path, default=DEFAULT_SURFACE)
    args = parser.parse_args()
    typ_base = measure_contract_surface(args.surface)
    typ05 = measure_typ05(args.surface)
    payload = {"typ_metrics": typ_base["metrics"], "file_fingerprints": typ_base["file_fingerprints"], "typ05": typ05}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        t = typ05["totals"]
        print("typ05 totals:", t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
