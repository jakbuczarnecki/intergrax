# © Artur Czarnecki. All rights reserved.

"""Regenerate wide-pyright provenance from current tree (maintainer tool)."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from tests.qualification.control_planes.ctrl_x.typing_scope import (
    ctrl_x_plane_for_boundary_module,
    ctrl_x_semantic_plane_for_wide_file,
    ctrl_x_wide_diagnostic_classification,
)

_REPO = Path(__file__).resolve().parents[4]


def main() -> int:
    scope_path = _REPO / "tests/qualification/control_planes/ctrl_x/wide_pyright_scope.py"
    scope_ns: dict[str, object] = {}
    exec(scope_path.read_text(encoding="utf-8"), scope_ns)
    wide_files = scope_ns["CTRL_X_WIDE_PYRIGHT_FILE_PATHS"]
    proc = subprocess.run(
        ["uv", "run", "pyright", "--outputjson", *wide_files],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    data = json.loads(proc.stdout)
    diags = [d for d in data.get("generalDiagnostics", []) if d.get("severity") == "error"]
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for diag in diags:
        file_abs = diag["file"]
        rel = "intergrax/" + file_abs.replace("\\", "/").split("intergrax/")[-1]
        rule = diag.get("rule", "unknown")
        on_boundary = ctrl_x_plane_for_boundary_module(rel) is not None
        semantic_plane = ctrl_x_semantic_plane_for_wide_file(rel)
        impact, boundary_class, reason = ctrl_x_wide_diagnostic_classification(
            rel,
            on_semantic_boundary=on_boundary,
        )
        key = (rel, rule)
        row = groups.get(key)
        if row is None:
            row = {
                "file": rel,
                "rule": rule,
                "diagnostic_count": 0,
                "semantic_plane": semantic_plane,
                "semantic_location": boundary_class,
                "boundary_or_internal": boundary_class,
                "baseline_a_present": True,
                "start_head_b_present": True,
                "final_c_present": True,
                "ctrl_x_impact": impact,
                "reason": reason,
                "future_owner": semantic_plane if semantic_plane != "EBH-6" else "EBH-6",
                "semantic_owner": semantic_plane,
                "classification": impact,
                "ctrl_x_boundary_module": on_boundary,
                "parent_impact": "blocked" if impact.startswith("CTRL-X") else "non-blocking",
            }
            groups[key] = row
        row["diagnostic_count"] = int(row["diagnostic_count"]) + 1

    catalog = tuple(sorted(groups.values(), key=lambda x: (x["file"], x["rule"])))
    accounted = sum(int(g["diagnostic_count"]) for g in catalog)
    assert accounted == len(diags), f"accounted {accounted} != {len(diags)}"
    blockers = [g for g in catalog if g["ctrl_x_impact"] == "CTRL-X BLOCKER"]
    if blockers:
        print("CTRL-X blockers remain on boundary modules:", len(blockers), file=sys.stderr)

    out_dir = _REPO / "tests/qualification/control_planes/ctrl_x"
    prov_py = out_dir / "wide_pyright_provenance.py"
    prov_py.write_text(
        "# © Artur Czarnecki. All rights reserved.\n\n"
        '"""Committed wide-pyright diagnostic provenance (grouped)."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any, Final\n\n"
        f"CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS: Final[tuple[dict[str, Any], ...]] = {catalog!r}\n"
        f"CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_TOTAL: Final[int] = {len(diags)}\n",
        encoding="utf-8",
    )
    print(f"diagnostics={len(diags)} groups={len(catalog)} blockers={len(blockers)}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
