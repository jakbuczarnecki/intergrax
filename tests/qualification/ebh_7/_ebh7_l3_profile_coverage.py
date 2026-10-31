# © Artur Czarnecki. All rights reserved.

"""Protocol 03 L3 pytest_profile coverage — CORRECTIVE_PYTEST must be in replay scope."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Final

from tests.qualification.ebh_5._proof_validation import parse_pytest_node_id

_REPO = Path(__file__).resolve().parents[3]
_MANIFEST = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/MANIFEST.json"


def l3_profile_command_argv() -> tuple[str, ...]:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cmd = manifest["pytest_profile"]["command"]
    parts = shlex.split(cmd, posix=False)
    if parts[:2] == ["uv", "run"] and parts[2] == "pytest":
        return tuple(parts[3:])
    if parts[0] == "pytest":
        return tuple(parts[1:])
    raise AssertionError(f"Unrecognized L3 profile command: {cmd!r}")


def l3_profile_covers_pytest_node(node_id: str) -> bool:
    """True when the node's module is executed by MANIFEST pytest_profile (file or directory scope)."""
    parsed = parse_pytest_node_id(node_id.split("[", 1)[0])
    if parsed is None:
        return False
    rel_path, _ = parsed
    rel_norm = rel_path.replace("\\", "/")
    argv = l3_profile_command_argv()
    for arg in argv:
        if arg.startswith("-"):
            continue
        arg_norm = arg.replace("\\", "/")
        if "::" in arg_norm:
            file_part = arg_norm.split("::", 1)[0]
            if rel_norm == file_part or rel_norm.startswith(file_part.rstrip("/") + "/"):
                return True
            continue
        if arg_norm.endswith("/"):
            prefix = arg_norm.rstrip("/")
            if rel_norm == prefix or rel_norm.startswith(prefix + "/"):
                return True
        elif rel_norm == arg_norm or rel_norm.startswith(arg_norm + "/"):
            return True
    return False


L3_PROFILE_COMMAND: Final[str] = json.loads(_MANIFEST.read_text(encoding="utf-8"))["pytest_profile"]["command"]
