# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register scaffolded application/agent projects in the root uv workspace."""

from __future__ import annotations

import re
from pathlib import Path


def ensure_workspace_member(repo_root: Path, member: str) -> bool:
    """Ensure ``member`` appears in root ``[tool.uv.workspace].members``.

    Returns True when the root ``pyproject.toml`` was modified.
    """
    cleaned = member.strip().replace("\\", "/").strip("/")
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return False
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(
        r"(\[tool\.uv\.workspace\][^\[]*?members\s*=\s*\[)([^\]]*?)(\])",
        text,
        flags=re.S,
    )
    if not match:
        return False
    body = match.group(2)
    existing = set(re.findall(r'["\']([^"\']+)["\']', body))
    if cleaned in existing:
        return False
    insertion = f'  "{cleaned}",\n'
    if cleaned.startswith("applications/"):
        agent_idx = body.find('"agents/')
        if agent_idx >= 0:
            new_body = body[:agent_idx] + insertion + body[agent_idx:]
        else:
            new_body = body.rstrip() + "\n" + insertion
    else:
        new_body = body.rstrip() + "\n" + insertion
    if not new_body.endswith("\n"):
        new_body += "\n"
    updated = text[: match.start(2)] + new_body + text[match.end(2) :]
    pyproject.write_text(updated, encoding="utf-8", newline="\n")
    return True
