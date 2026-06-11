# © Artur Czarnecki. All rights reserved.

"""Smoke-check ACP pattern scaffold output — default and explicit --pattern (ACP-11)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_scaffold(root: Path, *, slug: str, extra_args: list[str]) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "intergrax.scaffold",
        "new-agent",
        slug,
        "--capability",
        f"{slug}.basic",
        "--root",
        str(root),
        *extra_args,
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        print("check_scaffold_acp_pattern: FAIL", file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        return 1
    return 0


def _assert_typed_agent(root: Path, slug: str, *, base_class: str) -> int:
    agent_py = root / "agents" / slug / f"{slug}_agent.py"
    if not agent_py.is_file():
        print(f"check_scaffold_acp_pattern: FAIL — missing {agent_py}", file=sys.stderr)
        return 1
    content = agent_py.read_text(encoding="utf-8")
    forbidden = ("def get_steps", "async def run_step", "def decide_after_step")
    for token in forbidden:
        if token in content:
            print(f"check_scaffold_acp_pattern: FAIL — UAEP boilerplate found: {token}", file=sys.stderr)
            return 1
    if base_class not in content:
        print(f"check_scaffold_acp_pattern: FAIL — missing {base_class}", file=sys.stderr)
        return 1
    if "async def perceive" not in content:
        print("check_scaffold_acp_pattern: FAIL — missing perceive hook", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "agents").mkdir()

        if _run_scaffold(root, slug="acp_default", extra_args=[]):
            return 1
        if _assert_typed_agent(root, "acp_default", base_class="ReflexAgent"):
            return 1

        if _run_scaffold(root, slug="acp_smoke", extra_args=["--pattern", "react"]):
            return 1
        if _assert_typed_agent(root, "acp_smoke", base_class="ReActAgent"):
            return 1

    print("check_scaffold_acp_pattern: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
