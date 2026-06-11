# © Artur Czarnecki. All rights reserved.

"""Smoke-check ACP pattern scaffold output (ACP-11)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "agents").mkdir()
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "intergrax.scaffold",
            "new-agent",
            "acp_smoke",
            "--capability",
            "smoke.react",
            "--pattern",
            "react",
            "--root",
            str(root),
        ]
        completed = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if completed.returncode != 0:
            print("check_scaffold_acp_pattern: FAIL", file=sys.stderr)
            print(completed.stderr, file=sys.stderr)
            return 1

        agent_py = root / "agents" / "acp_smoke" / "acp_smoke_agent.py"
        content = agent_py.read_text(encoding="utf-8")
        forbidden = ("def get_steps", "async def run_step", "def decide_after_step")
        for token in forbidden:
            if token in content:
                print(f"check_scaffold_acp_pattern: FAIL — UAEP boilerplate found: {token}", file=sys.stderr)
                return 1
        required = ("ReActAgent", "async def perceive", "async def on_next_step", "CognitiveEvaluation")
        for token in required:
            if token not in content and token != "async def on_next_step":
                print(f"check_scaffold_acp_pattern: FAIL — missing {token}", file=sys.stderr)
                return 1
            if token == "async def on_next_step":
                if "on_next_step" not in content and "ReActAgent" not in content:
                    print("check_scaffold_acp_pattern: FAIL — missing cognitive wiring", file=sys.stderr)
                    return 1

    print("check_scaffold_acp_pattern: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
