#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Pre-push gate entrypoint (reads [tool.intergrax.push_gate], runs regression profile)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from push_gate_config import load_push_gate_config, push_gate_skipped  # noqa: E402


def main() -> int:
    config = load_push_gate_config()
    if push_gate_skipped(config):
        print(
            f"[WARN] Push gate skipped ({config.skip_env} is set). "
            "Use only for emergencies.",
            file=sys.stderr,
        )
        return 0
    if config.profile == "off":
        print("[INFO] Push gate profile is off ([tool.intergrax.push_gate]).")
        return 0

    print(f"[INFO] Running push gate profile={config.profile} ...")
    regression = _REPO / "scripts" / "ci" / "run_regression_gate_ci.py"
    completed = subprocess.run(
        [sys.executable, str(regression), "--profile", config.profile],
        cwd=_REPO,
        check=False,
    )
    if completed.returncode != 0:
        print("[ERROR] Push gate failed. Fix issues or set bypass only in emergencies.", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
