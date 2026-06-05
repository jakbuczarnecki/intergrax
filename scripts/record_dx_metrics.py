#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Write DX paydown metrics artifact (Phase DX-8.2)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "gate", "-q", "--co-empty"],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    gate_passed = proc.returncode == 0
    out = root / "build" / "architecture_hardening" / "dx_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "gate_passed": gate_passed,
        "author_file_count_minimal_stack": 4,
        "ttf_run_target_seconds": 60,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
