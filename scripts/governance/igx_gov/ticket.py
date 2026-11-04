# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CI = _REPO / "scripts" / "ci"


def _push_touches_runtime(paths: frozenset[str]) -> bool:
    prefixes = ("intergrax/runtime/", "agents/", "tests/")
    for path in paths:
        normalized = path.replace("\\", "/")
        if any(normalized.startswith(p) for p in prefixes):
            return True
    return False


def run_ticket(*, profile: str = "fast", skip_smoke: bool = False) -> int:
    ref = subprocess.run(
        [sys.executable, str(_REPO / "scripts/governance/verify_commit_ticket_refs.py"), "--mode", "push"],
        cwd=_REPO,
        check=False,
    )
    if ref.returncode != 0:
        return ref.returncode

    registry = subprocess.run(
        [sys.executable, str(_REPO / "scripts/governance/validate_policy_registry.py")],
        cwd=_REPO,
        check=False,
    )
    if registry.returncode != 0:
        return registry.returncode

    gate = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts/ci/run_regression_gate_ci.py"),
            "--profile",
            profile,
        ],
        cwd=_REPO,
        check=False,
    )
    if gate.returncode != 0:
        return gate.returncode

    if not skip_smoke and profile == "fast":
        sys.path.insert(0, str(_CI))
        from git_changed_paths import push_paths  # noqa: E402

        if _push_touches_runtime(push_paths()):
            print(
                "[WARN] Diff touches runtime/agents/tests — consider: "
                "uv run python -m scripts.governance.igx_gov ticket --profile smoke",
                file=sys.stderr,
            )

    print("[INFO] igx_gov ticket: OK")
    return 0
