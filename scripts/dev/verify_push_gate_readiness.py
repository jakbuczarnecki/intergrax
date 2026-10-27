#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Preflight checklist: local push gate + CI smoke SSOT ready for team rollout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github" / "workflows" / "unit-tests.yml"


def _fail(message: str) -> None:
    print(f"FAIL  {message}", file=sys.stderr)


def _ok(message: str) -> None:
    print(f"OK    {message}")


def main() -> int:
    errors = 0

    required_paths = (
        _REPO / ".githooks" / "pre-push",
        _REPO / ".githooks" / "pre-commit",
        _REPO / "scripts" / "ci" / "run_push_gate.py",
        _REPO / "scripts" / "ci" / "run_pre_commit_gate.py",
        _REPO / "scripts" / "ci" / "diff_scoped_gates.py",
        _REPO / "scripts" / "ci" / "run_regression_gate_ci.py",
        _REPO / "docs" / "project" / "technical" / "guides" / "LOCAL_PUSH_GATE.md",
        _REPO / "docs" / "project" / "technical" / "guides" / "ENTERPRISE_QUALITY_WORKFLOW.md",
    )
    for path in required_paths:
        if not path.is_file():
            _fail(f"missing {path.relative_to(_REPO)}")
            errors += 1
        else:
            _ok(f"present {path.relative_to(_REPO)}")

    sys.path.insert(0, str(_REPO / "scripts" / "ci"))
    try:
        from push_gate_config import load_push_gate_config  # noqa: PLC0415
        from run_regression_gate_ci import smoke_steps  # noqa: PLC0415

        config = load_push_gate_config()
        if config.profile not in {"smoke", "full", "all"}:
            _fail(f"push_gate profile={config.profile} (expected smoke for rollout)")
            errors += 1
        else:
            _ok(f"push_gate profile={config.profile}")

        steps = smoke_steps(include_uv_sync=False)
        if len(steps) != 5:
            _fail(f"smoke_steps count={len(steps)} (expected 5 without uv sync)")
            errors += 1
        else:
            _ok("smoke_steps SSOT (5 steps without uv sync)")
    except Exception as exc:  # noqa: BLE001
        _fail(f"import smoke SSOT: {exc}")
        errors += 1

    if _WORKFLOW.is_file():
        text = _WORKFLOW.read_text(encoding="utf-8")
        if "development" not in text or "run_regression_gate_ci.py --profile smoke" not in text:
            _fail("unit-tests.yml missing development push or smoke SSOT step")
            errors += 1
        else:
            _ok("unit-tests.yml smoke SSOT + development push")
    else:
        _fail("missing unit-tests.yml")
        errors += 1

    hooks = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    hooks_path = (hooks.stdout or "").strip()
    if hooks_path == ".githooks":
        _ok("core.hooksPath=.githooks (this clone)")
    else:
        print(
            f"WARN  core.hooksPath={hooks_path or '(default)'} — run scripts/dev/install_git_hooks.py",
            file=sys.stderr,
        )

    if errors:
        print(f"\n{errors} check(s) failed.", file=sys.stderr)
        return 1
    print("\nPush gate pipeline ready for rollout (install hooks per clone; configure branch protection on GitHub).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
