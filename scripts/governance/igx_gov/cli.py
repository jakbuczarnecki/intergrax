# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_GOV = _REPO / "scripts" / "governance"


def _run(script: str, *args: str) -> int:
    completed = subprocess.run(
        [sys.executable, "-u", str(_REPO / script), *args],
        cwd=_REPO,
        check=False,
    )
    return completed.returncode


def cmd_bootstrap(_: argparse.Namespace) -> int:
    steps = [
        ("scripts/dev/install_git_hooks.py", ()),
        ("scripts/dev/verify_push_gate_readiness.py", ()),
    ]
    for script, args in steps:
        code = _run(script, *args)
        if code != 0:
            return code
    print("[INFO] igx_gov bootstrap: OK")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    from scripts.governance.igx_gov import doctor as doctor_mod

    return doctor_mod.run_doctor()


def cmd_ticket(args: argparse.Namespace) -> int:
    from scripts.governance.igx_gov import ticket as ticket_mod

    return ticket_mod.run_ticket(profile=args.profile, skip_smoke=args.skip_smoke)


def cmd_before_push(args: argparse.Namespace) -> int:
    if args.push_gate_only:
        return _run("scripts/ci/run_push_gate.py")
    return cmd_ticket(argparse.Namespace(profile="fast", skip_smoke=True))


def cmd_lint(_: argparse.Namespace) -> int:
    return _run("scripts/ci/run_ruff_changed_paths.py", "--mode", "push")


def cmd_inventory(_: argparse.Namespace) -> int:
    from scripts.governance.igx_gov import inventory as inventory_mod

    inventory_mod.print_inventory()
    return 0


def cmd_sync_adapters(_: argparse.Namespace) -> int:
    print("[INFO] sync-adapters: Cursor MATRIX + advisory Claude/Codex fragments")
    code = _run("scripts/governance/generate_control_plane_matrix.py")
    if code != 0:
        return code
    return _run("scripts/governance/sync_adapter_fragments.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="igx_gov", description="InterGRAX Delivery DevOS governance CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("bootstrap", help="Install hooks + verify push gate readiness").set_defaults(func=cmd_bootstrap)
    sub.add_parser("doctor", help="Hooks, YAML, skills consistency checks").set_defaults(func=cmd_doctor)
    ticket_p = sub.add_parser("ticket", help="Pre-close / pre-push quality sequence")
    ticket_p.add_argument("--profile", default="fast", choices=("fast", "smoke", "enterprise", "full"))
    ticket_p.add_argument("--skip-smoke", action="store_true", help="Do not suggest smoke profile")
    ticket_p.set_defaults(func=cmd_ticket)
    bp = sub.add_parser("before-push", help="Alias for ticket or raw push gate")
    bp.add_argument("--push-gate-only", action="store_true")
    bp.set_defaults(func=cmd_before_push)
    sub.add_parser("lint", help="Ruff on changed Python paths (push mode)").set_defaults(func=cmd_lint)
    sub.add_parser("inventory", help="List DCP components").set_defaults(func=cmd_inventory)
    sub.add_parser(
        "sync-adapters",
        help="Regenerate Cursor MATRIX + advisory Claude/Codex adapter fragments",
    ).set_defaults(func=cmd_sync_adapters)

    args = parser.parse_args(argv)
    return int(args.func(args))
