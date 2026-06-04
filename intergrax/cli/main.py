# © Artur Czarnecki. All rights reserved.

"""Unified CLI: scaffold, run, doctor (Phase DX-3)."""

from __future__ import annotations

import argparse
import sys

from intergrax.scaffold.cli import main as scaffold_main
from intergrax.scaffold.cli import register_scaffold_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax",
        description="Intergrax harness developer CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_scaffold_commands(sub)
    from intergrax.cli.run import register_parser as register_run
    from intergrax.cli.doctor import register_parser as register_doctor
    from intergrax.cli.integrations_pick import register_parser as register_pick
    from intergrax.cli.init_project import register_parser as register_init

    register_run(sub)
    register_doctor(sub)
    register_pick(sub)
    register_init(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 0
    if argv[0] in {
        "new-agent",
        "new-application",
        "new-stack",
        "new-skill",
        "new-skill-bundle",
        "new-integration",
        "new-tool-bundle",
        "expand",
    }:
        return scaffold_main(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        from intergrax.cli.run import run_command

        return run_command(args)
    if args.command == "doctor":
        from intergrax.cli.doctor import run_doctor

        return run_doctor(args)
    if args.command == "integrations-pick":
        from intergrax.cli.integrations_pick import run_pick

        return run_pick(args)
    if args.command == "init":
        from intergrax.cli.init_project import run_init

        return run_init(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
