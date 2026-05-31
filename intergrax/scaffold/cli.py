# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified scaffold CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.scaffold.new_agent import create_agent, _slug as agent_slug, _class_name
from intergrax.scaffold.new_application import register_parser as register_application_parser
from intergrax.scaffold.new_application import run_new_application
from intergrax.scaffold.new_stack import register_parser as register_new_stack_parser
from intergrax.scaffold.new_stack import run_new_stack


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax.scaffold",
        description="Scaffold Intergrax agents (Tier-2) and applications (Tier-3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    new_agent = sub.add_parser("new-agent", help="Create agents/<name>/ from UAEP template")
    new_agent.add_argument("name", help="Agent slug (e.g. document_automation)")
    new_agent.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        default=[],
        help="Capability id (repeatable; default: <name>.basic)",
    )
    new_agent.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    new_agent.add_argument("--force", action="store_true", help="Overwrite if exists")

    register_application_parser(sub)
    register_new_stack_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "new-agent":
        try:
            path = create_agent(
                name=args.name,
                capabilities=args.capabilities,
                root=args.root.resolve(),
                force=args.force,
            )
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        slug = agent_slug(args.name)
        class_name = _class_name(slug)
        print(f"Created UAEP agent scaffold at {path}")
        print(f"  Register: from {slug}.{slug}_agent import {class_name}")
        print(f"  Test:     uv run pytest {path / 'tests'} -q")
        print(f"  Guide:    docs/AGENT_CREATION_GUIDE.md")
        return 0

    if args.command == "new-application":
        return run_new_application(args)

    if args.command == "new-stack":
        return run_new_stack(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
