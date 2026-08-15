# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified scaffold CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.scaffold.new_agent import (
    SCAFFOLD_PATTERNS,
    create_agent,
    _slug as agent_slug,
    _class_name,
)
from intergrax.scaffold.new_application import register_parser as register_application_parser
from intergrax.scaffold.new_application import run_new_application
from intergrax.scaffold.new_integration import register_parser as register_new_integration_parser
from intergrax.scaffold.new_integration import run_new_integration
from intergrax.scaffold.new_skill import register_parser as register_new_skill_parser
from intergrax.scaffold.new_skill import run_new_skill
from intergrax.scaffold.new_tool_bundle import register_parser as register_new_tool_bundle_parser
from intergrax.scaffold.new_tool_bundle import run_new_tool_bundle
from intergrax.scaffold.new_stack import register_parser as register_new_stack_parser
from intergrax.scaffold.new_stack import run_new_stack
from intergrax.scaffold.expand_application import register_parser as register_expand_parser
from intergrax.scaffold.expand_application import run_expand


def register_scaffold_commands(sub: argparse._SubParsersAction) -> None:
    """Register scaffold subcommands on a parent parser."""

    new_agent = sub.add_parser(
        "new-agent",
        help="Create agents/<name>/ (default: typed reflex cognitive pattern)",
    )
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
    new_agent.add_argument(
        "--reference",
        action="store_true",
        help="Use HarnessReferenceAgent template (lab/product hosts inject LabHarnessContext)",
    )
    new_agent.add_argument(
        "--pattern",
        choices=sorted(SCAFFOLD_PATTERNS),
        default=None,
        help="Cognitive pattern (default: reflex when --uaep not set)",
    )
    new_agent.add_argument(
        "--uaep",
        action="store_true",
        help="Removed — raises error; use default ACP pattern scaffold",
    )

    register_application_parser(sub)
    register_new_stack_parser(sub)
    register_expand_parser(sub)
    register_new_skill_parser(sub)
    register_new_integration_parser(sub)
    register_new_tool_bundle_parser(sub)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intergrax.scaffold",
        description="Scaffold Intergrax agents (Tier-2) and applications (Tier-3).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    register_scaffold_commands(sub)
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
                reference=args.reference,
                pattern=args.pattern,
                uaep=args.uaep,
            )
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        slug = agent_slug(args.name)
        class_name = _class_name(slug)
        if args.uaep or args.reference:
            print(f"Created UAEP agent scaffold at {path}")
        else:
            pattern_label = args.pattern or "reflex"
            print(f"Created ACP pattern ({pattern_label}) agent scaffold at {path}")
        print(f"  Register: from {slug}.{slug}_agent import {class_name}")
        print(f"  Test:     uv run pytest {path / 'tests'} -q")
        print(f"  Guide:    docs/project/technical/guides/AGENT_CREATION_GUIDE.md")
        return 0

    if args.command == "new-application":
        return run_new_application(args)

    if args.command == "new-stack":
        return run_new_stack(args)

    if args.command == "expand":
        return run_expand(args)

    if args.command in ("new-skill", "new-skill-bundle"):
        return run_new_skill(args)

    if args.command == "new-integration":
        return run_new_integration(args)

    if args.command == "new-tool-bundle":
        return run_new_tool_bundle(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
