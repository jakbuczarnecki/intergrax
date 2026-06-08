# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Scaffold Tier-2 agent + Tier-3 application in one CLI (Phase N.10)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intergrax.scaffold.agent_catalog import resolve_agent_specs
from intergrax.scaffold.application_names import ScaffoldApplicationNames
from intergrax.scaffold.new_agent import _class_name, _slug as agent_slug, create_agent
from intergrax.scaffold.new_application import _default_port, create_application


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "new-stack",
        help="Create agents/<slug>/ and applications/<slug>_application/ (Tier-2 + Tier-3)",
    )
    parser.add_argument(
        "name",
        help="Stack name (e.g. my_feature → agents/my_feature + my_feature_application)",
    )
    parser.add_argument(
        "--capability",
        dest="capabilities",
        action="append",
        default=[],
        help="Agent capability id (repeatable; default: <slug>.basic)",
    )
    parser.add_argument(
        "--profile",
        choices=("lab", "product"),
        default="lab",
        help="Tier-3 application profile (default: lab)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="HTTP port (lab default 8091, product default 8000)",
    )
    parser.add_argument(
        "--prefix",
        dest="route_prefix",
        default=None,
        help="HTTP route prefix (default: /v1/<short_id>)",
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Create only the agent under agents/",
    )
    parser.add_argument(
        "--app-only",
        action="store_true",
        help="Create only the application (agent must already exist)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: cwd)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite if exists")
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Minimal stack: no agent notebook/README extras; lab app without Docker/MCP/deploy doc",
    )


def run_new_stack(args: argparse.Namespace) -> int:
    if args.agent_only and args.app_only:
        print("error: use at most one of --agent-only and --app-only", file=sys.stderr)
        return 1

    root = args.root.resolve()
    slug = agent_slug(args.name)
    caps = args.capabilities or [f"{slug}.basic"]
    port = _default_port(args.profile, args.port)

    agent_path: Path | None = None
    app_path: Path | None = None

    try:
        if not args.app_only:
            agent_path = create_agent(
                name=slug,
                capabilities=caps,
                root=root,
                force=args.force,
                minimal=args.minimal,
            )
        if not args.agent_only:
            resolve_agent_specs([slug])
            app_path = create_application(
                name=slug,
                agents=[slug],
                profile=args.profile,
                root=root,
                route_prefix=args.route_prefix,
                port=port,
                force=args.force,
                minimal=args.minimal and args.profile == "lab",
            )
    except (ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    names = ScaffoldApplicationNames.resolve(slug, route_prefix=args.route_prefix, port=port)
    class_name = _class_name(slug)

    label = "minimal" if args.minimal else "standard"
    print(f"Stack {slug!r} — Tier-2 + Tier-3 scaffold ({label})")
    if agent_path is not None:
        print(f"  Agent:  {agent_path}")
        print(f"  Test:   uv run pytest {agent_path / 'tests'} -q")
    if app_path is not None:
        print(f"  App:    {app_path}")
        print(f"  Test:   uv run pytest {app_path / names.tests_pkg} -q")
        print(f"  Start:  uv run uvicorn {names.pkg}.host.main:app --host 127.0.0.1 --port {port}")
        print(f"  Docker: applications/{names.pkg}/docker/build-docker.sh")
    print(f"  Mount:  AgentBinding.mount({class_name}, ...) in {names.pkg}/manifest.py")
    print("  Guide:  docs/guides/AGENT_CREATION_GUIDE.md — Step 4E")
    return 0
