# © Artur Czarnecki. All rights reserved.

"""``intergrax envs`` — Tier-3 environment registry CLI (APP-OPS-4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.applications._shared.registry_ops_wiring import (
    format_environment_entry,
    get_environment,
    list_environments,
)


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("envs", help="Tier-3 environment registry")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    envs_sub = parser.add_subparsers(dest="envs_command", required=True)
    list_parser = envs_sub.add_parser("list", help="List registered environments")
    list_parser.add_argument("--app", dest="app_id", default=None, help="Filter by app_id")
    show = envs_sub.add_parser("show", help="Show one environment")
    show.add_argument("environment_id", help="Environment id (e.g. legal-strict)")


def _ensure_paths(repo_root: Path) -> None:
    for path in (repo_root, repo_root / "applications", repo_root / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def run_envs(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    _ensure_paths(repo_root)

    if args.envs_command == "list":
        entries = list_environments(repo_root, app_id=getattr(args, "app_id", None))
        if getattr(args, "json", False):
            payload = [entry.model_dump(mode="json") for entry in entries]
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        for entry in entries:
            print(format_environment_entry(entry))
        return 0

    if args.envs_command == "show":
        entry = get_environment(repo_root, args.environment_id)
        if entry is None:
            print(f"unknown environment: {args.environment_id!r}")
            return 2
        print(json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    return 2
