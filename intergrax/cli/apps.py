# © Artur Czarnecki. All rights reserved.

"""``intergrax apps`` — Tier-3 application registry CLI (APP-OPS-4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.applications._shared.registry_ops_wiring import (
    format_application_entry,
    get_application,
    list_applications,
    sync_platform_registries,
)


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("apps", help="Tier-3 application registry")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    apps_sub = parser.add_subparsers(dest="apps_command", required=True)
    apps_sub.add_parser("list", help="List registered applications")
    apps_sub.add_parser("sync", help="Rebuild registry artifacts under build/")
    show = apps_sub.add_parser("show", help="Show one application")
    show.add_argument("app_id", help="Application app_id")


def _ensure_paths(repo_root: Path) -> None:
    for path in (repo_root, repo_root / "applications", repo_root / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def run_apps(args: argparse.Namespace) -> int:
    repo_root = args.root.resolve()
    _ensure_paths(repo_root)

    if args.apps_command == "sync":
        app_registry, env_registry = sync_platform_registries(repo_root)
        print(f"synced {len(app_registry.entries)} applications")
        print(f"synced {len(env_registry.entries)} environments")
        return 0

    if args.apps_command == "list":
        entries = list_applications(repo_root)
        if getattr(args, "json", False):
            payload = [entry.model_dump(mode="json") for entry in entries]
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        for entry in entries:
            print(format_application_entry(entry))
        return 0

    if args.apps_command == "show":
        entry = get_application(repo_root, args.app_id)
        if entry is None:
            print(f"unknown application: {args.app_id!r}")
            return 2
        print(json.dumps(entry.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0

    return 2
