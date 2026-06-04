# © Artur Czarnecki. All rights reserved.

"""``intergrax init`` — external project template (Phase DX-6.3)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("init", help="Create a minimal external Intergrax harness project")
    parser.add_argument("name", help="Project directory name")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true")


def run_init(args: argparse.Namespace) -> int:
    template = Path(__file__).resolve().parent.parent / "scaffold" / "external_project"
    target = args.root.resolve() / args.name
    if target.exists() and not args.force:
        print(f"error: {target} already exists", file=__import__("sys").stderr)
        return 1
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(template, target)
    print(f"Created harness project at {target}")
    print("  pip install intergrax  # or use monorepo editable install")
    print("  uv run python app.py")
    return 0
