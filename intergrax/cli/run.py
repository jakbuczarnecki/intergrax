# © Artur Czarnecki. All rights reserved.

"""``intergrax run`` — start a Tier-3 ASGI app (Phase DX-3.2)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("run", help="Run uvicorn for module:app (loads .env from cwd)")
    parser.add_argument(
        "target",
        help="ASGI target, e.g. my_lab_application.host.main:app",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def run_command(args: argparse.Namespace) -> int:
    _load_dotenv()
    module_path, _, attr = args.target.partition(":")
    if not module_path or not attr:
        print("error: target must be module:app", file=sys.stderr)
        return 1
    import uvicorn

    print(f"Starting {args.target} at http://{args.host}:{args.port}/")
    print("  Tip: intergrax doctor — verify tier imports and scaffold alignment")
    uvicorn.run(
        args.target,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0
