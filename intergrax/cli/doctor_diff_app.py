# © Artur Czarnecki. All rights reserved.

"""``intergrax doctor diff-app`` — environment diff for Tier-3 hosts (APP-EVOL-6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.applications._shared.environment_diff_wiring import (
    build_application_environment_diff,
    format_application_environment_diff,
)
from intergrax.applications.contracts.application_environment_diff import DiffRiskLevel
from intergrax.applications.contracts.manifest import ApplicationManifest


def _load_product_manifest(app_id: str) -> ApplicationManifest:
    from intergrax.applications._shared.product_manifest_registry import iter_product_manifests

    for product_id, manifest in iter_product_manifests():
        if product_id == app_id or manifest.app_id == app_id:
            return manifest
    raise ValueError(f"unknown product app id: {app_id!r}")


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "diff-app",
        help="Diff two application environment versions for pre-deploy review",
    )
    parser.add_argument("--app", required=True, help="Product app_id (legal, research, ...)")
    parser.add_argument("--left", required=True, help="Left manifest version label")
    parser.add_argument("--right", required=True, help="Right manifest version label")
    parser.add_argument("--json", action="store_true", help="Emit JSON diff artifact")
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit 1 when risk_level is high",
    )


def run_doctor_diff_app(args: argparse.Namespace) -> int:
    repo_root = Path(args.root).resolve() if hasattr(args, "root") else Path.cwd().resolve()
    for path in (repo_root, repo_root / "applications", repo_root / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    try:
        manifest = _load_product_manifest(args.app)
    except ValueError as exc:
        print(str(exc))
        return 2

    env = manifest.resolved_environment()
    left_manifest = manifest.model_copy(update={"version": args.left})
    right_manifest = manifest.model_copy(update={"version": args.right})
    diff = build_application_environment_diff(
        left_manifest,
        env,
        right_manifest,
        env,
    )

    if args.json:
        print(json.dumps(diff.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(format_application_environment_diff(diff))

    if args.fail_on_high and diff.risk_level is DiffRiskLevel.HIGH:
        return 1
    return 0
