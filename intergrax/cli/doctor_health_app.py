# © Artur Czarnecki. All rights reserved.

"""``intergrax doctor health-app`` — environment health score for Tier-3 hosts (APP-OPS-3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.applications._shared.health_score_wiring import (
    build_application_health_score,
    format_environment_health_score,
)
from intergrax.applications.contracts.environment_health_score import PRODUCTION_READY_THRESHOLD
from intergrax.applications.contracts.manifest import ApplicationManifest


def _load_product_manifest(app_id: str) -> tuple[str, ApplicationManifest]:
    from intergrax.applications._shared.product_manifest_registry import iter_product_manifests

    for product_id, manifest in iter_product_manifests():
        if product_id == app_id or manifest.app_id == app_id:
            return product_id, manifest
    raise ValueError(f"unknown product app id: {app_id!r}")


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "health-app",
        help="Score Tier-3 application environment health for release review",
    )
    parser.add_argument("--app", required=True, help="Product app_id (legal, research, ...)")
    parser.add_argument("--json", action="store_true", help="Emit JSON health artifact")
    parser.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Write JSON health artifact to path (release tag CI)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help=f"Exit 1 when overall score is below threshold (default: {PRODUCTION_READY_THRESHOLD})",
    )


def run_doctor_health_app(args: argparse.Namespace) -> int:
    repo_root = Path(args.root).resolve() if hasattr(args, "root") else Path.cwd().resolve()
    for path in (repo_root, repo_root / "applications", repo_root / "agents"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)

    try:
        product_id, manifest = _load_product_manifest(args.app)
    except ValueError as exc:
        print(str(exc))
        return 2

    rollup = build_application_health_score(product_id, manifest, repo_root=repo_root)
    env_score = rollup.environments[0]

    if args.json:
        print(json.dumps(rollup.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        print(format_environment_health_score(env_score))

    if args.write is not None:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            json.dumps(rollup.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    threshold = (
        PRODUCTION_READY_THRESHOLD if args.fail_below is None else float(args.fail_below)
    )
    if env_score.overall < threshold or not rollup.production_ready:
        return 1
    return 0
