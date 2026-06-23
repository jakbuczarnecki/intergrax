# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
#!/usr/bin/env python3

"""Create maintainer-curated public GitHub issues from YAML definitions.

Default mode is a dry-run. Real GitHub Issues are created only when --apply is
provided. The script uses the GitHub CLI (`gh`) and skips issues whose exact
title already exists.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install project dependencies first.") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "docs" / "public-adoption" / "curated_public_issues.yml"
DEFAULT_REPOSITORY = "jakbuczarnecki/intergrax"


@dataclass(frozen=True)
class CuratedIssue:
    """Validated issue definition loaded from YAML."""

    issue_id: str
    order: int
    title: str
    body: str
    labels: list[str]


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture output."""

    return subprocess.run(args, check=False, capture_output=True, text=True)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration."""

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("Config root must be a mapping")
    return loaded


def validate_safety(config: dict[str, Any]) -> None:
    """Validate safety flags from the YAML file."""

    safety = config.get("safety")
    if not isinstance(safety, dict):
        raise ConfigError("Missing safety section")
    if safety.get("dry_run_should_be_default") is not True:
        raise ConfigError("safety.dry_run_should_be_default must be true")
    if safety.get("requires_explicit_apply_flag") is not True:
        raise ConfigError("safety.requires_explicit_apply_flag must be true")
    if safety.get("creates_real_github_issues") is not False:
        raise ConfigError("YAML safety flag creates_real_github_issues must remain false")


def select_issues(
    config: dict[str, Any], *, wave: str, only: set[str], allow_deferred: bool
) -> list[CuratedIssue]:
    """Select issues for the requested wave."""

    waves = config.get("waves")
    if not isinstance(waves, dict):
        raise ConfigError("Missing waves section")

    wave_config = waves.get(wave)
    if not isinstance(wave_config, dict):
        available = ", ".join(sorted(str(key) for key in waves))
        raise ConfigError(f"Unknown wave '{wave}'. Available waves: {available}")

    if wave_config.get("open_now") is not True and not allow_deferred:
        raise ConfigError(f"Wave '{wave}' is deferred. Use --allow-deferred to inspect it.")

    raw_issues = wave_config.get("issues")
    if not isinstance(raw_issues, list):
        raise ConfigError(f"Wave '{wave}' must contain an issues list")

    selected: list[CuratedIssue] = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            raise ConfigError(f"Invalid issue entry in wave '{wave}'")

        issue_id = require_str(raw, "id")
        if only and issue_id not in only:
            continue

        labels = raw.get("labels")
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ConfigError(f"Issue '{issue_id}' must define labels as a string list")

        order = raw.get("order", 9999)
        if not isinstance(order, int):
            raise ConfigError(f"Issue '{issue_id}' order must be an integer")

        selected.append(
            CuratedIssue(
                issue_id=issue_id,
                order=order,
                title=require_str(raw, "title"),
                body=require_str(raw, "body"),
                labels=labels,
            )
        )

    missing = sorted(only - {issue.issue_id for issue in selected})
    if missing:
        raise ConfigError(f"Requested issue IDs not found in wave '{wave}': {', '.join(missing)}")

    return sorted(selected, key=lambda item: (item.order, item.issue_id))


def require_str(raw: dict[str, Any], field: str) -> str:
    """Return a required non-empty string field."""

    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        issue_id = raw.get("id", "<unknown>")
        raise ConfigError(f"Issue '{issue_id}' must define non-empty string field '{field}'")
    return value


def load_existing_issue_titles(repo: str) -> dict[str, dict[str, Any]]:
    """Load existing issue titles using GitHub CLI."""

    completed = run_command(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,title,state,url",
        ]
    )
    if completed.returncode != 0:
        raise ConfigError(f"Unable to list GitHub issues via gh: {completed.stderr.strip()}")

    loaded = json.loads(completed.stdout or "[]")
    if not isinstance(loaded, list):
        raise ConfigError("Unexpected gh issue list response")

    result: dict[str, dict[str, Any]] = {}
    for item in loaded:
        if isinstance(item, dict) and isinstance(item.get("title"), str):
            result[item["title"]] = item
    return result


def create_issue(repo: str, issue: CuratedIssue) -> None:
    """Create one issue using GitHub CLI."""

    args = [
        "gh",
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        issue.title,
        "--body",
        issue.body,
    ]
    for label in issue.labels:
        args.extend(["--label", label])

    completed = run_command(args)
    if completed.returncode != 0:
        raise ConfigError(f"Unable to create issue '{issue.title}': {completed.stderr.strip()}")
    print(f"CREATE {issue.issue_id}: {completed.stdout.strip()}")


def print_plan(issues: list[CuratedIssue], existing: dict[str, dict[str, Any]]) -> None:
    """Print the dry-run plan."""

    print("Curated public issue plan:")
    for issue in issues:
        current = existing.get(issue.title)
        if current:
            print(
                f"  SKIP   {issue.issue_id}: {issue.title} "
                f"(already exists as #{current.get('number')}, state={current.get('state')})"
            )
        else:
            print(f"  CREATE {issue.issue_id}: {issue.title} labels={issue.labels}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--wave", default="wave_1")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--allow-deferred", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the curated issue creation workflow."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = load_config(args.config)
        validate_safety(config)
        issues = select_issues(
            config,
            wave=args.wave,
            only=set(args.only),
            allow_deferred=args.allow_deferred,
        )
        if not issues:
            print("No issues selected.")
            return 0

        existing = load_existing_issue_titles(args.repo)
        print_plan(issues, existing)

        if not args.apply:
            print("\nDry-run only. Pass --apply to create missing GitHub issues.")
            return 0

        print("\nApplying changes:")
        for issue in issues:
            if issue.title in existing:
                print(f"SKIP   {issue.issue_id}: already exists")
                continue
            create_issue(args.repo, issue)
        return 0
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
