# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
#!/usr/bin/env python3

"""Create and verify maintainer-curated public GitHub issues.

The YAML file is the source of truth. The script uses GitHub CLI (`gh`),
skips issues whose exact title already exists, and creates missing GitHub Issues
only when --apply is provided.

When --wave is omitted, all waves in the YAML are processed.
"""

from __future__ import annotations

import argparse
import json
import os
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
DOTENV_CANDIDATES = (REPO_ROOT / ".env", REPO_ROOT / ".env.local")


@dataclass(frozen=True)
class CuratedIssue:
    """Validated issue definition loaded from YAML."""

    issue_id: str
    wave: str
    order: int
    title: str
    body: str
    labels: list[str]
    github_issue_number: int | None
    github_issue_url: str | None


class ConfigError(ValueError):
    """Raised when the YAML configuration is invalid."""


def parse_dotenv_line(line: str) -> tuple[str, str] | None:
    """Parse one simple KEY=VALUE .env line."""

    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def load_dotenv_values(path: Path) -> dict[str, str]:
    """Load simple .env key/value pairs without requiring python-dotenv."""

    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_dotenv_line(line)
        if parsed is None:
            continue
        key, value = parsed
        values[key] = value
    return values


def github_cli_environment() -> dict[str, str]:
    """Build subprocess environment for GitHub CLI.

    GitHub CLI accepts GH_TOKEN. Many project .env files use GITHUB_TOKEN.
    This function loads .env/.env.local and mirrors GITHUB_TOKEN to GH_TOKEN
    when GH_TOKEN is not already set.
    """

    env = os.environ.copy()
    for dotenv_path in DOTENV_CANDIDATES:
        for key, value in load_dotenv_values(dotenv_path).items():
            env.setdefault(key, value)

    if not env.get("GH_TOKEN") and env.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = env["GITHUB_TOKEN"]

    return env


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and capture output."""

    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=github_cli_environment(),
    )


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


def wave_names(config: dict[str, Any]) -> list[str]:
    """Return all wave names from the YAML config."""

    waves = config.get("waves")
    if not isinstance(waves, dict):
        raise ConfigError("Missing waves section")
    return [str(key) for key in waves.keys()]


def select_issues(
    config: dict[str, Any], *, wave: str, only: set[str], allow_deferred: bool
) -> list[CuratedIssue]:
    """Select issues for the requested wave."""

    waves = config.get("waves")
    if not isinstance(waves, dict):
        raise ConfigError("Missing waves section")

    wave_config = waves.get(wave)
    if not isinstance(wave_config, dict):
        available = ", ".join(wave_names(config))
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
                wave=wave,
                order=order,
                title=require_str(raw, "title"),
                body=require_str(raw, "body"),
                labels=labels,
                github_issue_number=optional_int(raw, "github_issue_number"),
                github_issue_url=optional_str(raw, "github_issue_url"),
            )
        )

    return sorted(selected, key=lambda item: (item.order, item.issue_id))


def select_all_issues(config: dict[str, Any], *, only: set[str]) -> list[CuratedIssue]:
    """Select all issues from all YAML waves."""

    selected: list[CuratedIssue] = []
    for wave in wave_names(config):
        selected.extend(select_issues(config, wave=wave, only=only, allow_deferred=True))

    if only:
        selected_ids = {issue.issue_id for issue in selected}
        missing = sorted(only - selected_ids)
        if missing:
            raise ConfigError(f"Requested issue IDs not found: {', '.join(missing)}")

    return sorted(selected, key=lambda item: (item.wave, item.order, item.issue_id))


def select_requested_issues(config: dict[str, Any], *, wave: str | None, only: set[str]) -> list[CuratedIssue]:
    """Select either one wave or all waves."""

    if wave:
        selected = select_issues(config, wave=wave, only=only, allow_deferred=True)
        if only:
            selected_ids = {issue.issue_id for issue in selected}
            missing = sorted(only - selected_ids)
            if missing:
                raise ConfigError(f"Requested issue IDs not found in wave '{wave}': {', '.join(missing)}")
        return selected
    return select_all_issues(config, only=only)


def require_str(raw: dict[str, Any], field: str) -> str:
    """Return a required non-empty string field."""

    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        issue_id = raw.get("id", "<unknown>")
        raise ConfigError(f"Issue '{issue_id}' must define non-empty string field '{field}'")
    return value


def optional_str(raw: dict[str, Any], field: str) -> str | None:
    """Return an optional non-empty string field."""

    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        issue_id = raw.get("id", "<unknown>")
        raise ConfigError(f"Issue '{issue_id}' optional field '{field}' must be a string")
    return value


def optional_int(raw: dict[str, Any], field: str) -> int | None:
    """Return an optional integer field."""

    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, int):
        issue_id = raw.get("id", "<unknown>")
        raise ConfigError(f"Issue '{issue_id}' optional field '{field}' must be an integer")
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
            "number,title,state,url,labels",
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
    print(f"CREATE {issue.wave}/{issue.issue_id}: {completed.stdout.strip()}")


def print_plan(issues: list[CuratedIssue], existing: dict[str, dict[str, Any]]) -> None:
    """Print the dry-run plan."""

    print("Curated public issue plan:")
    for issue in issues:
        current = existing.get(issue.title)
        prefix = f"{issue.wave}/{issue.issue_id}"
        if current:
            print(
                f"  SKIP   {prefix}: {issue.title} "
                f"(already exists as #{current.get('number')}, state={current.get('state')})"
            )
        else:
            print(f"  CREATE {prefix}: {issue.title} labels={issue.labels}")


def check_sync(issues: list[CuratedIssue], existing: dict[str, dict[str, Any]]) -> int:
    """Check whether YAML issue metadata matches GitHub issue state."""

    print("Curated public issue sync check:")
    mismatches = 0

    for issue in issues:
        current = existing.get(issue.title)
        if not current:
            print(f"  MISSING {issue.wave}/{issue.issue_id}: {issue.title}")
            mismatches += 1
            continue

        problems: list[str] = []
        current_number = current.get("number")
        current_url = current.get("url")
        current_state = str(current.get("state", "")).lower()
        current_labels = label_names(current.get("labels"))

        if issue.github_issue_number is not None and current_number != issue.github_issue_number:
            problems.append(f"number yaml={issue.github_issue_number} github={current_number}")
        if issue.github_issue_url is not None and current_url != issue.github_issue_url:
            problems.append(f"url yaml={issue.github_issue_url} github={current_url}")
        if sorted(issue.labels) != sorted(current_labels):
            problems.append(f"labels yaml={sorted(issue.labels)} github={sorted(current_labels)}")
        if current_state != "open":
            problems.append(f"state={current.get('state')}")

        if problems:
            print(f"  MISMATCH {issue.wave}/{issue.issue_id}: {issue.title} :: {'; '.join(problems)}")
            mismatches += 1
        else:
            print(f"  OK      {issue.wave}/{issue.issue_id}: #{current_number} {issue.title}")

    if mismatches:
        print(f"\nSync check failed: {mismatches} issue(s) need attention.")
        return 1

    print("\nSync check passed: YAML and GitHub issues are aligned.")
    return 0


def label_names(raw_labels: Any) -> list[str]:
    """Normalize GitHub CLI label output into a list of label names."""

    if not isinstance(raw_labels, list):
        return []

    names: list[str] = []
    for label in raw_labels:
        if isinstance(label, str):
            names.append(label)
        elif isinstance(label, dict) and isinstance(label.get("name"), str):
            names.append(label["name"])
    return names


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument(
        "--wave",
        default=None,
        help="Optional wave key to process. When omitted, all waves in the YAML are processed.",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--allow-deferred", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--check-sync",
        action="store_true",
        help="Check whether YAML metadata matches existing GitHub issues. Never creates issues.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the curated issue creation or sync-check workflow."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = load_config(args.config)
        validate_safety(config)
        only = set(args.only)

        existing = load_existing_issue_titles(args.repo)
        issues = select_requested_issues(config, wave=args.wave, only=only)

        if not issues:
            print("No issues selected.")
            return 0

        if args.check_sync:
            return check_sync(issues, existing)

        print_plan(issues, existing)

        if not args.apply:
            print("\nDry-run only. Pass --apply to create missing GitHub issues.")
            return 0

        print("\nApplying changes:")
        for issue in issues:
            if issue.title in existing:
                print(f"SKIP   {issue.wave}/{issue.issue_id}: already exists")
                continue
            create_issue(args.repo, issue)
            existing[issue.title] = {"title": issue.title, "state": "open"}
        return 0
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
