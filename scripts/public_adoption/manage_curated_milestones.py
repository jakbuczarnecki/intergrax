# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.
#!/usr/bin/env python3

"""Create and verify milestones for maintainer-curated public GitHub issues.

The curated issue YAML remains the source of truth for waves and issue numbers.
This script maps each wave to a public-adoption milestone, creates missing
milestones when --apply is provided, and assigns issues to the expected wave
milestone.
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

MILESTONES_BY_WAVE: dict[str, dict[str, str]] = {
    "wave_1": {
        "title": "Public Adoption — Wave 1",
        "description": "First evaluator / proof-path feedback issues for the curated public discussion map.",
    },
    "wave_2": {
        "title": "Public Adoption — Wave 2",
        "description": "Architecture clarity and integration-surface feedback issues for the curated public discussion map.",
    },
    "wave_3": {
        "title": "Architecture Discussion — Wave 3",
        "description": "Core Harness AI / Agent OS architecture discussion issues.",
    },
    "wave_4": {
        "title": "Product Validation — Wave 4",
        "description": "Product and application validation issues for Intergrax public adoption.",
    },
    "wave_5": {
        "title": "Deep Technical Review — Wave 5",
        "description": "Advanced architecture, governance, reliability, security, observability, and developer-experience review issues.",
    },
}


@dataclass(frozen=True)
class IssueMilestoneTarget:
    """Expected GitHub issue milestone assignment."""

    wave: str
    issue_id: str
    number: int
    title: str
    milestone_title: str


class ConfigError(ValueError):
    """Raised when configuration or GitHub state is invalid."""


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


def load_all_dotenv_values() -> dict[str, str]:
    """Load repository .env values in deterministic order.

    Later files override earlier files. This makes .env.local able to override
    .env while keeping the project-local configuration explicit.
    """

    values: dict[str, str] = {}
    for dotenv_path in DOTENV_CANDIDATES:
        values.update(load_dotenv_values(dotenv_path))
    return values


def github_cli_environment() -> dict[str, str]:
    """Build subprocess environment for GitHub CLI.

    Project .env/.env.local values are loaded first. For GitHub authentication,
    the repository-local GH_TOKEN or GITHUB_TOKEN intentionally overrides any
    stale token inherited from the parent shell. GitHub CLI uses GH_TOKEN.
    """

    env = os.environ.copy()
    dotenv_values = load_all_dotenv_values()

    for key, value in dotenv_values.items():
        env.setdefault(key, value)

    if dotenv_values.get("GH_TOKEN"):
        env["GH_TOKEN"] = dotenv_values["GH_TOKEN"]
    elif dotenv_values.get("GITHUB_TOKEN"):
        env["GH_TOKEN"] = dotenv_values["GITHUB_TOKEN"]
    elif not env.get("GH_TOKEN") and env.get("GITHUB_TOKEN"):
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


def run_json_command(args: list[str]) -> Any:
    """Run a subprocess expected to return JSON."""

    completed = run_command(args)
    if completed.returncode != 0:
        raise ConfigError(completed.stderr.strip() or f"Command failed: {' '.join(args)}")
    return json.loads(completed.stdout or "[]")


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration."""

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("Config root must be a mapping")
    return loaded


def select_targets(config: dict[str, Any], *, wave: str | None) -> list[IssueMilestoneTarget]:
    """Select milestone assignments from YAML waves."""

    waves = config.get("waves")
    if not isinstance(waves, dict):
        raise ConfigError("Missing waves section")

    selected_waves = [wave] if wave else list(waves.keys())
    targets: list[IssueMilestoneTarget] = []

    for wave_name in selected_waves:
        if wave_name not in waves:
            available = ", ".join(str(key) for key in waves.keys())
            raise ConfigError(f"Unknown wave '{wave_name}'. Available waves: {available}")
        if wave_name not in MILESTONES_BY_WAVE:
            raise ConfigError(f"No milestone mapping configured for wave '{wave_name}'")

        wave_config = waves[wave_name]
        if not isinstance(wave_config, dict):
            raise ConfigError(f"Wave '{wave_name}' must be a mapping")
        raw_issues = wave_config.get("issues")
        if not isinstance(raw_issues, list):
            raise ConfigError(f"Wave '{wave_name}' must contain an issues list")

        milestone_title = MILESTONES_BY_WAVE[wave_name]["title"]
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                raise ConfigError(f"Invalid issue entry in wave '{wave_name}'")
            issue_number = raw_issue.get("github_issue_number")
            if not isinstance(issue_number, int):
                issue_id = raw_issue.get("id", "<unknown>")
                raise ConfigError(f"Issue '{issue_id}' in wave '{wave_name}' has no github_issue_number")
            title = raw_issue.get("title")
            if not isinstance(title, str) or not title.strip():
                issue_id = raw_issue.get("id", "<unknown>")
                raise ConfigError(f"Issue '{issue_id}' in wave '{wave_name}' has no title")
            issue_id = raw_issue.get("id")
            if not isinstance(issue_id, str) or not issue_id.strip():
                raise ConfigError(f"Issue #{issue_number} in wave '{wave_name}' has no id")

            targets.append(
                IssueMilestoneTarget(
                    wave=str(wave_name),
                    issue_id=issue_id,
                    number=issue_number,
                    title=title,
                    milestone_title=milestone_title,
                )
            )

    return sorted(targets, key=lambda item: (item.wave, item.number))


def list_milestones(repo: str) -> dict[str, dict[str, Any]]:
    """Return existing milestones by title."""

    loaded = run_json_command(
        [
            "gh",
            "api",
            f"repos/{repo}/milestones",
            "--method",
            "GET",
            "--paginate",
            "-f",
            "state=all",
        ]
    )
    if not isinstance(loaded, list):
        raise ConfigError("Unexpected milestone list response")

    result: dict[str, dict[str, Any]] = {}
    for item in loaded:
        if isinstance(item, dict) and isinstance(item.get("title"), str):
            result[item["title"]] = item
    return result


def list_issues(repo: str) -> dict[int, dict[str, Any]]:
    """Return existing issues by number."""

    loaded = run_json_command(
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
            "number,title,milestone,state",
        ]
    )
    if not isinstance(loaded, list):
        raise ConfigError("Unexpected issue list response")

    result: dict[int, dict[str, Any]] = {}
    for item in loaded:
        if isinstance(item, dict) and isinstance(item.get("number"), int):
            result[item["number"]] = item
    return result


def milestone_title(raw_issue: dict[str, Any]) -> str | None:
    """Extract milestone title from GitHub CLI issue JSON."""

    raw = raw_issue.get("milestone")
    if isinstance(raw, dict) and isinstance(raw.get("title"), str):
        return raw["title"]
    return None


def print_plan(
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
    issues: dict[int, dict[str, Any]],
) -> None:
    """Print dry-run plan."""

    print("Curated public milestone plan:")

    for wave_name, milestone_config in MILESTONES_BY_WAVE.items():
        if any(target.wave == wave_name for target in targets):
            title = milestone_config["title"]
            if title in milestones:
                print(f"  SKIP   milestone: {title} already exists")
            else:
                print(f"  CREATE milestone: {title}")

    for target in targets:
        current = issues.get(target.number)
        if current is None:
            print(f"  MISSING issue #{target.number}: {target.title}")
            continue

        current_milestone = milestone_title(current)
        if current_milestone == target.milestone_title:
            print(f"  SKIP   issue #{target.number}: already in {target.milestone_title}")
        else:
            print(
                f"  ASSIGN issue #{target.number}: {current_milestone or '<none>'} -> {target.milestone_title}"
            )


def missing_required_milestones(
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
) -> list[str]:
    """Return expected milestone titles that do not exist yet."""

    expected = sorted({target.milestone_title for target in targets})
    return [title for title in expected if title not in milestones]


def require_existing_milestones(
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
) -> None:
    """Fail if any expected milestone is missing."""

    missing = missing_required_milestones(targets, milestones)
    if missing:
        formatted = ", ".join(missing)
        raise ConfigError(
            "Missing required milestone(s): "
            f"{formatted}. Create them manually in GitHub or run with --apply using a token that can create milestones."
        )


def create_missing_milestones(
    repo: str,
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
) -> None:
    """Create missing milestones for selected waves."""

    selected_waves = {target.wave for target in targets}
    for wave_name, milestone_config in MILESTONES_BY_WAVE.items():
        if wave_name not in selected_waves:
            continue
        title = milestone_config["title"]
        if title in milestones:
            print(f"SKIP   milestone: {title} already exists")
            continue

        completed = run_command(
            [
                "gh",
                "api",
                f"repos/{repo}/milestones",
                "-X",
                "POST",
                "-f",
                f"title={title}",
                "-f",
                f"description={milestone_config['description']}",
            ]
        )
        if completed.returncode != 0:
            raise ConfigError(f"Unable to create milestone '{title}': {completed.stderr.strip()}")
        print(f"CREATE milestone: {title}")
        milestones[title] = {"title": title}


def assign_issue(repo: str, target: IssueMilestoneTarget) -> None:
    """Assign one issue to its milestone."""

    completed = run_command(
        [
            "gh",
            "issue",
            "edit",
            str(target.number),
            "--repo",
            repo,
            "--milestone",
            target.milestone_title,
        ]
    )
    if completed.returncode != 0:
        raise ConfigError(
            f"Unable to assign issue #{target.number} to milestone '{target.milestone_title}': "
            f"{completed.stderr.strip()}"
        )
    print(f"ASSIGN issue #{target.number}: {target.milestone_title}")


def assign_issues(
    repo: str,
    targets: list[IssueMilestoneTarget],
    issues: dict[int, dict[str, Any]],
) -> None:
    """Assign all selected issues to expected milestones."""

    for target in targets:
        current = issues.get(target.number)
        if current is None:
            raise ConfigError(f"Issue #{target.number} not found")
        if milestone_title(current) == target.milestone_title:
            print(f"SKIP   issue #{target.number}: already in {target.milestone_title}")
            continue
        assign_issue(repo, target)


def check_sync(targets: list[IssueMilestoneTarget], issues: dict[int, dict[str, Any]]) -> int:
    """Check milestone assignments."""

    print("Curated public milestone sync check:")
    mismatches = 0

    for target in targets:
        current = issues.get(target.number)
        if current is None:
            print(f"  MISSING issue #{target.number}: {target.title}")
            mismatches += 1
            continue

        current_milestone = milestone_title(current)
        if current_milestone != target.milestone_title:
            print(
                f"  MISMATCH issue #{target.number}: "
                f"milestone={current_milestone or '<none>'} expected={target.milestone_title}"
            )
            mismatches += 1
        else:
            print(f"  OK      issue #{target.number}: {target.milestone_title}")

    if mismatches:
        print(f"\nMilestone sync check failed: {mismatches} issue(s) need attention.")
        return 1

    print("\nMilestone sync check passed: curated public issues are assigned to expected milestones.")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--wave", default=None, help="Optional wave key to process. Defaults to all waves.")
    parser.add_argument("--apply", action="store_true", help="Create missing milestones and assign issues.")
    parser.add_argument(
        "--assign-only",
        action="store_true",
        help="Assign issues to existing milestones only. Never creates milestones.",
    )
    parser.add_argument("--check-sync", action="store_true", help="Verify milestone assignments. Never mutates GitHub.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run milestone management workflow."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.apply and args.assign_only:
            raise ConfigError("Use either --apply or --assign-only, not both.")

        config = load_config(args.config)
        targets = select_targets(config, wave=args.wave)
        milestones = list_milestones(args.repo)
        issues = list_issues(args.repo)

        if args.check_sync:
            return check_sync(targets, issues)

        print_plan(targets, milestones, issues)

        if not args.apply and not args.assign_only:
            print("\nDry-run only. Pass --apply to create milestones and assign issues, or --assign-only to assign existing milestones only.")
            return 0

        if args.assign_only:
            print("\nApplying issue assignments only:")
            require_existing_milestones(targets, milestones)
            assign_issues(args.repo, targets, issues)
            return 0

        print("\nApplying milestone changes:")
        create_missing_milestones(args.repo, targets, milestones)
        refreshed_milestones = list_milestones(args.repo)
        require_existing_milestones(targets, refreshed_milestones)
        refreshed_issues = list_issues(args.repo)
        assign_issues(args.repo, targets, refreshed_issues)
        return 0
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
