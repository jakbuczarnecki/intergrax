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
import re
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
DEFAULT_CONFIG_PATH = REPO_ROOT / "docs" / "project" / "maintainers" / "public-adoption" / "curated_public_issues.yml"
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


def normalize_milestone_title(title: str | None) -> str:
    """Normalize milestone titles for robust matching.

    GitHub UI copy/paste can easily turn an em dash into an en dash or ASCII
    dash, and manual creation may add extra spaces. The canonical docs keep the
    typographic em dash, but matching should be tolerant.
    """

    if not title:
        return ""
    normalized = title.strip()
    normalized = normalized.replace("—", "-").replace("–", "-").replace("−", "-")
    normalized = re.sub(r"\s*-\s*", " - ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


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
    """Load repository .env values in deterministic order."""

    values: dict[str, str] = {}
    for dotenv_path in DOTENV_CANDIDATES:
        values.update(load_dotenv_values(dotenv_path))
    return values


def github_cli_environment() -> dict[str, str]:
    """Build subprocess environment for GitHub CLI."""

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
    """Return existing milestones by normalized title."""

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
            result[normalize_milestone_title(item["title"])] = item
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


def find_milestone(milestones: dict[str, dict[str, Any]], title: str) -> dict[str, Any] | None:
    """Find a milestone by tolerant title matching."""

    return milestones.get(normalize_milestone_title(title))


def resolved_milestone_title(milestones: dict[str, dict[str, Any]], title: str) -> str:
    """Return the actual GitHub milestone title, falling back to canonical title."""

    milestone = find_milestone(milestones, title)
    if isinstance(milestone, dict) and isinstance(milestone.get("title"), str):
        return milestone["title"]
    return title


def print_existing_milestones(milestones: dict[str, dict[str, Any]]) -> None:
    """Print milestones visible to the GitHub API."""

    print("Milestones visible to GitHub API:")
    if not milestones:
        print("  <none>")
        return
    for item in sorted(milestones.values(), key=lambda raw: str(raw.get("title", ""))):
        title = item.get("title", "<missing title>")
        state = item.get("state", "<unknown state>")
        number = item.get("number", "<unknown number>")
        print(f"  #{number}: {title} state={state}")


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
            existing = find_milestone(milestones, title)
            if existing is not None:
                print(f"  SKIP   milestone: {existing.get('title', title)} already exists")
            else:
                print(f"  CREATE milestone: {title}")

    for target in targets:
        current = issues.get(target.number)
        if current is None:
            print(f"  MISSING issue #{target.number}: {target.title}")
            continue

        expected_title = resolved_milestone_title(milestones, target.milestone_title)
        current_milestone = milestone_title(current)
        if normalize_milestone_title(current_milestone) == normalize_milestone_title(target.milestone_title):
            print(f"  SKIP   issue #{target.number}: already in {current_milestone}")
        else:
            print(f"  ASSIGN issue #{target.number}: {current_milestone or '<none>'} -> {expected_title}")


def missing_required_milestones(
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
) -> list[str]:
    """Return expected milestone titles that do not exist yet."""

    expected = sorted({target.milestone_title for target in targets})
    return [title for title in expected if find_milestone(milestones, title) is None]


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
        if find_milestone(milestones, title) is not None:
            existing_title = resolved_milestone_title(milestones, title)
            print(f"SKIP   milestone: {existing_title} already exists")
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
        milestones[normalize_milestone_title(title)] = {"title": title}


def assign_issue(repo: str, target: IssueMilestoneTarget, milestone_title_to_use: str) -> None:
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
            milestone_title_to_use,
        ]
    )
    if completed.returncode != 0:
        raise ConfigError(
            f"Unable to assign issue #{target.number} to milestone '{milestone_title_to_use}': "
            f"{completed.stderr.strip()}"
        )
    print(f"ASSIGN issue #{target.number}: {milestone_title_to_use}")


def assign_issues(
    repo: str,
    targets: list[IssueMilestoneTarget],
    milestones: dict[str, dict[str, Any]],
    issues: dict[int, dict[str, Any]],
) -> None:
    """Assign all selected issues to expected milestones."""

    for target in targets:
        current = issues.get(target.number)
        if current is None:
            raise ConfigError(f"Issue #{target.number} not found")
        current_title = milestone_title(current)
        if normalize_milestone_title(current_title) == normalize_milestone_title(target.milestone_title):
            print(f"SKIP   issue #{target.number}: already in {current_title}")
            continue
        assign_issue(repo, target, resolved_milestone_title(milestones, target.milestone_title))


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
        if normalize_milestone_title(current_milestone) != normalize_milestone_title(target.milestone_title):
            print(
                f"  MISMATCH issue #{target.number}: "
                f"milestone={current_milestone or '<none>'} expected={target.milestone_title}"
            )
            mismatches += 1
        else:
            print(f"  OK      issue #{target.number}: {current_milestone}")

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
    parser.add_argument(
        "--list-milestones",
        action="store_true",
        help="List milestones visible to GitHub API and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run milestone management workflow."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.apply and args.assign_only:
            raise ConfigError("Use either --apply or --assign-only, not both.")

        milestones = list_milestones(args.repo)
        if args.list_milestones:
            print_existing_milestones(milestones)
            return 0

        config = load_config(args.config)
        targets = select_targets(config, wave=args.wave)
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
            assign_issues(args.repo, targets, milestones, issues)
            return 0

        print("\nApplying milestone changes:")
        create_missing_milestones(args.repo, targets, milestones)
        refreshed_milestones = list_milestones(args.repo)
        require_existing_milestones(targets, refreshed_milestones)
        refreshed_issues = list_issues(args.repo)
        assign_issues(args.repo, targets, refreshed_milestones, refreshed_issues)
        return 0
    except (ConfigError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
