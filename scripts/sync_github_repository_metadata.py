# © Artur Czarnecki. All rights reserved.
"""Sync GitHub repository metadata from .github/repository-metadata.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / ".github" / "repository-metadata.json"
MAX_DESCRIPTION_LENGTH = 350
MAX_TOPICS = 20


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    description = str(payload["description"]).strip()
    homepage = str(payload["homepage"]).strip()
    topics = [str(topic).strip().lower() for topic in payload["topics"]]

    if not description:
        raise ValueError("description must not be empty")
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(
            f"description exceeds GitHub limit ({len(description)}/{MAX_DESCRIPTION_LENGTH})"
        )
    if not homepage:
        raise ValueError("homepage must not be empty")
    if not topics:
        raise ValueError("topics must contain at least one entry")
    if len(topics) > MAX_TOPICS:
        raise ValueError(f"topics exceed GitHub limit ({len(topics)}/{MAX_TOPICS})")
    if len(set(topics)) != len(topics):
        raise ValueError("topics must be unique")

    return {
        "description": description,
        "homepage": homepage,
        "topics": topics,
    }


def _run_gh(args: list[str], *, input_bytes: bytes | None = None) -> None:
    subprocess.run(
        ["gh", *args],
        check=True,
        input=input_bytes,
    )


def _print_plan(manifest: dict[str, object]) -> None:
    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print(f"Description ({len(manifest['description'])} chars):")
    print(f"  {manifest['description']}")
    print(f"Homepage: {manifest['homepage']}")
    print(f"Topics ({len(manifest['topics'])}):")
    for topic in manifest["topics"]:
        print(f"  - {topic}")


def _apply(manifest: dict[str, object], repository: str) -> None:
    _run_gh(
        [
            "repo",
            "edit",
            repository,
            "--description",
            str(manifest["description"]),
            "--homepage",
            str(manifest["homepage"]),
        ]
    )
    topics_payload = json.dumps({"names": manifest["topics"]})
    _run_gh(
        [
            "api",
            "--method",
            "PUT",
            f"repos/{repository}/topics",
            "--input",
            "-",
        ],
        input_bytes=topics_payload.encode("utf-8"),
    )


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        description="Validate or sync GitHub repository metadata manifest."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply manifest to GitHub via gh CLI (requires GH_TOKEN or gh auth).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to repository metadata manifest.",
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest.resolve())
    _print_plan(manifest)

    if not args.apply:
        print(
            "\nDry run only. To sync to GitHub run:\n"
            "  sync-github-metadata.bat apply   (Windows)\n"
            "  ./sync-github-metadata.sh apply  (Linux/macOS)\n"
            "  or: uv run python scripts/sync_github_repository_metadata.py --apply"
        )
        return 0

    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            check=True,
            capture_output=True,
            text=True,
        )
        repository = result.stdout.strip()
    if not repository:
        raise RuntimeError("Could not resolve target repository")

    _apply(manifest, repository)
    print(f"\nSynced metadata to {repository}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"gh command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
