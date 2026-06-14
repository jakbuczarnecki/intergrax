# © Artur Czarnecki. All rights reserved.
"""Sync GitHub repository metadata from .github/repo-management/repository-metadata.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MANIFEST_PATH = SCRIPT_DIR / "repository-metadata.json"
MAX_DESCRIPTION_LENGTH = 350
MAX_TOPICS = 20
GITHUB_API = "https://api.github.com"
_GITHUB_REMOTE_RE = re.compile(
    r"(?:github\.com[/:]|github\.com%2F)(?P<owner>[^/]+)/(?P<repo>[^/.]+)"
)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = attribute_access.optional(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _load_dotenv_if_present() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


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

    manifest: dict[str, object] = {
        "description": description,
        "homepage": homepage,
        "topics": topics,
    }
    repository = payload.get("repository")
    if repository is not None:
        repository_text = str(repository).strip()
        if repository_text:
            manifest["repository"] = repository_text
    return manifest


def _print_plan(manifest: dict[str, object], manifest_path: Path) -> None:
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")
    print(f"Description ({len(str(manifest['description']))} chars):")
    print(f"  {manifest['description']}")
    print(f"Homepage: {manifest['homepage']}")
    print(f"Topics ({len(manifest['topics'])}):")
    for topic in manifest["topics"]:
        print(f"  - {topic}")


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _run_gh(args: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=True,
        input=input_bytes,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def _resolve_token() -> str | None:
    for key in ("GH_TOKEN", "GITHUB_TOKEN"):
        token = os.environ.get(key, "").strip()
        if token:
            return token
    if not _gh_available():
        return None
    try:
        result = _run_gh(["auth", "token"])
    except subprocess.CalledProcessError:
        return None
    token = result.stdout.strip()
    return token or None


def _parse_github_remote(url: str) -> str | None:
    match = _GITHUB_REMOTE_RE.search(url.strip())
    if not match:
        return None
    return f"{match.group('owner')}/{match.group('repo')}"


def _resolve_repository(manifest: dict[str, object]) -> str:
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if repository:
        return repository

    manifest_repo = manifest.get("repository")
    if isinstance(manifest_repo, str) and manifest_repo.strip():
        return manifest_repo.strip()

    if _gh_available():
        try:
            result = _run_gh(
                ["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"]
            )
            repository = result.stdout.strip()
            if repository:
                return repository
        except subprocess.CalledProcessError:
            pass

    git_result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if git_result.returncode == 0:
        repository = _parse_github_remote(git_result.stdout)
        if repository:
            return repository

    raise RuntimeError(
        "Could not resolve target repository. Set GITHUB_REPOSITORY, add "
        '"repository" to the manifest, or run from a git clone with origin remote.'
    )


def _github_request(
    *,
    method: str,
    path: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> None:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{GITHUB_API}{path}",
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "intergrax-metadata-sync",
            **({"Content-Type": "application/json"} if payload is not None else {}),
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"GitHub API {method} {path} failed ({exc.code}): {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc}") from exc


def _apply_via_api(manifest: dict[str, object], repository: str, token: str) -> None:
    _github_request(
        method="PATCH",
        path=f"/repos/{repository}",
        token=token,
        payload={
            "description": manifest["description"],
            "homepage": manifest["homepage"],
        },
    )
    _github_request(
        method="PUT",
        path=f"/repos/{repository}/topics",
        token=token,
        payload={"names": manifest["topics"]},
    )


def _apply_via_gh(manifest: dict[str, object], repository: str) -> None:
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
    topics_payload = json.dumps({"names": manifest["topics"]}).encode("utf-8")
    _run_gh(
        [
            "api",
            "--method",
            "PUT",
            f"repos/{repository}/topics",
            "--input",
            "-",
        ],
        input_bytes=topics_payload,
    )


def _apply(manifest: dict[str, object], repository: str) -> None:
    token = _resolve_token()
    if token:
        _apply_via_api(manifest, repository, token)
        return
    if _gh_available():
        _apply_via_gh(manifest, repository)
        return
    raise RuntimeError(
        "No GitHub credentials found. Add GH_TOKEN to .env (see "
        ".github/repo-management/README.md), run `gh auth login`, or export "
        "GH_TOKEN / GITHUB_TOKEN with repo scope."
    )


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    _load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        description="Validate or sync GitHub repository metadata manifest."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply manifest to GitHub (gh CLI or GH_TOKEN / GITHUB_TOKEN).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to repository metadata manifest.",
    )
    args = parser.parse_args(argv)

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    _print_plan(manifest, manifest_path)

    if not args.apply:
        print(
            "\nDry run only. To sync to GitHub run:\n"
            "  .github\\repo-management\\sync-github-metadata.bat         (Windows)\n"
            "  ./.github/repo-management/sync-github-metadata.sh        (Linux/macOS)\n"
            "  or: uv run python .github/repo-management/sync_github_repository_metadata.py --apply"
        )
        return 0

    repository = _resolve_repository(manifest)
    _apply(manifest, repository)
    print(f"\nSynced metadata to {repository}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        message = stderr or f"gh command failed with exit code {exc.returncode}"
        print(message, file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
