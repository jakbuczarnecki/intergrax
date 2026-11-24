#!/usr/bin/env python3
"""Cursor-only candidate publication. No merge or certification operation.

The local checkout is not an authorization source. Read the per-wave record
from canonical development and the approved ticket from the immutable base.
Server-side permissions and Git hooks continue to apply.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

from igx_integrator import (
    REPO,
    SHA,
    TICKET,
    Blocked,
    command,
    gh_api,
    list_pr_files,
    local_worktree,
    valid_path,
)

WAVE = re.compile(r"^[A-Z0-9][A-Z0-9-]{0,63}$")
WORKER = re.compile(r"^(?:ec-v[0-9]+|af2|feat|fix)/[A-Za-z0-9][A-Za-z0-9._/-]{0,100}$")
TARGETS = {"development", "arch-freeze/rw-train"}
REMOTE = re.compile(
    r"^(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
    r"jakbuczarnecki/intergrax(?:\.git)?/?$",
    re.IGNORECASE,
)


def validate_authorization(p: Any, wave: str) -> None:
    if not isinstance(p, dict) or type(p.get("schema")) is not int or p["schema"] != 1 or p.get("wave") != wave:
        raise Blocked("invalid candidate authorization schema or wave")
    if p.get("mode") != "CHATGPT_DIRECT_IMPLEMENTATION":
        raise Blocked("candidate publication not authorized")
    if p.get("target") not in TARGETS:
        raise Blocked("target outside candidate publication scope")
    kind = p.get("kind")
    if kind not in {"product", "docs"} or (p["target"] == "arch-freeze/rw-train" and kind != "product"):
        raise Blocked("target/kind outside candidate publication scope")
    branch = p.get("branch")
    if not isinstance(branch, str) or not WORKER.fullmatch(branch):
        raise Blocked("invalid candidate branch")
    if any(x in {"", ".", ".."} or x.startswith(".") or x.endswith(".lock") for x in branch.split("/")):
        raise Blocked("invalid candidate ref component")
    if ".." in branch or branch.endswith("."):
        raise Blocked("invalid candidate ref")
    for field in ("base_sha",):
        if not isinstance(p.get(field), str) or not SHA.fullmatch(p[field]):
            raise Blocked("invalid " + field)
    if "head_sha" in p and (not isinstance(p["head_sha"], str) or not SHA.fullmatch(p["head_sha"])):
        raise Blocked("invalid optional candidate head")
    if not isinstance(p.get("ticket"), str) or not TICKET.fullmatch(p["ticket"]):
        raise Blocked("invalid canonical ticket")
    paths = p.get("write_set")
    if not isinstance(paths, list) or not paths or not all(isinstance(x, str) for x in paths):
        raise Blocked("missing exact write set")
    if len(paths) != len({x.casefold() for x in paths}) or any(not valid_path(x, kind) for x in paths):
        raise Blocked("duplicate or protected write-set path")
    # Candidate publication must never carry approval, governance or workflow mutations.
    if any(
        not valid_path(x.casefold(), kind)
        or x.casefold().startswith(("governance/", ".github/"))
        or any(part.endswith((".", " ")) or ":" in part for part in x.split("/"))
        for x in paths
    ):
        raise Blocked("candidate publication cannot change governance or workflows")
    title = p.get("pr_title")
    if not isinstance(title, str) or not 8 <= len(title) <= 160 or "\n" in title or "\r" in title:
        raise Blocked("invalid PR title")


def read_authorization(wave: str) -> dict[str, Any]:
    if not WAVE.fullmatch(wave):
        raise Blocked("invalid wave id")
    data = gh_api(f"contents/.cursor/publication/{wave}.json?ref=development")
    if not isinstance(data, dict) or data.get("type") != "file" or data.get("encoding") != "base64":
        raise Blocked("canonical candidate authorization missing")
    try:
        p = json.loads(base64.b64decode("".join(data["content"].split()), validate=True).decode("utf-8"))
    except (KeyError, ValueError, UnicodeError) as exc:
        raise Blocked("invalid canonical candidate authorization") from exc
    validate_authorization(p, wave)
    return p


def git(checkout: Path, *args: str) -> str:
    return command(["git", *args], cwd=checkout)


def assert_scoped(paths: set[str], p: dict[str, Any]) -> None:
    if not paths or not paths.issubset(set(p["write_set"])):
        raise Blocked("candidate changes outside exact authorized write set")


def diff_paths(checkout: Path, base: str, head: str) -> set[str]:
    return set(filter(None, git(checkout, "diff", "--name-only", "--no-renames", "-z", base, head).split("\0")))


def verify_local(p: dict[str, Any], checkout: Path) -> str:
    local_worktree(checkout)
    if Path(git(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout.resolve():
        raise Blocked("run from the assigned worktree root")
    for flags in (("--all",), ("--push", "--all")):
        urls = git(checkout, "remote", "get-url", *flags, "origin").splitlines()
        if len(urls) != 1 or not REMOTE.fullmatch(urls[0]):
            raise Blocked("unexpected origin fetch or push URL")
    if git(checkout, "branch", "--show-current") != p["branch"]:
        raise Blocked("candidate branch differs from authorization")
    git(checkout, "check-ref-format", f"refs/heads/{p['branch']}")
    head = git(checkout, "rev-parse", "HEAD")
    if not SHA.fullmatch(head) or (p.get("head_sha") is not None and p["head_sha"] != head):
        raise Blocked("candidate head differs from authorization")
    if git(checkout, "status", "--porcelain"):
        raise Blocked("candidate worktree must be clean")
    base = p["base_sha"]
    git(checkout, "merge-base", "--is-ancestor", base, head)
    ticket = git(checkout, "show", f"{base}:{p['ticket']}")
    if not re.search(r"(?m)^Approved: (?!PENDING\b).+", ticket):
        raise Blocked("base ticket lacks approval")
    if not re.search(r"(?m)^## Lifecycle state\s*\n\s*(plan_approved|in_progress|quality_gate|ready_for_audit)\s*$", ticket):
        raise Blocked("base ticket does not authorize implementation")
    assert_scoped(diff_paths(checkout, base, head), p)
    commits = git(checkout, "rev-list", "--reverse", f"{base}..{head}").splitlines()
    if not commits or len(commits) > 100:
        raise Blocked("candidate commit range empty or exceeds bounded publication")
    parent = base
    for commit in commits:
        if git(checkout, "show", "-s", "--format=%P", commit) != parent:
            raise Blocked("candidate must be a linear isolated implementation history")
        assert_scoped(diff_paths(checkout, parent, commit), p)
        message = git(checkout, "show", "-s", "--format=%B", commit)
        if f"Ref: {p['ticket']}" not in message.splitlines():
            raise Blocked("candidate commit lacks canonical ticket trailer")
        parent = commit
    return head


def remote_sha(checkout: Path, branch: str) -> str | None:
    ref = f"refs/heads/{branch}"
    rows = git(checkout, "ls-remote", "--heads", "origin", ref).splitlines()
    if not rows:
        return None
    if len(rows) != 1:
        raise Blocked("ambiguous remote ref")
    sha, sep, name = rows[0].partition("\t")
    if not sep or name != ref or not SHA.fullmatch(sha):
        raise Blocked("invalid remote ref response")
    return sha


def verify_target(p: dict[str, Any], checkout: Path) -> None:
    target = p["target"]
    target_head = remote_sha(checkout, target)
    if target_head is None:
        raise Blocked("authorized target does not exist")
    git(checkout, "fetch", "origin", f"refs/heads/{target}")
    git(checkout, "merge-base", "--is-ancestor", p["base_sha"], target_head)


def open_prs(p: dict[str, Any]) -> list[dict[str, Any]]:
    result = json.loads(
        command(["gh", "pr", "list", "-R", REPO, "--head", p["branch"], "--state", "open",
                 "--json", "number", "--limit", "100"])
    )
    if not isinstance(result, list) or len(result) > 1 or any(not isinstance(x, dict) for x in result):
        raise Blocked("ambiguous open candidate PR")
    return result


def verify_draft(p: dict[str, Any], number: int, head: str | None = None) -> dict[str, Any]:
    pr = gh_api(f"pulls/{number}")
    if (
        not isinstance(pr, dict)
        or pr.get("state") != "open"
        or pr.get("draft") is not True
        or pr.get("merged")
        or pr.get("head", {}).get("ref") != p["branch"]
        or pr.get("base", {}).get("ref") != p["target"]
        or pr.get("head", {}).get("repo", {}).get("full_name") != REPO
        or pr.get("base", {}).get("repo", {}).get("full_name") != REPO
    ):
        raise Blocked("existing PR is not the authorized same-repository draft")
    if head is not None:
        if pr.get("head", {}).get("sha") != head:
            raise Blocked("published PR head differs from candidate")
        changed = set()
        for item in list_pr_files(number):
            changed.add(item.get("filename"))
            if item.get("previous_filename"):
                changed.add(item["previous_filename"])
        assert_scoped(changed, p)
    return pr


def publish(p: dict[str, Any], checkout: Path) -> dict[str, Any]:
    head = verify_local(p, checkout)
    verify_target(p, checkout)
    existing = open_prs(p)
    if existing:
        number = existing[0].get("number")
        if type(number) is not int or number <= 0:
            raise Blocked("invalid existing PR number")
        verify_draft(p, number)
    remote = remote_sha(checkout, p["branch"])
    if remote and remote != head:
        git(checkout, "fetch", "origin", f"refs/heads/{p['branch']}")
        git(checkout, "merge-base", "--is-ancestor", remote, head)
    if remote != head:
        git(checkout, "push", "--no-follow-tags", "--recurse-submodules=no", "origin",
            f"{head}:refs/heads/{p['branch']}")
    if remote_sha(checkout, p["branch"]) != head:
        raise Blocked("remote candidate head not confirmed")
    if not existing:
        command(["gh", "pr", "create", "-R", REPO, "--draft", "--base", p["target"], "--head", p["branch"],
                 "--title", p["pr_title"], "--body",
                 f"Candidate for external acceptance. Wave: {p['wave']}. Ref: {p['ticket']}."],
                cwd=checkout)
        created = open_prs(p)
        if len(created) != 1 or type(created[0].get("number")) is not int or created[0]["number"] <= 0:
            raise Blocked("candidate draft PR creation not confirmed")
        number = created[0]["number"]
    pr = verify_draft(p, number, head)
    return {"result": "PUBLISHED_DRAFT", "pr": number, "url": pr["html_url"], "head_sha": head}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True)
    args = parser.parse_args()
    try:
        p = read_authorization(args.wave)
        print(json.dumps(publish(p, Path.cwd())))
        return 0
    except (Blocked, ValueError, OSError, TypeError, KeyError, AttributeError) as exc:
        print(f"PUBLICATION_FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
