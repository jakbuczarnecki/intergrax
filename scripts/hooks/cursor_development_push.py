#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Preflight for Cursor-allowed ``git push origin development`` (fast-forward only)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REF_DEVELOPMENT = re.compile(r"^refs/heads/development$", re.IGNORECASE)
_SAFE_TAIL = re.compile(r"^\s*(?:-[A-Za-z][\w-]*(?:\s+=\s+[^\s;&|]+|\s+[^\s;&|-][^\s;&|]*)?)*\s*")
_ALLOWED_REMOTE = "origin"


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def push_target_is_allowed_development(push_tail: str) -> bool:
    """Return True when the refspec targets only ``origin`` → ``development`` (no force)."""
    tail = push_tail.strip()
    if not tail or tail.startswith("-") and not tail.startswith("--"):
        return False
    if re.search(r"(?i)(?:^|\s)(?:\+|--force\b|--force-with-lease\b)", tail):
        return False
    if re.search(r"(?i)\b(?:--all|--tags|--mirror)\b", tail):
        return False
    tokens = [t for t in re.split(r"\s+", _SAFE_TAIL.sub("", tail).strip()) if t]
    if not tokens or tokens[0].lower() != _ALLOWED_REMOTE:
        return False
    if len(tokens) == 1:
        return False
    refspecs = tokens[1:]
    for spec in refspecs:
        if spec.startswith(":") or spec.endswith(":"):
            return False
        if ":" in spec:
            _local, remote = spec.split(":", 1)
            remote = remote.strip()
            if remote and not _REF_DEVELOPMENT.fullmatch(remote):
                return False
            if _local.upper() == "HEAD" and (not remote or _REF_DEVELOPMENT.fullmatch(remote)):
                continue
            return False
        if spec.lower() != "development" and not _REF_DEVELOPMENT.fullmatch(spec):
            return False
    return True


def validate_development_push(repo: Path | None = None) -> str | None:
    """Return an error message to deny, or None when push preflight passes."""
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return None
    if _truthy_env("IGX_CURSOR_PUSH_PRECHECK_SKIP"):
        return None
    root = repo or _REPO
    sym = _run_git(root, "symbolic-ref", "HEAD")
    if sym.returncode != 0:
        return "Cannot resolve current branch (detached HEAD?)."
    branch = sym.stdout.strip()
    if branch != "refs/heads/development":
        return "Push is allowed only from branch development."

    status = _run_git(root, "status", "--porcelain")
    if status.returncode != 0:
        return "git status failed."
    if status.stdout.strip():
        return "Working tree must be clean before git push origin development."

    origin = _run_git(root, "rev-parse", "origin/development")
    if origin.returncode != 0:
        return "Missing origin/development; run git fetch origin development first."
    head = _run_git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        return "Cannot resolve HEAD."

    ancestor = _run_git(root, "merge-base", "--is-ancestor", origin.stdout.strip(), head.stdout.strip())
    if ancestor.returncode != 0:
        return (
            "Local development is not a fast-forward of origin/development. "
            "Fetch and reconcile — force push is not allowed."
        )

    ref_gate = subprocess.run(
        [sys.executable, str(root / "scripts/governance/verify_commit_ticket_refs.py"), "--mode", "push"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ref_gate.returncode != 0:
        detail = (ref_gate.stderr or ref_gate.stdout or "").strip().splitlines()
        hint = detail[-1] if detail else "verify_commit_ticket_refs failed"
        return f"Ticket Ref gate failed: {hint}"

    policy = subprocess.run(
        [sys.executable, str(root / "scripts/governance/validate_policy_registry.py")],
        cwd=root,
        check=False,
    )
    if policy.returncode != 0:
        return "Policy registry validation failed before push."
    return None
