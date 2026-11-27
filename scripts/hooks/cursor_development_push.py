#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Preflight for Cursor-allowed ``git push origin development`` (fast-forward only)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_ALLOWED_REFS = frozenset({"development", "refs/heads/development", "HEAD:refs/heads/development"})


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
    tokens = push_tail.split()
    # Only the documented, single-ref form. In particular, never allow --no-verify,
    # arbitrary options, multiple refspecs, or refspecs that change the destination.
    return len(tokens) == 2 and tokens[0] == "origin" and tokens[1] in _ALLOWED_REFS


def _uv_executable() -> str | None:
    executable = shutil.which("uv")
    if executable:
        return executable
    # Match the Windows Git pre-push hook's PATH recovery without changing global PATH.
    homes = [Path.home()]
    if os.environ.get("USERPROFILE"):
        homes.append(Path(os.environ["USERPROFILE"]))
    for home in homes:
        for name in ("uv.exe", "uv"):
            candidate = home / ".local" / "bin" / name
            if candidate.is_file():
                return str(candidate)
    return None


def _run_gate(root: Path, uv: str, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [uv, "run", "--no-sync", "--project", str(root), "python", str(root / script), *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _gate_failure(name: str, result: subprocess.CompletedProcess[str]) -> str:
    details = (result.stderr + "\n" + result.stdout).strip()
    return f"{name} failed (exit {result.returncode}): {details[-1200:] or 'no diagnostic output'}"


def validate_development_push(repo: Path | None = None) -> str | None:
    """Return an error message to deny, or None when push preflight passes."""
    if os.environ.get("IGX_HOOKS", "").strip().lower() == "off":
        return None
    if _truthy_env("IGX_CURSOR_PUSH_PRECHECK_SKIP"):
        return None
    checkout = repo or Path.cwd()
    try:
        top = _run_git(checkout, "rev-parse", "--show-toplevel")
        if top.returncode != 0 or not top.stdout.strip():
            return "Cursor push preflight: cannot resolve Git checkout from the shell working directory."
        root = Path(top.stdout.strip()).resolve()
        return _validate_checkout(root)
    except (OSError, ValueError) as exc:
        return f"Cursor push preflight could not execute: {exc}"


def _validate_checkout(root: Path) -> str | None:
    sym = _run_git(root, "symbolic-ref", "HEAD")
    if sym.returncode != 0:
        return "Cannot resolve current branch (detached HEAD?)."
    branch = sym.stdout.strip()
    if branch != "refs/heads/development":
        return "Push is allowed only from branch development."

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

    uv = _uv_executable()
    if uv is None:
        return "Cursor push preflight: uv not found on PATH or in the user .local/bin directory."
    # Use the same project environment as the actual Git pre-push hook; the hook
    # launcher itself may be a system Python without PyYAML or other dependencies.
    ref_gate = _run_gate(
        root,
        uv,
        "scripts/governance/verify_commit_ticket_refs.py",
        "--ci",
        "--mode",
        "range",
        "--rev-range",
        f"{origin.stdout.strip()}..{head.stdout.strip()}",
    )
    if ref_gate.returncode != 0:
        return _gate_failure("Ticket Ref gate", ref_gate)
    policy = _run_gate(root, uv, "scripts/governance/validate_policy_registry.py")
    if policy.returncode != 0:
        return _gate_failure("Policy registry validation", policy)
    return None
