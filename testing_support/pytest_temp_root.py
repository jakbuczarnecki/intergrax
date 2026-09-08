# © Artur Czarnecki. All rights reserved.

"""Invocation-scoped pytest basetemp resolution for deterministic temp isolation."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

PYTEST_TEMP_NAMESPACE = "build/pytest"
LEGACY_SHARED_BASETEMP_MARKER = "build/pytest-basetemp"
LEGACY_SHARED_CACHE_MARKER = "build/pytest-cache"
_SAFE_INVOCATION_ID_RE = re.compile(r"^[\w.-]+$")

_INVOCATION_ID: str | None = None
_INVOCATION_BASETEMP: Path | None = None


def reset_invocation_basetemp_cache() -> None:
    """Reset per-process basetemp cache (test infrastructure only)."""
    global _INVOCATION_BASETEMP, _INVOCATION_ID
    _INVOCATION_BASETEMP = None
    _INVOCATION_ID = None


def _safe_invocation_id() -> str:
    invocation_id = f"{os.getpid()}-{os.urandom(4).hex()}"
    if not _SAFE_INVOCATION_ID_RE.fullmatch(invocation_id):
        raise ValueError(f"generated pytest invocation id is not path-safe: {invocation_id}")
    return invocation_id


def _invocation_id_for_process() -> str:
    global _INVOCATION_ID
    if _INVOCATION_ID is None:
        _INVOCATION_ID = _safe_invocation_id()
    return _INVOCATION_ID


def explicit_cli_basetemp(invocation_args: Sequence[str]) -> str | None:
    """Return basetemp path from explicit CLI args, or None when absent."""
    args = list(invocation_args)
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--basetemp":
            if index + 1 >= len(args):
                return None
            return args[index + 1]
        if arg.startswith("--basetemp="):
            return arg.partition("=")[2]
        index += 1
    return None


def allocate_invocation_pytest_basetemp(repo_root: Path) -> Path:
    """Allocate a unique basetemp directory for one pytest invocation."""
    namespace = repo_root / PYTEST_TEMP_NAMESPACE
    namespace.mkdir(parents=True, exist_ok=True)
    basetemp = namespace / _safe_invocation_id()
    basetemp.mkdir(parents=True, exist_ok=False)
    return basetemp


def invocation_basetemp_path(repo_root: Path) -> Path:
    """Return the process-scoped basetemp path, creating it when needed."""
    namespace = repo_root / PYTEST_TEMP_NAMESPACE
    namespace.mkdir(parents=True, exist_ok=True)
    basetemp = namespace / _invocation_id_for_process()
    basetemp.mkdir(parents=True, exist_ok=True)
    return basetemp


def invocation_cache_dir(repo_root: Path) -> Path:
    """Return the process-scoped pytest cache directory, creating it when needed."""
    cache_dir = invocation_basetemp_path(repo_root) / "pytest-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_or_allocate_invocation_basetemp(repo_root: Path) -> Path:
    """Return the invocation basetemp, allocating once per process."""
    global _INVOCATION_BASETEMP
    if _INVOCATION_BASETEMP is None:
        _INVOCATION_BASETEMP = invocation_basetemp_path(repo_root)
    return _INVOCATION_BASETEMP


def apply_invocation_pytest_cache_dir(early_config: pytest.Config, repo_root: Path) -> Path:
    """Assign invocation-owned pytest cache directory for the current process."""
    cache_dir = invocation_cache_dir(repo_root)
    relative = cache_dir.relative_to(repo_root).as_posix()
    early_config._inicache["cache_dir"] = relative
    early_config.inicfg["cache_dir"] = relative
    return cache_dir


def apply_invocation_pytest_basetemp(config: pytest.Config, repo_root: Path) -> Path | None:
    """Set invocation-owned basetemp unless the operator passed one on the CLI."""
    if explicit_cli_basetemp(config.invocation_params.args) is not None:
        return None
    basetemp = get_or_allocate_invocation_basetemp(repo_root)
    config.option.basetemp = str(basetemp)
    return basetemp


def resolved_pytest_basetemp(config: pytest.Config) -> Path:
    """Return the effective basetemp path for the current pytest invocation."""
    basetemp = config.option.basetemp
    if basetemp is None:
        raise RuntimeError("pytest basetemp is not configured")
    return Path(str(basetemp))
