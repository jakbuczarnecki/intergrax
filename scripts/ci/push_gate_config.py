#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Load [tool.intergrax.push_gate] from pyproject.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_VALID_PROFILES = frozenset({"fast", "smoke", "enterprise", "full", "all", "off"})


@dataclass(frozen=True)
class PushGateConfig:
    profile: str
    skip_env: str
    skip_uv_sync_on_push: bool
    pre_commit_ruff: bool
    diff_scoped_gates: bool
    skip_env_pre_commit: str


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_push_gate_config(pyproject: Path | None = None) -> PushGateConfig:
    path = pyproject or _PYPROJECT
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    section = data.get("tool", {}).get("intergrax", {}).get("push_gate", {})
    profile = str(section.get("profile", "fast")).strip().lower()
    if profile not in _VALID_PROFILES:
        raise ValueError(f"invalid [tool.intergrax.push_gate] profile: {profile}")
    skip_env = str(section.get("skip_env", "INTERGRAX_SKIP_PUSH_GATE")).strip()
    if not skip_env:
        raise ValueError("[tool.intergrax.push_gate] skip_env must be non-empty")
    skip_uv_sync_on_push = bool(section.get("skip_uv_sync_on_push", True))
    pre_commit_ruff = bool(section.get("pre_commit_ruff", True))
    diff_scoped_gates = bool(section.get("diff_scoped_gates", True))
    skip_env_pre_commit = str(section.get("skip_env_pre_commit", "INTERGRAX_SKIP_PRE_COMMIT_GATE")).strip()
    if not skip_env_pre_commit:
        raise ValueError("[tool.intergrax.push_gate] skip_env_pre_commit must be non-empty")
    return PushGateConfig(
        profile=profile,
        skip_env=skip_env,
        skip_uv_sync_on_push=skip_uv_sync_on_push,
        pre_commit_ruff=pre_commit_ruff,
        diff_scoped_gates=diff_scoped_gates,
        skip_env_pre_commit=skip_env_pre_commit,
    )


_GLOB_CHARS = frozenset("*?[]")


def load_ruff_exempt_prefixes(pyproject: Path | None = None) -> tuple[str, ...]:
    """Plain directory/file entries of ``[tool.ruff] extend-exclude`` (SSOT for gate exemptions).

    Ruff ignores ``exclude`` for paths passed explicitly on the command line unless
    ``force-exclude`` is set, and ``force-exclude`` would also re-apply Ruff's default
    excludes (``build``, ``dist``, ...) to tracked sources. The changed-path gates therefore
    filter explicit paths against these entries only. Glob entries are not honored here,
    so the gates can only be stricter than ``ruff check .``.
    """
    path = pyproject or _PYPROJECT
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    entries = data.get("tool", {}).get("ruff", {}).get("extend-exclude", [])
    prefixes: list[str] = []
    for entry in entries if isinstance(entries, list) else []:
        text = str(entry).replace("\\", "/").strip().strip("/")
        if text and not (_GLOB_CHARS & set(text)):
            prefixes.append(text)
    return tuple(prefixes)


def ruff_exempt(path: str, prefixes: tuple[str, ...]) -> bool:
    rel = path.replace("\\", "/").removeprefix("./")
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def split_ruff_exempt(paths: list[str], prefixes: tuple[str, ...] | None = None) -> tuple[list[str], list[str]]:
    """Return (lint_paths, exempt_paths) for explicit Ruff invocations."""
    active = load_ruff_exempt_prefixes() if prefixes is None else prefixes
    lint = [p for p in paths if not ruff_exempt(p, active)]
    exempt = [p for p in paths if ruff_exempt(p, active)]
    return lint, exempt


def push_gate_skipped(config: PushGateConfig | None = None) -> bool:
    cfg = config or load_push_gate_config()
    return _truthy_env(os.environ.get(cfg.skip_env))
