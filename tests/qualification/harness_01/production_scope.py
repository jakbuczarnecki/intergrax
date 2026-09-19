# © Artur Czarnecki. All rights reserved.

"""Production tree boundaries for HARNESS-01 static gates."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_EXCLUDED_REL_SEGMENTS = frozenset(
    {
        "tests",
        "testing_support",
        "platform_proofs",
        "__pycache__",
        ".pytest_cache",
    }
)

_APPLICATION_HOST_GLOBS = (
    "applications/*/host/**/*.py",
    "intergrax/applications/**/host/**/*.py",
)


def repo_root() -> Path:
    return _REPO_ROOT


def relative_posix(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def is_excluded_qualification_path(rel_posix: str) -> bool:
    parts = rel_posix.split("/")
    return any(segment in _EXCLUDED_REL_SEGMENTS for segment in parts)


def iter_py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def iter_production_intergrax_py_files() -> list[Path]:
    root = _REPO_ROOT / "intergrax"
    out: list[Path] = []
    for path in iter_py_files(root):
        rel = relative_posix(path)
        if is_excluded_qualification_path(rel):
            continue
        out.append(path)
    return out


def iter_production_agent_py_files() -> list[Path]:
    root = _REPO_ROOT / "intergrax" / "agents"
    return iter_py_files(root)


def iter_application_host_py_files() -> list[Path]:
    out: list[Path] = []
    for pattern in _APPLICATION_HOST_GLOBS:
        for path in _REPO_ROOT.glob(pattern):
            if path.is_file() and path.suffix == ".py":
                rel = relative_posix(path)
                if not is_excluded_qualification_path(rel):
                    out.append(path)
    return out
