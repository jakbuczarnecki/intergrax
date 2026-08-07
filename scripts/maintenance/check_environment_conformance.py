#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fast, deterministic proof of the canonical Intergrax development environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_ROOT = (REPO_ROOT / ".venv").resolve()
COMMAND_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class EnvironmentProfile:
    name: str
    requires_managed_python: bool
    dependencies: tuple[tuple[str, str, str | None, bool], ...]


_PROFILES = {
    "local": EnvironmentProfile(
        name="local",
        requires_managed_python=True,
        dependencies=(
            ("PYTEST", "pytest", "pytest", True),
            ("RUFF", "ruff", None, True),
            ("PYDANTIC", "pydantic", "pydantic", False),
            ("FASTAPI", "fastapi", "fastapi", False),
        ),
    ),
    "ci": EnvironmentProfile(
        name="ci",
        requires_managed_python=False,
        dependencies=(
            ("PYTEST", "pytest", "pytest", True),
            ("PYDANTIC", "pydantic", "pydantic", False),
            ("FASTAPI", "fastapi", "fastapi", False),
        ),
    ),
}


def profile_contract(name: str) -> EnvironmentProfile:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown environment profile: {name}") from exc


def _within(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


def evaluate_venv_paths(
    executable: Path, prefix: Path, venv_root: Path
) -> tuple[bool, str]:
    resolved_executable = executable.resolve()
    resolved_prefix = prefix.resolve()
    resolved_venv = venv_root.resolve()
    executable_ok = _within(resolved_executable, resolved_venv)
    prefix_ok = resolved_prefix == resolved_venv
    detail = f"executable={resolved_executable}; prefix={resolved_prefix}"
    return executable_ok and prefix_ok, detail


def evaluate_pythonpath(environment: dict[str, str]) -> tuple[bool, str]:
    if environment.get("PYTHONPATH", ""):
        return False, "non-empty PYTHONPATH injection detected"
    return True, "empty or unset"


def evaluate_base_provenance(
    actual: Path,
    actual_prefix: Path,
    expected: Path,
    managed_root: Path,
    version_parts: dict[str, Any],
) -> tuple[bool, str]:
    actual = actual.resolve()
    actual_prefix = actual_prefix.resolve()
    expected = expected.resolve()
    managed_root = managed_root.resolve()
    if version_parts.get("major") != 3 or version_parts.get("minor") != 12:
        return False, f"uv returned non-3.12 interpreter: {expected}"
    if not _within(expected, managed_root):
        return False, f"uv interpreter is outside managed root: {expected}"
    if actual != expected:
        return False, f"base executable={actual}; expected={expected}"
    if actual_prefix != expected.parent:
        return False, f"base_prefix={actual_prefix}; expected={expected.parent}"
    if any(marker in str(actual).casefold() for marker in ("conda", "anaconda")):
        return False, f"Conda/Anaconda interpreter detected: {actual}"
    return True, f"base={actual}; managed_root={managed_root}"


def parse_pyvenv_cfg(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip().casefold()] = value.strip()
    return values


def evaluate_pyvenv_cfg(
    cfg_path: Path, base_prefix: Path
) -> tuple[bool, str, dict[str, str]]:
    if not cfg_path.is_file():
        return False, f"missing {cfg_path}", {}

    try:
        values = parse_pyvenv_cfg(cfg_path)
        home = Path(values["home"]).resolve()
    except (KeyError, OSError, ValueError) as exc:
        return False, f"invalid pyvenv.cfg: {exc}", values if "values" in locals() else {}

    expected_home = base_prefix.resolve()
    missing = [
        key
        for key in ("home", "include-system-site-packages")
        if key not in values
    ]
    if missing:
        return False, f"missing keys: {', '.join(missing)}", values
    if home != expected_home:
        return False, f"home={home} does not match base_prefix={expected_home}", values
    if values["include-system-site-packages"].casefold() != "false":
        return False, "include-system-site-packages must be false", values
    return True, f"home={home}; isolated=true", values


def _uv_path() -> Path | None:
    executable = shutil.which("uv")
    return Path(executable).resolve() if executable else None


def _run_uv(repo_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    executable = _uv_path()
    if executable is None:
        raise RuntimeError("uv is not available on PATH")
    return subprocess.run(
        [str(executable), *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )


def _command_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = (result.stderr or result.stdout).strip().replace("\n", " ")
    if len(output) > 240:
        output = output[:237] + "..."
    return f"exit_code={result.returncode}" + (f"; {output}" if output else "")


def evaluate_managed_python(
    repo_root: Path, base_executable: Path, base_prefix: Path
) -> tuple[bool, str]:
    try:
        listed = _run_uv(
            repo_root,
            [
                "python",
                "list",
                "3.12",
                "--managed-python",
                "--only-installed",
                "--output-format",
                "json",
            ],
        )
        managed_dir = _run_uv(repo_root, ["python", "dir"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return False, f"unable to query uv-managed Python: {exc}"

    if listed.returncode != 0:
        return False, f"uv python list failed: {_command_detail(listed)}"
    if managed_dir.returncode != 0:
        return False, f"uv python dir failed: {_command_detail(managed_dir)}"

    try:
        entries: list[dict[str, Any]] = json.loads(listed.stdout)
        entry = next(
            item
            for item in entries
            if item.get("implementation", "").casefold() == "cpython"
        )
        expected = Path(entry["path"]).resolve()
        managed_root = Path(managed_dir.stdout.strip()).resolve()
    except (KeyError, StopIteration, TypeError, ValueError, OSError) as exc:
        return False, f"invalid uv managed-Python response: {exc}"

    return evaluate_base_provenance(
        base_executable,
        base_prefix,
        expected,
        managed_root,
        entry.get("version_parts", {}),
    )


def check_distribution(
    distribution_name: str,
    module_name: str | None,
    venv_root: Path,
    require_executable: bool = False,
) -> tuple[bool, str]:
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        version = distribution.version
        distribution_root = Path(distribution.locate_file("")).resolve()
    except importlib.metadata.PackageNotFoundError:
        return False, f"{distribution_name} is not installed"
    except OSError as exc:
        return False, f"{distribution_name} metadata unavailable: {exc}"

    if not _within(distribution_root, venv_root):
        return False, f"package provenance is outside .venv: {distribution_root}"

    module_path = ""
    if module_name is not None:
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            return False, f"module lookup failed: {exc}"
        if spec is None or spec.origin in (None, "built-in", "frozen"):
            return False, f"module {module_name} has no filesystem origin"
        module_path = str(Path(spec.origin).resolve())
        if not _within(Path(module_path), venv_root):
            return False, f"module provenance is outside .venv: {module_path}"

    executable_path = ""
    if require_executable:
        executable_name = distribution_name + (".exe" if os.name == "nt" else "")
        executable = (
            venv_root / ("Scripts" if os.name == "nt" else "bin") / executable_name
        ).resolve()
        if not executable.is_file() or not _within(executable, venv_root):
            return False, f"venv executable missing: {executable}"
        executable_path = f"; executable={executable}"

    module_detail = f"; module={module_path}" if module_path else ""
    return True, f"version={version}; package={distribution_root}{module_detail}{executable_path}"


def _repository_root_check() -> Check:
    required = (".python-version", "pyproject.toml", "uv.lock")
    missing = [name for name in required if not (REPO_ROOT / name).is_file()]
    if missing:
        return Check(
            "REPOSITORY_ROOT",
            False,
            f"root={REPO_ROOT}; missing={', '.join(missing)}",
        )
    return Check("REPOSITORY_ROOT", True, f"root={REPO_ROOT}")


def _lock_check() -> Check:
    try:
        result = _run_uv(REPO_ROOT, ["lock", "--check", "--offline"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return Check("LOCK_CONSISTENCY", False, str(exc))
    return Check(
        "LOCK_CONSISTENCY",
        result.returncode == 0,
        _command_detail(result),
    )


def _print_check(check: Check) -> None:
    status = "PASS" if check.ok else "FAIL"
    print(f"{check.name}: {status} ({check.detail})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the Intergrax repository environment contract."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="local",
        help="environment contract to prove (default: local)",
    )
    profile = profile_contract(parser.parse_args().profile)

    checks: list[Check] = [_repository_root_check()]

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(
        Check(
            "PYTHON_VERSION",
            sys.version_info[:2] == (3, 12),
            version,
        )
    )

    venv_ok, venv_detail = evaluate_venv_paths(
        Path(sys.executable), Path(sys.prefix), VENV_ROOT
    )
    checks.append(Check("VENV", venv_ok, venv_detail))
    checks.append(
        Check(
            "PYTHONPATH",
            *evaluate_pythonpath(dict(os.environ)),
        )
    )

    base_executable = Path(getattr(sys, "_base_executable", ""))
    base_prefix = Path(sys.base_prefix)
    cfg_ok, cfg_detail, _ = evaluate_pyvenv_cfg(
        VENV_ROOT / "pyvenv.cfg", base_prefix
    )
    checks.append(Check("PYVENV_CFG", cfg_ok, cfg_detail))
    if profile.requires_managed_python:
        managed_ok, managed_detail = evaluate_managed_python(
            REPO_ROOT, base_executable, base_prefix
        )
        checks.append(Check("MANAGED_PYTHON", managed_ok, managed_detail))

    checks.append(_lock_check())
    checks.extend(
        Check(
            name,
            *check_distribution(
                distribution_name,
                module_name,
                VENV_ROOT,
                require_executable=require_executable,
            ),
        )
        for name, distribution_name, module_name, require_executable in profile.dependencies
    )

    print(f"ENVIRONMENT_CONFORMANCE: {'PASS' if all(c.ok for c in checks) else 'FAIL'}")
    print()
    print(f"Profile: {profile.name}")
    print()
    print("Repository:")
    _print_check(checks[0])
    print()
    print("Python:")
    python_checks = ["PYTHON_VERSION", "VENV", "PYVENV_CFG"]
    if profile.requires_managed_python:
        python_checks.append("MANAGED_PYTHON")
    for name in python_checks:
        _print_check(next(check for check in checks if check.name == name))
    if not profile.requires_managed_python:
        print(
            "MANAGED_PYTHON: NOT_REQUIRED "
            "(CI interpreter provenance is provided by setup-uv/runner)"
        )
    print(f"executable: {Path(sys.executable).resolve()}")
    print(f"prefix: {Path(sys.prefix).resolve()}")
    print(f"base_prefix: {Path(sys.base_prefix).resolve()}")
    print()
    print("Isolation:")
    _print_check(next(check for check in checks if check.name == "PYTHONPATH"))
    print()
    print("Dependencies:")
    dependency_checks = ["LOCK_CONSISTENCY", *[item[0] for item in profile.dependencies]]
    for name in dependency_checks:
        _print_check(next(check for check in checks if check.name == name))

    failures = [check for check in checks if not check.ok]
    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(f"* {failure.name}: {failure.detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
