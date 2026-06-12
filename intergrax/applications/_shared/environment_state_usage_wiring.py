# © Artur Czarnecki. All rights reserved.

"""Typed ``ApplicationEnvironmentState`` usage lint for Tier-3 hooks (APP-PROD-6)."""

from __future__ import annotations

import ast
from pathlib import Path

HARNESS_ENV_STATE_WIRING_MARKER = "apply_application_environment_state_wiring"

RAW_RUNTIME_STATE_ACCESS_PATTERNS = (
    'runtime_state["app_env_state',
    "runtime_state['app_env_state",
    'runtime_state.get("app_env_state',
    "runtime_state.get('app_env_state",
)

TYPED_STATE_HELPER_MARKERS = (
    "from_runtime_state",
    "patch_runtime_state",
    "apply_to_runtime_state",
    "seed_application_environment_state",
)

SKIP_RAW_ACCESS_SUFFIXES = (
    "/contracts/environment_state.py",
    "/application_environment_state_middleware.py",
    "/environment_state_usage_wiring.py",
)


def check_harness_environment_state_wiring(harness_path: Path) -> list[str]:
    """Ensure ``build_harness_host_runtime`` mounts env-state lifecycle middleware."""
    rel = harness_path.as_posix()
    if not harness_path.is_file():
        return [f"missing harness runtime module: {rel}"]
    text = harness_path.read_text(encoding="utf-8")
    if HARNESS_ENV_STATE_WIRING_MARKER not in text:
        return [
            f"{rel}: must call {HARNESS_ENV_STATE_WIRING_MARKER} "
            "for ApplicationEnvironmentState lifecycle sync",
        ]
    return []


def _rel_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _should_skip_raw_access_scan(rel: str) -> bool:
    return any(rel.endswith(suffix) for suffix in SKIP_RAW_ACCESS_SUFFIXES)


def check_no_raw_app_env_state_access(path: Path, *, repo_root: Path) -> list[str]:
    """Reject ad-hoc ``runtime_state`` access to ``app_env_state.v1``."""
    rel = _rel_path(path, repo_root)
    if _should_skip_raw_access_scan(rel):
        return []
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    for pattern in RAW_RUNTIME_STATE_ACCESS_PATTERNS:
        if pattern in text:
            violations.append(
                f"{rel}: forbidden raw app env state access {pattern!r}; "
                "use ApplicationEnvironmentState.from_runtime_state / patch_runtime_state",
            )
    return violations


def _on_hook_source_segments(tree: ast.AST, source: str) -> list[str]:
    segments: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "on_hook":
            segment = ast.get_source_segment(source, node)
            if segment:
                segments.append(segment)
    return segments


def check_on_hook_typed_state_usage(path: Path, *, repo_root: Path) -> list[str]:
    """Require typed helpers when ``on_hook`` touches application environment state."""
    rel = _rel_path(path, repo_root)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError:
        return []

    violations: list[str] = []
    for segment in _on_hook_source_segments(tree, source):
        touches_state = "app_env_state" in segment or "APP_ENV_STATE_RUNTIME_KEY" in segment
        if not touches_state:
            continue
        if not any(marker in segment for marker in TYPED_STATE_HELPER_MARKERS):
            violations.append(
                f"{rel}: on_hook must use ApplicationEnvironmentState typed helpers "
                "(from_runtime_state / patch_runtime_state / apply_to_runtime_state) "
                "when reading or writing app_env_state.v1",
            )
    return violations


def iter_tier3_python_sources(repo_root: Path) -> list[Path]:
    """Scan application packages and Tier-3 shared wiring for hook/state usage."""
    roots = (
        repo_root / "applications",
        repo_root / "intergrax" / "applications" / "_shared",
    )
    paths: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        paths.extend(sorted(root.rglob("*.py")))
    return paths


def check_environment_state_usage(repo_root: Path) -> list[str]:
    """Run APP-PROD-6 lint across harness wiring and Tier-3 Python sources."""
    violations: list[str] = []
    harness_path = repo_root / "intergrax" / "applications" / "_shared" / "harness_host_runtime.py"
    violations.extend(check_harness_environment_state_wiring(harness_path))

    for path in iter_tier3_python_sources(repo_root):
        violations.extend(check_no_raw_app_env_state_access(path, repo_root=repo_root))
        violations.extend(check_on_hook_typed_state_usage(path, repo_root=repo_root))
    return violations
