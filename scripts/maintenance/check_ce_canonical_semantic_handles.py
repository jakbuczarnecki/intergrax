# © Artur Czarnecki. All rights reserved.

"""CE-01-R1A: canonical ContextEngine runtime surface must not use legacy handle ABI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context"

CANONICAL_RUNTIME_MODULES = (
    CONTEXT_DIR / "context_engine.py",
    CONTEXT_DIR / "assembly_runtime_deps.py",
)

LEGACY_BRIDGE_MODULE = CONTEXT_DIR / "legacy_assembly_runtime_bridge.py"

SEMANTIC_LITERALS = frozenset(
    {
        "runtime_config",
        "messages",
        "max_output_tokens",
        "context_optimization_policy",
        "nexus_ucl_runtime",
        "event_bus",
        "node_id",
        "agent_id",
    }
)

HANDLE_GET_PATTERN = re.compile(
    r"\.handles\.get\(\s*[\"']([^\"']+)[\"']"
)

FORBIDDEN_IMPORTS_IN_ENGINE = (
    "legacy_assembly_runtime_bridge",
    "ensure_context_assembly_runtime",
    "try_build_runtime_from_legacy_handles",
    "build_context_assembly_runtime_from_legacy_handles",
)

FORBIDDEN_REFLECTION_ON_CANONICAL = re.compile(r"\b(getattr|hasattr)\s*\(")


def _scan_handle_gets(path: Path, text: str) -> list[str]:
    violations: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    for match in HANDLE_GET_PATTERN.finditer(text):
        key = match.group(1)
        if key in SEMANTIC_LITERALS:
            violations.append(f"{rel}: handles.get({key!r})")
    return violations


def _scan_reflection(path: Path, text: str) -> list[str]:
    violations: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    if FORBIDDEN_REFLECTION_ON_CANONICAL.search(text):
        violations.append(f"{rel}: getattr/hasattr on canonical runtime module")
    return violations


def main() -> int:
    violations: list[str] = []

    engine_path = CONTEXT_DIR / "context_engine.py"
    engine_text = engine_path.read_text(encoding="utf-8")
    for token in FORBIDDEN_IMPORTS_IN_ENGINE:
        if token in engine_text:
            violations.append(f"context_engine.py: forbidden import/reference {token!r}")
    violations.extend(_scan_handle_gets(engine_path, engine_text))

    deps_path = CONTEXT_DIR / "assembly_runtime_deps.py"
    deps_text = deps_path.read_text(encoding="utf-8")
    violations.extend(_scan_handle_gets(deps_path, deps_text))
    violations.extend(_scan_reflection(deps_path, deps_text))

    contracts_path = REPO_ROOT / "intergrax" / "context" / "contracts.py"
    contracts_text = contracts_path.read_text(encoding="utf-8")
    if "_hydrate_runtime_from_legacy_handles" in contracts_text:
        violations.append("contracts.py: automatic semantic runtime hydration")
    if "legacy_assembly_runtime_bridge" in contracts_text:
        violations.append("contracts.py: imports legacy assembly runtime bridge")

    if violations:
        print("CE canonical typed runtime violations:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
