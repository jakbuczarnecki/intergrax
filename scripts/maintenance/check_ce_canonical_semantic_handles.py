# © Artur Czarnecki. All rights reserved.

"""CE-01-R1A: canonical ContextEngine runtime surface must not use legacy handle ABI."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_DIR = REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context"
CE_ASSEMBLY_RUNTIME = REPO_ROOT / "intergrax" / "context" / "assembly_runtime.py"

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

LEGACY_BRIDGE_CALL = re.compile(
    r"\b(build_context_assembly_runtime_from_legacy_handles|try_build_runtime_from_legacy_handles)\s*\("
)

FORBIDDEN_RUNTIME_ANNOTATIONS = re.compile(
    r"runtime:\s*(Any|object|dict|Mapping)\b"
)

FORBIDDEN_CE_ASSEMBLY_IMPORTS = (
    "from intergrax.runtime.events.event_bus import RuntimeEventBus",
    "from intergrax.runtime.token_optimization.message_sequence_artifact import MessageSequenceArtifactExecutor",
)

FORBIDDEN_CONTEXT_ENGINE_ISINSTANCE = (
    "isinstance(ucl_runtime, NexusUCLRuntimeDependencies)",
    "isinstance(runtime.event_bus, RuntimeEventBus)",
)

PRODUCTION_SCAN_ROOTS = (
    REPO_ROOT / "intergrax",
    REPO_ROOT / "agents",
    REPO_ROOT / "applications",
)


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
    if FORBIDDEN_RUNTIME_ANNOTATIONS.search(contracts_text):
        violations.append("contracts.py: ContextProviderContext.runtime must use typed CE contract")

    assembly_runtime_text = CE_ASSEMBLY_RUNTIME.read_text(encoding="utf-8")
    for token in FORBIDDEN_CE_ASSEMBLY_IMPORTS:
        if token in assembly_runtime_text:
            violations.append(f"assembly_runtime.py: forbidden concrete import {token!r}")
    for forbidden in FORBIDDEN_CONTEXT_ENGINE_ISINSTANCE:
        if forbidden in engine_text:
            violations.append(f"context_engine.py: forbidden concrete runtime check {forbidden!r}")

    bridge_rel = LEGACY_BRIDGE_MODULE.relative_to(REPO_ROOT).as_posix()
    for root in PRODUCTION_SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel == bridge_rel:
                continue
            if "/tests/" in f"/{rel}/" or rel.startswith("tests/"):
                continue
            text = path.read_text(encoding="utf-8")
            if LEGACY_BRIDGE_CALL.search(text):
                violations.append(f"{rel}: production legacy assembly runtime bridge call")

    if violations:
        print("CE canonical typed runtime violations:", file=sys.stderr)
        for item in violations:
            print(f"  - {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
