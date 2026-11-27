# © Artur Czarnecki. All rights reserved.

"""RW-plan architecture guard NT-OWN-05-1: one mechanism per named responsibility.

Covered here: the host LLM resolver (decision DP-16, finding A4-016). Exactly one product module
may both define ``resolve_llm_adapter`` and construct adapters; the Tier-3 module of the same name
may only delegate. The other four named responsibilities (plugin discovery, context-window tables,
tool-input guard, evaluator contract) have no step-0 scan in the W-B plan and are not guarded
here. Pure guard: it tightens or loosens no existing gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_FACTORIES = frozenset({"create_adapter", "create_adapter_with_failover"})
_CANONICAL = "intergrax/runtime/wiring/llm_resolver.py"
_DELEGATING_FACADE = "intergrax/applications/_shared/llm_resolver.py"


def _is_application_docker_runtime_context_mirror(parts: tuple[str, ...]) -> bool:
    """Distribution mirrors are vendored copies, not composition roots (W3 C1-R)."""
    return len(parts) >= 4 and parts[0] == "applications" and parts[2] == "docker" and parts[3] == "runtime-context"


def _resolver_definitions(*extra_roots: Path) -> dict[str, bool]:
    """Map module path to whether the module also calls an adapter factory."""
    found: dict[str, bool] = {}
    scan_roots: list[tuple[Path, Path]] = [
        (_REPO_ROOT / name, _REPO_ROOT) for name in ("intergrax", "agents", "applications")
    ]
    for extra in extra_roots:
        scan_roots.append((extra, extra))
    for root, base in scan_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                relative = path.relative_to(base).as_posix()
            except ValueError:
                continue
            parts = path.relative_to(base).parts
            if "/tests/" in relative:
                continue
            if _is_application_docker_runtime_context_mirror(parts):
                continue
            text = path.read_text(encoding="utf-8")
            if "def resolve_llm_adapter" not in text:
                continue
            tree = ast.parse(text)
            defines = any(
                isinstance(node, ast.FunctionDef) and node.name == "resolve_llm_adapter" for node in ast.walk(tree)
            )
            if not defines:
                continue
            constructs = any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id in _ADAPTER_FACTORIES)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in _ADAPTER_FACTORIES)
                )
                for node in ast.walk(tree)
            )
            found[relative] = constructs
    return found


def test_nt_own_05_1_one_host_llm_resolver() -> None:
    definitions = _resolver_definitions()
    assert set(definitions) == {_CANONICAL, _DELEGATING_FACADE}
    assert [path for path, constructs in definitions.items() if constructs] == [_CANONICAL]


@pytest.mark.parametrize(
    ("parts", "is_mirror"),
    [
        (("applications", "demo", "docker", "runtime-context", "wiring", "llm_resolver.py"), True),
        (("applications", "demo", "host", "llm_resolver.py"), False),
        (("applications", "demo", "docker", "other", "llm_resolver.py"), False),
        (("intergrax", "runtime", "wiring", "llm_resolver.py"), False),
    ],
)
def test_application_docker_runtime_context_mirror_segments(parts: tuple[str, ...], is_mirror: bool) -> None:
    assert _is_application_docker_runtime_context_mirror(parts) is is_mirror


def test_distribution_mirror_resolver_excluded_from_duplicate_scan(tmp_path: Path) -> None:
    """Only ``applications/*/docker/runtime-context/**`` mirrors are skipped; not other docker trees."""
    mirror = tmp_path / "applications" / "demo" / "docker" / "runtime-context" / "fake_llm.py"
    mirror.parent.mkdir(parents=True)
    mirror.write_text(
        "def resolve_llm_adapter():\n    return create_adapter('x')\n",
        encoding="utf-8",
    )
    parts = mirror.relative_to(tmp_path).parts
    assert _is_application_docker_runtime_context_mirror(parts)
    definitions = _resolver_definitions(tmp_path)
    assert "applications/demo/docker/runtime-context/fake_llm.py" not in definitions


def test_non_mirror_module_with_adapter_factory_would_fail_gate(tmp_path: Path) -> None:
    rogue = tmp_path / "applications" / "demo" / "host" / "rogue_resolver.py"
    rogue.parent.mkdir(parents=True)
    rogue.write_text(
        "def resolve_llm_adapter():\n    return create_adapter('x')\n",
        encoding="utf-8",
    )
    relative = rogue.relative_to(tmp_path).as_posix()
    assert not _is_application_docker_runtime_context_mirror(rogue.relative_to(tmp_path).parts)
    definitions = _resolver_definitions(tmp_path)
    assert definitions[relative] is True
    constructing = [path for path, constructs in definitions.items() if constructs]
    assert relative in constructing
    assert _CANONICAL not in constructing or len(constructing) > 1


def test_facade_must_not_construct_adapters() -> None:
    path = _REPO_ROOT / _DELEGATING_FACADE
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constructs = any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in _ADAPTER_FACTORIES)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in _ADAPTER_FACTORIES)
        )
        for node in ast.walk(tree)
    )
    assert constructs is False
