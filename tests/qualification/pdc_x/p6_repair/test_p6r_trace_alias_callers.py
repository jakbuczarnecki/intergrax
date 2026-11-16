# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 6 - in-repo modules use ``TraceArtifactRef``; the COMPAT-X ``ArtifactRef`` shim stays for externals."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]

#: Modules that serve the legacy ``ArtifactRef`` name through a COMPAT-X ``__getattr__`` shim.
SHIM_MODULES = frozenset(
    {
        "intergrax.runtime.nexus.artifacts.models",
        "intergrax.runtime.nexus.tracing.trace_models",
        "intergrax.runtime.execution.trace_persistence_models_bridge",
    }
)

#: The seven modules this batch moves to ``TraceArtifactRef``.
MIGRATED = (
    "intergrax/runtime/execution/agent_runtime_context.py",
    "intergrax/runtime/execution/trace_persistence_models_bridge.py",
    "intergrax/runtime/nexus/context/context_manager.py",
    "intergrax/runtime/nexus/context/shared_task_context.py",
    "intergrax/runtime/nexus/engine/runtime_state.py",
    "intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py",
    "intergrax/runtime/nexus/tracing/persistence_models.py",
)

#: Classified leftovers: an in-repo importer of the legacy name that lies outside the B-P6-R write set.
#: The entry must disappear once that module moves (``test_classified_leftovers_still_exist`` enforces it).
CLASSIFIED_LEFTOVERS = frozenset(
    {
        "intergrax/runtime/observability/emitter.py",
    }
)

_PRODUCT_TREES = ("intergrax", "agents", "applications")


def _imports_legacy_alias(source: str) -> bool:
    """True when ``source`` imports the legacy ``ArtifactRef`` name from a shim module."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module in SHIM_MODULES:
            if any(alias.name == "ArtifactRef" for alias in node.names):
                return True
    return False


def _legacy_importers() -> set[str]:
    """Repo-relative paths of product modules that import the legacy ``ArtifactRef`` from a shim module."""
    offenders: set[str] = set()
    for tree_name in _PRODUCT_TREES:
        for path in (REPO_ROOT / tree_name).rglob("*.py"):
            if ".venv" in path.parts or "build" in path.parts:
                continue
            if _imports_legacy_alias(path.read_text(encoding="utf-8")):
                offenders.add(path.relative_to(REPO_ROOT).as_posix())
    return offenders


def _is_shim_comparison(tree: ast.AST, constant: ast.Constant) -> bool:
    """True when ``constant`` is the right-hand side of ``name == "ArtifactRef"`` inside a module ``__getattr__``."""
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef) and func.name == "__getattr__":
            for node in ast.walk(func):
                if isinstance(node, ast.Compare) and constant in node.comparators:
                    return True
    return False


@pytest.mark.parametrize("relative", MIGRATED)
def test_migrated_module_no_longer_mentions_the_alias(relative: str) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    # String annotations ("ArtifactRef") count as uses; the bridge's COMPAT-X ``__getattr__`` compares
    # the requested name with a string, which is the shim itself and not a use of the alias.
    names |= {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.isidentifier()
        and not _is_shim_comparison(tree, n)
    }
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "ArtifactRef" not in names
    assert "ArtifactRef" not in imported
    assert "TraceArtifactRef" in imported or "TraceArtifactRef" in names


def test_no_in_repo_module_reintroduces_the_alias() -> None:
    """The gate: only classified leftovers may still import the legacy name from a shim module."""
    assert _legacy_importers() - CLASSIFIED_LEFTOVERS == set()


def test_classified_leftovers_still_exist() -> None:
    assert CLASSIFIED_LEFTOVERS <= _legacy_importers()


def test_the_gate_detects_a_reintroduced_alias() -> None:
    """Self-test of the detector on synthetic sources."""
    nl = chr(10)
    shim = "intergrax.runtime.nexus.artifacts.models"
    assert _imports_legacy_alias(f"from {shim} import ArtifactRef{nl}")
    assert _imports_legacy_alias(f"if TYPE_CHECKING:{nl}    from {shim} import ArtifactRef{nl}")
    assert not _imports_legacy_alias(f"from {shim} import TraceArtifactRef{nl}")
    assert not _imports_legacy_alias(f"from intergrax.contracts.artifact_ref import ArtifactRef{nl}")


@pytest.mark.parametrize("module_name", sorted(SHIM_MODULES))
def test_the_compat_shim_still_serves_external_callers(module_name: str) -> None:
    from intergrax.contracts.tracing import TraceArtifactRef

    module = importlib.import_module(module_name)
    namespace: dict[str, object] = {}
    exec(f"from {module_name} import ArtifactRef", namespace)  # noqa: S102 - the old import statement
    assert namespace["ArtifactRef"] is TraceArtifactRef
    assert module.ArtifactRef is TraceArtifactRef  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        module.NoSuchName  # noqa: B018


def test_the_bridge_exports_the_unambiguous_name() -> None:
    from intergrax.contracts.tracing import TraceArtifactRef
    from intergrax.runtime.execution import trace_persistence_models_bridge as bridge

    assert bridge.TraceArtifactRef is TraceArtifactRef
    assert "TraceArtifactRef" in bridge.__all__
    assert "ArtifactRef" not in bridge.__all__
    assert "ArtifactRef" not in bridge.__dict__
