# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R017 (single definition, COMPAT-X shim) and R018 (no ambiguous ArtifactRef alias)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SHIM = "intergrax/applications/contracts/application_artifacts.py"
CANONICAL = "intergrax/contracts/task_artifacts.py"
SHARED_NAMES = (
    "ArtifactSecurityClass",
    "ArtifactVisibility",
    "ArtifactRetentionPolicy",
    "ApplicationArtifactRef",
    "WorkspaceArtifactRef",
    "SandboxArtifactRef",
    "RunArtifactBundle",
)


def _class_definitions() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in (REPO_ROOT / "intergrax").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                found.setdefault(node.name, []).append(path.relative_to(REPO_ROOT).as_posix())
    return found


# ------------------------------------------------------------------ R017


def test_each_artifact_contract_is_defined_exactly_once() -> None:
    definitions = _class_definitions()
    for name in SHARED_NAMES:
        assert definitions.get(name) == [CANONICAL], (name, definitions.get(name))


def test_application_artifacts_is_a_pure_reexport_shim() -> None:
    tree = ast.parse((REPO_ROOT / SHIM).read_text(encoding="utf-8"))
    assert not [n for n in tree.body if isinstance(n, ast.ClassDef)]
    shim = importlib.import_module("intergrax.applications.contracts.application_artifacts")
    canonical = importlib.import_module("intergrax.contracts.task_artifacts")
    for name in SHARED_NAMES:
        assert getattr(shim, name) is getattr(canonical, name), name


def test_shim_keeps_every_public_symbol_of_the_old_module() -> None:
    shim = importlib.import_module("intergrax.applications.contracts.application_artifacts")
    from intergrax.contracts.run_artifact_bundle import RUN_ARTIFACT_BUNDLE_METADATA_KEY

    assert shim.RUN_ARTIFACT_BUNDLE_METADATA_KEY == RUN_ARTIFACT_BUNDLE_METADATA_KEY
    assert shim.APPLICATION_ARTIFACTS_STAGING_KEY == "application_artifacts.v1"
    for name in (*SHARED_NAMES, "RUN_ARTIFACT_BUNDLE_METADATA_KEY", "APPLICATION_ARTIFACTS_STAGING_KEY"):
        assert name in shim.__all__, name


def test_legacy_import_path_still_constructs_the_same_models() -> None:
    from intergrax.applications.contracts.application_artifacts import ApplicationArtifactRef
    from intergrax.contracts.task_artifacts import ApplicationArtifactRef as Canonical

    ref = ApplicationArtifactRef(
        artifact_id="a", kind="k", uri="file:///x", task_id="t", owner_app_id="app", tenant_id="tn"
    )
    assert isinstance(ref, Canonical)


def test_bundle_sandbox_refs_never_carry_an_empty_digest(tmp_path: Path) -> None:
    """The bundle builder computes the digest of the bytes it already holds."""
    text = (REPO_ROOT / "intergrax/runtime/nexus/orchestration/run_artifact_bundle_builder.py").read_text(
        encoding="utf-8"
    )
    assert 'sha256=""' not in text


# ------------------------------------------------------------------ R018


#: Same-named classes that live outside the B-D8 write set and are reported to the orchestrator.
#: ``intergrax/runtime/replay/models.py`` defines its own ``@dataclass ArtifactRef`` used by
#: ``replay_engine.py``. The set may only shrink: a new same-named class fails the first test.
KNOWN_OUT_OF_SCOPE_ARTIFACT_REF_CLASSES = {"intergrax/runtime/replay/models.py"}


def test_no_new_class_named_artifact_ref_appears() -> None:
    found = set(_class_definitions()["ArtifactRef"])
    assert "intergrax/contracts/artifact_ref.py" in found
    assert found - {"intergrax/contracts/artifact_ref.py"} <= KNOWN_OUT_OF_SCOPE_ARTIFACT_REF_CLASSES


@pytest.mark.xfail(
    strict=True,
    reason="PDC-D8-R018 residual: runtime/replay/models.py ArtifactRef is outside the B-D8 write set; "
    "XPASS (and a failing run) once the orchestrator renames it, which forces removal of this marker",
)
def test_only_one_class_named_artifact_ref_exists() -> None:
    assert _class_definitions()["ArtifactRef"] == ["intergrax/contracts/artifact_ref.py"]


def test_no_module_binds_an_alias_named_artifact_ref() -> None:
    """No ``X as ArtifactRef`` import and no ``ArtifactRef = ...`` binding anywhere in product code."""
    offenders: list[str] = []
    for path in (REPO_ROOT / "intergrax").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.asname == "ArtifactRef" and alias.name != "ArtifactRef":
                        offenders.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "ArtifactRef":
                        offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_trace_artifact_ref_keeps_its_unambiguous_name() -> None:
    from intergrax.contracts.tracing import TraceArtifactRef
    from intergrax.runtime.nexus.artifacts import models as nexus_models
    from intergrax.runtime.nexus.tracing import trace_models

    assert nexus_models.TraceArtifactRef is TraceArtifactRef
    assert trace_models.TraceArtifactRef is TraceArtifactRef
    assert "ArtifactRef" not in nexus_models.__all__
    assert "ArtifactRef" not in trace_models.__all__


@pytest.mark.parametrize(
    "module_name",
    ["intergrax.runtime.nexus.artifacts.models", "intergrax.runtime.nexus.tracing.trace_models"],
)
def test_legacy_alias_still_resolves_to_trace_artifact_ref(module_name: str) -> None:
    """COMPAT-X shim: old import paths keep working and point to the unambiguous TraceArtifactRef."""
    from intergrax.contracts.tracing import TraceArtifactRef

    module = importlib.import_module(module_name)
    assert module.ArtifactRef is TraceArtifactRef
    namespace: dict[str, object] = {}
    exec(f"from {module_name} import ArtifactRef", namespace)  # noqa: S102 - old import statement
    assert namespace["ArtifactRef"] is TraceArtifactRef
    with pytest.raises(AttributeError):
        module.NoSuchName  # noqa: B018
