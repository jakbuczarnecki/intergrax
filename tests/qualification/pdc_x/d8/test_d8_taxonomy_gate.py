# © Artur Czarnecki. All rights reserved.

"""PDC-D8-R001: four artifact classes, four owners, no global storage engine."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from intergrax.contracts.artifact_ref import ArtifactClass
from intergrax.runtime.nexus.artifacts.in_memory_artifact_store import InMemoryArtifactStore
from intergrax.runtime.nexus.artifacts.store_base import ArtifactClassNotOwnedError
from intergrax.runtime.nexus.artifacts.taxonomy import (
    ARTIFACT_CLASS_OWNERS,
    STORE_ACCEPTED_CLASSES,
    assert_taxonomy_consistent,
    owner_of,
)

from ._d8_support import make_artifact

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_exactly_four_classes_with_one_owner_each() -> None:
    assert {c.value for c in ArtifactClass} == {
        "business",
        "execution_evidence",
        "context",
        "tool_output",
    }
    assert set(ARTIFACT_CLASS_OWNERS) == set(ArtifactClass)
    owner_ids = [owner.owner_id for owner in ARTIFACT_CLASS_OWNERS.values()]
    assert len(owner_ids) == len(set(owner_ids)) == 4
    assert {o.owner_id for o in ARTIFACT_CLASS_OWNERS.values()} == {
        "collaborative_work",
        "evidence_plane",
        "ucl",
        "tool_runtime",
    }
    assert_taxonomy_consistent()


def test_every_owner_module_exists_in_the_repository() -> None:
    for artifact_class in ArtifactClass:
        owner = owner_of(artifact_class)
        assert (REPO_ROOT / owner.owner_path).exists(), owner.owner_path


def test_business_artifacts_stay_with_collaborative_work() -> None:
    owner = owner_of(ArtifactClass.BUSINESS)
    assert owner.owner_id == "collaborative_work"
    assert owner.owner_path.startswith("intergrax/collaborative_work")


def test_context_artifacts_stay_with_ucl() -> None:
    assert owner_of(ArtifactClass.CONTEXT).owner_path.startswith("intergrax/runtime/context_lifecycle")


def test_execution_store_refuses_classes_owned_elsewhere() -> None:
    assert ArtifactClass.BUSINESS not in STORE_ACCEPTED_CLASSES
    assert ArtifactClass.CONTEXT not in STORE_ACCEPTED_CLASSES
    store = InMemoryArtifactStore()
    for foreign in (ArtifactClass.BUSINESS, ArtifactClass.CONTEXT):
        with pytest.raises(ArtifactClassNotOwnedError):
            store.put(make_artifact(artifact_class=foreign))
    store.put(make_artifact(artifact_class=ArtifactClass.EXECUTION_EVIDENCE))
    store.put(make_artifact(artifact_id="art-2", artifact_class=ArtifactClass.TOOL_OUTPUT))


def test_no_global_artifact_storage_engine_is_defined() -> None:
    """No class named like a global artifact engine/registry exists in product code."""
    forbidden_fragments = ("GlobalArtifact", "ArtifactStorageEngine", "UniversalArtifact")
    offenders: list[str] = []
    for path in (REPO_ROOT / "intergrax").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(f in node.name for f in forbidden_fragments):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.name}")
    assert offenders == []
