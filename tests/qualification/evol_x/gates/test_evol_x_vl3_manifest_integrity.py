# © Artur Czarnecki. All rights reserved.

import pytest
from testing_support.change_safety.selection.ci_manifest import (
    build_manifest_document,
    collect_evol_x_node_ids,
    load_pinned_manifest,
    manifest_content_digest,
    verify_manifest_integrity,
)

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_evol_x_vl3_manifest_matches_collect_only() -> None:
    manifest = load_pinned_manifest()
    assert manifest["profile"] == "evol_x_p3_l3"
    assert manifest["excluded_populations"] == ["DIM-B-PROD-Q-LIVE"]
    collected = collect_evol_x_node_ids()
    assert verify_manifest_integrity(manifest, collected) == []


def test_evol_x_vl3_manifest_missing_node_detected() -> None:
    manifest = build_manifest_document(("tests/a.py::t1", "tests/a.py::t2"))
    problems = verify_manifest_integrity(manifest, ("tests/a.py::t1",))
    assert any(p.startswith("missing_from_collect") for p in problems)


def test_evol_x_vl3_manifest_unpinned_node_and_shrunk_manifest_detected() -> None:
    manifest = build_manifest_document(("tests/a.py::t1",))
    assert any(
        p.startswith("unpinned_collected")
        for p in verify_manifest_integrity(manifest, ("tests/a.py::t1", "tests/a.py::t2"))
    )
    shrunk = {**manifest, "pytest_node_ids": []}
    assert "manifest_content_digest mismatch" in verify_manifest_integrity(shrunk, ())


def test_evol_x_vl3_manifest_digest_is_content_bound() -> None:
    assert manifest_content_digest(["a"]) != manifest_content_digest(["b"])
    assert manifest_content_digest(["a", "b"]) != manifest_content_digest(["b", "a"])
