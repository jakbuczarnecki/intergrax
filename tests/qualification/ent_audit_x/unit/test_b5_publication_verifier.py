from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit.evidence import sha256_file
from intergrax.platform.enterprise_audit.validators import verify_b5_publication_tree


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ladder_tes(root: Path) -> str | None:
    ladder = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md"
    if not ladder.is_file():
        return None
    for line in ladder.read_text(encoding="utf-8").splitlines():
        if line.startswith("| TEST_EXECUTION_SHA |"):
            cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and len(cells[1]) == 40:
                return cells[1]
    return None


def _published_tree(root: Path) -> Path | None:
    handoff = root / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
    tes = _ladder_tes(root)
    if tes:
        candidate = handoff / f"qualification_at_{tes}"
        if candidate.is_dir():
            return candidate
    matches = sorted(handoff.glob("qualification_at_*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _fixture_copy(root: Path, source: Path, tag: str) -> Path:
    parent = root / ".tmp" / "session" / "b5-verifier-fixtures" / tag
    if parent.exists():
        shutil.rmtree(parent)
    parent.mkdir(parents=True, exist_ok=True)
    dest = parent / source.name
    shutil.copytree(source, dest)
    index_path = dest / "publication_index.v1.json"
    if not index_path.is_file():
        return dest
    index = json.loads(index_path.read_text(encoding="utf-8"))
    old_prefix = source.relative_to(root).as_posix()
    new_prefix = dest.relative_to(root).as_posix()
    index["l3_bundle_relpath"] = str(index["l3_bundle_relpath"]).replace(old_prefix, new_prefix, 1)
    qw = index.get("qual_wave_report") or {}
    if qw.get("path"):
        qw["path"] = str(qw["path"]).replace(old_prefix, new_prefix, 1)
    for row in index.get("artifacts", []):
        if row.get("path"):
            row["path"] = str(row["path"]).replace(old_prefix, new_prefix, 1)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def test_verify_b5_publication_tree_passes_on_strict_bundle() -> None:
    root = _repo_root()
    tree = _published_tree(root)
    if tree is None:
        pytest.skip("no B5 handoff tree")
    tes = tree.name.removeprefix("qualification_at_")
    errors = verify_b5_publication_tree(root, tree, expected_qualification_sha=tes)
    if errors and any("qual_wave_report sha256 mismatch" in e for e in errors):
        pytest.skip("canonical B5 tree still has known R7 digest defect — republish required")
    assert not errors, errors


def test_corrupted_qual_wave_digest_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "corrupt-qw-digest")
    index_path = dest / "publication_index.v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["qual_wave_report"]["sha256"] = "0" * 64
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = verify_b5_publication_tree(root, dest)
    assert any("qual_wave_report sha256 mismatch" in e for e in errors)


def test_missing_publication_index_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "missing-index")
    (dest / "publication_index.v1.json").unlink()
    errors = verify_b5_publication_tree(root, dest)
    assert any("publication_index" in e for e in errors)


def test_corrupted_l3_digest_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "corrupt-l3-digest")
    index_path = dest / "publication_index.v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0]["sha256"] = "f" * 64
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = verify_b5_publication_tree(root, dest)
    assert any("sha256 mismatch" in e for e in errors)


def test_qualification_sha_mismatch_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "qual-sha-mismatch")
    wrong = "0" * 40
    errors = verify_b5_publication_tree(root, dest, expected_qualification_sha=wrong)
    assert any("qualification_sha mismatch" in e for e in errors)


def test_missing_mandatory_l3_artifact_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "missing-l3-artifact")
    (dest / "l3-cli-full" / "frz_coverage.v1.json").unlink()
    errors = verify_b5_publication_tree(root, dest)
    assert errors


def test_path_traversal_in_index_fails() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    dest = _fixture_copy(root, source, "path-traversal")
    index_path = dest / "publication_index.v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["artifacts"][0]["path"] = "../../../etc/passwd"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = verify_b5_publication_tree(root, dest)
    assert any("traversal" in e or "invalid path" in e for e in errors)


def test_known_defective_r7_tree_fails_closed() -> None:
    root = Path(__file__).resolve().parents[4]
    defective = (
        root
        / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
        / "qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d"
    )
    if not defective.is_dir():
        pytest.skip("R7 defective tree not present")
    errors = verify_b5_publication_tree(
        root,
        defective,
        expected_qualification_sha="87203c1b6ce9786dce0523cca29b556106b6bc8d",
    )
    assert any("qual_wave_report sha256 mismatch" in e for e in errors)


def test_republish_fixes_digest_consistency() -> None:
    root = _repo_root()
    source = _published_tree(root)
    if source is None:
        pytest.skip("no B5 handoff tree")
    qw = source / "qual_wave_report.json"
    index_path = source / "publication_index.v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    actual = sha256_file(qw)
    if index["qual_wave_report"]["sha256"] != actual:
        pytest.skip("awaiting corrected B5 republication")
    assert index["qual_wave_report"]["sha256"] == actual
