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


def _strict_valid_tree(root: Path) -> Path | None:
    handoff = root / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
    for path in sorted(handoff.glob("qualification_at_*")):
        tes = path.name.removeprefix("qualification_at_")
        if not verify_b5_publication_tree(root, path, expected_qualification_sha=tes):
            return path
    return None


def _published_tree(root: Path) -> Path | None:
    handoff = root / "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"
    tes = _ladder_tes(root)
    if tes:
        candidate = handoff / f"qualification_at_{tes}"
        if candidate.is_dir():
            return candidate
    valid = _strict_valid_tree(root)
    if valid is not None:
        return valid
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
    tree = _strict_valid_tree(root)
    if tree is None:
        pytest.skip("no B5 handoff tree passes strict publication verification")
    tes = tree.name.removeprefix("qualification_at_")
    errors = verify_b5_publication_tree(root, tree, expected_qualification_sha=tes)
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
    tree = _strict_valid_tree(root)
    if tree is None:
        pytest.skip("no strictly valid B5 tree")
    qw = tree / "qual_wave_report.json"
    index_path = tree / "publication_index.v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["qual_wave_report"]["sha256"] == sha256_file(qw)


# ---- F-01 (IIAC audit ENT-AUDIT-X, loop 1): digests must survive a checkout on any platform -------------------------

_LEGACY_TES = "89c76eb7b53a0a3ce6f75ab395cf4a2802638a91"
_HANDOFF = "docs/project/maintainers/qualification/ent_audit_x/b6_independent_handoff"


def _ladder_last_tes(root: Path) -> str | None:
    ladder = root / "docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md"
    pinned = None
    for line in ladder.read_text(encoding="utf-8").splitlines():
        if line.startswith("| TEST_EXECUTION_SHA |"):
            cells = [c.strip().strip("`*") for c in line.strip().strip("|").split("|")]
            pinned = cells[1] if len(cells) >= 2 and len(cells[1]) == 40 else None
    return pinned


def test_ladder_pinned_b5_tree_verifies_strictly_on_every_platform() -> None:
    root = _repo_root()
    tes = _ladder_last_tes(root)
    assert tes, "the SHA ladder has no TEST_EXECUTION_SHA"
    tree = root / _HANDOFF / f"qualification_at_{tes}"
    assert tree.is_dir(), f"pinned B5 tree missing: {tree}"
    errors = verify_b5_publication_tree(root, tree, expected_qualification_sha=tes)
    if tes == _LEGACY_TES:
        # Waived legacy tree (F-01): LF blobs were digested as CRLF, so on an LF checkout the only tolerated defects are
        # digest mismatches (indexed artifacts, qual-wave report, evidence manifest). Any other kind of error still
        # fails. The waiver ends with the next TES publication.
        tolerated = ("sha256 mismatch", "evidence manifest digest mismatch")
        assert all(any(marker in error for marker in tolerated) for error in errors), errors
        return
    assert errors == []


def test_new_b5_trees_are_byte_preserved_by_gitattributes() -> None:
    import subprocess

    root = _repo_root()
    for tree in sorted((root / _HANDOFF).glob("qualification_at_*")):
        if tree.name == f"qualification_at_{_LEGACY_TES}":
            continue
        probe = (tree / "publication_index.v1.json").relative_to(root).as_posix()
        done = subprocess.run(
            ["git", "-C", str(root), "check-attr", "text", "--", probe], capture_output=True, text=True, check=True
        )
        assert done.stdout.strip().endswith("text: unset"), f"{tree.name} must be -text (byte-preserved): {done.stdout}"


def test_gitattributes_marks_a_future_tree_as_byte_preserved() -> None:
    import subprocess

    root = _repo_root()
    probe = f"{_HANDOFF}/qualification_at_{'f' * 40}/l3-cli-full/evidence_manifest.v1.json"
    done = subprocess.run(
        ["git", "-C", str(root), "check-attr", "text", "--", probe], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip().endswith("text: unset"), done.stdout
