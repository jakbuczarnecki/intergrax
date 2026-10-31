# © Artur Czarnecki. All rights reserved.

"""Protocol 03 immutable dossier verification and executable tree identity (B5-F01/F02)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_REPLAY_REL = "docs/project/maintainers/qualification/ebh_7_p03_independent_replay"
_MANIFEST_REL = f"{_REPLAY_REL}/MANIFEST.json"
_DIGESTS_REL = f"{_REPLAY_REL}/ARTIFACT_DIGESTS.json"

# Documentation / CI / P03 B5 meta-gates only — must not change certified executable semantics.
_EXECUTABLE_IDENTITY_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "docs/project/maintainers/qualification/",
    ".github/",
    "scripts/qualification/verify_ebh7_p03_immutable_evidence.py",
)
_EXECUTABLE_IDENTITY_ALLOWLIST_EXACT: frozenset[str] = frozenset(
    {
        "tests/qualification/ebh_7/test_ebh_7_p03_b5_immutable_gates.py",
        "tests/qualification/ebh_7/test_ebh_7_b6_f10_f11_corrective_gates.py",
        "tests/qualification/ebh_7/_ebh7_f10_ecap_normative_matrix.py",
        "tests/qualification/ebh_7/_constants.py",
        "scripts/qualification/render_ebh7_p03_artifact_digests.py",
    }
)


def _path_allowed(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized in _EXECUTABLE_IDENTITY_ALLOWLIST_EXACT:
        return True
    return any(normalized.startswith(prefix) for prefix in _EXECUTABLE_IDENTITY_ALLOWLIST_PREFIXES)


def _git_show(sha: str, path: str) -> bytes:
    posix = path.replace("\\", "/")
    return subprocess.check_output(["git", "show", f"{sha}:{posix}"], cwd=_REPO)


def _git_is_ancestor(ancestor: str, head: str) -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, head],
        cwd=_REPO,
        capture_output=True,
    )
    return proc.returncode == 0


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _resolve_replay_path(rel: str) -> str:
    if rel.startswith("../"):
        return str(Path(_REPLAY_REL).parent / rel[3:]).replace("\\", "/")
    return f"{_REPLAY_REL}/{rel}".replace("\\", "/")


def verify_immutable_digests_at_git_sha(
    git_tree_sha: str,
    report_lines: list[str] | None = None,
) -> None:
    """Verify ARTIFACT_DIGESTS at git_tree_sha against git object bytes (no working-tree mutation)."""
    lines = report_lines if report_lines is not None else []
    lines.append(f"immutable_git_objects git_tree_sha={git_tree_sha}")
    payload = json.loads(_git_show(git_tree_sha, _DIGESTS_REL))
    recorded: dict[str, str] = payload.get("sha256", {})
    if not recorded:
        raise AssertionError("ARTIFACT_DIGESTS sha256 map empty")
    for rel, expected in recorded.items():
        git_path = _resolve_replay_path(rel)
        actual = _sha256_bytes(_git_show(git_tree_sha, git_path))
        status = "PASS" if actual == expected else "FAIL"
        lines.append(f"{status} rel={rel} expected={expected} actual={actual}")
        if actual != expected:
            raise AssertionError(f"Immutable digest mismatch for {rel} @ {git_tree_sha}")
    lines.append("immutable_git_objects=PASS")


def verify_immutable_digests_working_tree(report_lines: list[str] | None = None) -> None:
    """Verify dossier bytes @ HEAD against ARTIFACT_DIGESTS on disk (read-only, no mutation)."""
    lines = report_lines if report_lines is not None else []
    lines.append("immutable_working_tree begin")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    payload = json.loads((_REPO / _DIGESTS_REL).read_text(encoding="utf-8"))
    recorded: dict[str, str] = payload.get("sha256", {})
    for rel, expected in recorded.items():
        git_path = _resolve_replay_path(rel)
        actual = _sha256_bytes(_git_show(head, git_path))
        status = "PASS" if actual == expected else "FAIL"
        lines.append(f"{status} rel={rel} expected={expected} actual={actual}")
        if actual != expected:
            raise AssertionError(f"Digest mismatch for {rel}")
    lines.append("immutable_working_tree=PASS")


def verify_manifest_pins(
    *,
    test_execution_sha: str,
    evidence_publication_sha: str,
    report_lines: list[str] | None = None,
) -> None:
    manifest_path = _REPO / _MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = report_lines if report_lines is not None else []
    test_pin = manifest["TEST_EXECUTION_SHA"]
    evidence_pin = manifest["EVIDENCE_PUBLICATION_SHA"]
    lines.append(f"manifest_test_pin={test_pin}")
    lines.append(f"manifest_evidence_pin={evidence_pin}")
    if test_pin != test_execution_sha:
        raise AssertionError(f"MANIFEST TEST_EXECUTION_SHA {test_pin} != {test_execution_sha}")
    if evidence_pin != evidence_publication_sha:
        raise AssertionError(f"MANIFEST EVIDENCE_PUBLICATION_SHA {evidence_pin} != {evidence_publication_sha}")
    lines.append("manifest_pins=PASS")


def verify_executable_tree_identity(
    test_execution_sha: str,
    replay_head_sha: str,
    report_lines: list[str] | None = None,
) -> None:
    """Fail-closed if replay differs from TEST outside documentation/CI allowlist."""
    lines = report_lines if report_lines is not None else []
    if not _git_is_ancestor(test_execution_sha, replay_head_sha):
        raise AssertionError(f"TEST_EXECUTION {test_execution_sha} not ancestor of replay {replay_head_sha}")
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", test_execution_sha, replay_head_sha],
        cwd=_REPO,
        text=True,
    )
    changed = [ln.strip() for ln in diff.splitlines() if ln.strip()]
    lines.append(f"diff_paths_count={len(changed)}")
    violations = [p for p in changed if not _path_allowed(p)]
    for p in changed:
        tag = "allowlist" if _path_allowed(p) else "VIOLATION"
        lines.append(f"  {tag}: {p}")
    if violations:
        raise AssertionError(
            "Executable tree identity violated: " + ", ".join(violations[:8]) + ("..." if len(violations) > 8 else "")
        )
    lines.append("executable_tree_identity=PASS")


def verify_git_status_clean(report_lines: list[str] | None = None) -> None:
    lines = report_lines if report_lines is not None else []
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=_REPO,
        text=True,
    )
    if status.strip():
        lines.append(status)
        raise AssertionError("Working tree not clean before pytest")
    lines.append("git_status_clean=PASS")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EBH-7 P03 B5 immutable evidence checks")
    parser.add_argument("--test-execution-sha", required=True)
    parser.add_argument("--evidence-publication-sha", required=True)
    parser.add_argument("--replay-head-sha", default="")
    parser.add_argument(
        "--git-tree-sha",
        default="",
        help="Git commit for git-objects mode (defaults to --replay-head-sha)",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "git-objects",
            "working-tree",
            "identity",
            "manifest-pins",
            "git-clean",
            "replay-disk",
            "all-pre-pytest",
        ),
        required=True,
    )
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)
    lines: list[str] = []
    try:
        if args.mode in ("git-objects", "all-pre-pytest"):
            tree_sha = args.git_tree_sha or args.replay_head_sha
            if not tree_sha:
                raise SystemExit("--git-tree-sha or --replay-head-sha required for git-objects")
            verify_immutable_digests_at_git_sha(tree_sha, lines)
        if args.mode in ("identity", "all-pre-pytest"):
            if not args.replay_head_sha:
                raise SystemExit("--replay-head-sha required for identity")
            verify_executable_tree_identity(args.test_execution_sha, args.replay_head_sha, lines)
        if args.mode in ("manifest-pins", "all-pre-pytest", "replay-disk"):
            verify_manifest_pins(
                test_execution_sha=args.test_execution_sha,
                evidence_publication_sha=args.evidence_publication_sha,
                report_lines=lines,
            )
        if args.mode in ("working-tree", "all-pre-pytest", "replay-disk"):
            verify_immutable_digests_working_tree(lines)
        if args.mode in ("git-clean", "all-pre-pytest", "replay-disk"):
            verify_git_status_clean(lines)
    except AssertionError as exc:
        lines.append(f"RESULT=FAIL reason={exc}")
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), file=sys.stderr)
        return 1
    lines.append("RESULT=PASS")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
