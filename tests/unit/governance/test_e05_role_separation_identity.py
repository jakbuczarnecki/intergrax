"""Negative tests and controls for the IIAC validator hardening (ticket T-IGX-044-F, node PT-E05).

NT-E05-01 is the RW-PLAN row NT-ROW-010 (EQO-A finding A2-038: independence rested on string comparison and role
labels). Every test builds a throw-away git repository in the pytest temp directory, so nothing tracked is written,
and there is no network. Each test names the weakness it targets and asserts an exit code or an error text.

Groups: role and identity (01-12), manifest (13-18), gate and delegation (19-24), wiring (25-28), controls (C1-C3).
Helpers that only exist after the change (``iiac_activation_gate._corpus_errors``, ``_manifest_state_errors``,
``iiac_common.manifest_countersign_digest``) are reached by attribute access inside test bodies, never by a
module-level import, so that the file collects at the pre-change commit and fails at its own assertions there.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts.governance import iiac_activation_gate, iiac_common
from scripts.governance.iiac_common import validate_manifest_dict

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts/governance"
QUAL = REPO / "docs/project/maintainers/qualification"
POLICY = "IIAC-2026-11-05.1"
STAGE = "IIAC-GOV-1"
ROLES = ("implementer", "auditor", "certifier")
IDS = {
    "implementer": "local_11111111-aaaa-4aaa-8aaa-111111111111",
    "auditor": "local_22222222-bbbb-4bbb-8bbb-222222222222",
    "certifier": "local_33333333-cccc-4ccc-8ccc-333333333333",
}
_COUNTER = itertools.count(1)


# --------------------------------------------------------------------------- throw-away repository helpers


def _git(repo: Path, *args: str) -> str:
    env = {"GIT_AUTHOR_DATE": "2026-11-17T09:00:00+00:00", "GIT_COMMITTER_DATE": "2026-11-17T09:00:00+00:00"}
    proc = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "user.name=e05",
            "-c",
            "user.email=e05@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return proc.stdout.strip()


def _commit(repo: Path, files: dict[str, bytes], message: str) -> str:
    for rel, data in files.items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        _git(repo, "add", "--", rel)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _independent_countersign_digest(manifest: dict[str, Any]) -> str:
    """The digest a countersignature binds to, written independently of the code under test (see the protocol)."""
    body = {key: value for key, value in manifest.items() if key != "handoff_record_ref"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return _sha256(canonical.encode("utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return path


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


# --------------------------------------------------------------------------- manifest world (v2 final)


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repo = tmp_path_factory.mktemp("e05_world")
    _git(repo, "init", "-q", "-b", "main")
    shas: dict[str, str] = {}
    shas["A"] = _commit(repo, {"src/impl.txt": b"implementation\n"}, "audited implementation")
    receipts = {role: f"receipts/{role}.txt" for role in ROLES}
    shas["R"] = _commit(
        repo,
        {receipts[r]: f"host session export\nsession_uuid: {IDS[r]}\n".encode() for r in ROLES},
        "host receipts",
    )
    shas["X"] = _commit(repo, {"audit/work.txt": b"audit execution\n"}, "audit execution")
    evidence = {
        "evidence/report.md": b"audit report\n",
        "evidence/scope.json": b'{"scope": "fixture"}\n',
        "evidence/matrix.txt": b"matrix\n",
        "evidence/findings.json": b'{"findings": []}\n',
    }
    shas["E"] = _commit(repo, evidence, "evidence publication")
    shas["C"] = _commit(repo, {"cert/c.txt": b"certification\n"}, "certification publication")
    _git(repo, "checkout", "-q", "-b", "side-gov")
    shas["G"] = _commit(repo, {"scripts/governance/tamper.py": b"x = 1\n"}, "auditor governance edit")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "checkout", "-q", "-b", "side-product")
    shas["P"] = _commit(repo, {"intergrax/product.py": b"y = 2\n"}, "auditor product edit")
    _git(repo, "checkout", "-q", "main")

    manifest: dict[str, Any] = {
        "manifest_schema_version": "v2",
        "manifest_state": "final",
        "stage_id": "E05-FIXTURE",
        "certification_profile": "iiac-v1",
        "policy_version": POLICY,
        "implementation_session_id": IDS["implementer"],
        "implementation_agent_id": "implementer-agent",
        "auditor_session_id": IDS["auditor"],
        "auditor_agent_id": "auditor-agent",
        "certifier_session_id": IDS["certifier"],
        "certifier_agent_id": "certifier-agent",
        "audited_implementation_sha": shas["A"],
        "test_execution_sha": shas["A"],
        "audit_execution_sha": shas["X"],
        "evidence_publication_sha": shas["E"],
        "certification_publication_sha": shas["C"],
        "audit_report_digest": _sha256(evidence["evidence/report.md"]),
        "audit_report_path": "evidence/report.md",
        "audit_scope_manifest_digest": _sha256(evidence["evidence/scope.json"]),
        "audit_scope_path": "evidence/scope.json",
        "test_matrix_digest": _sha256(evidence["evidence/matrix.txt"]),
        "test_matrix_path": "evidence/matrix.txt",
        "findings_digest": _sha256(evidence["evidence/findings.json"]),
        "findings_path": "evidence/findings.json",
        "verdict": "PASS",
        "limitations": ["synthetic fixture - not a certification"],
        "identity_provenance": {
            role: {
                "issuer": "host-session-export",
                "session_uuid": IDS[role],
                "started_at_utc": "2026-11-17T08:00:00Z",
                "host_receipt_ref": f"{receipts[role]}@{shas['R']}",
            }
            for role in ROLES
        },
    }
    state = {"repo": repo, "shas": shas, "manifest": manifest, "receipts": receipts}
    # Set-up guard: the unmodified manifest is clean, so a negative test cannot pass for an unrelated reason.
    _countersigned(state, manifest)  # the base manifest names a handoff record that countersigns exactly its digest
    path = _write_json(repo.parent / "e05_world_good.json", manifest)
    audit = _run("verify_iiac_audit_manifest.py", str(path), "--repo", str(repo))
    separation = _run("verify_iiac_role_separation.py", str(path), "--repo", str(repo))
    assert audit.returncode == 0, audit.stderr
    assert separation.returncode == 0, separation.stderr
    return state


def _countersigned(
    state: dict[str, Any], manifest: dict[str, Any], *, digest: str | None = None, line: bool = True
) -> dict[str, Any]:
    """Commit a handoff record that carries (or lacks) the countersignature and point the manifest at it."""
    repo: Path = state["repo"]
    rel = f"handoff/handoff_{next(_COUNTER)}.md"
    text = "# Handoff record\n"
    if line:
        bound = digest or _independent_countersign_digest(manifest)
        text += f"Countersigned: operator @ 2026-11-17T12:00:00+01:00 manifest-digest {bound}\n"
    sha = _commit(repo, {rel: text.encode()}, "handoff record")
    manifest["handoff_record_ref"] = f"{rel}@{sha}"
    return manifest


def _fresh(world: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(world["manifest"])


def _separation(world: dict[str, Any], tmp_path: Path, manifest: dict[str, Any], *extra: str):
    path = _write_json(tmp_path / "manifest.json", manifest)
    return _run("verify_iiac_role_separation.py", str(path), "--repo", str(world["repo"]), *extra)


def _audit(world: dict[str, Any], tmp_path: Path, manifest: dict[str, Any], *extra: str):
    path = _write_json(tmp_path / "manifest.json", manifest)
    return _run("verify_iiac_audit_manifest.py", str(path), "--repo", str(world["repo"]), *extra)


def _v1(manifest: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in manifest.items() if k not in ("manifest_schema_version", "manifest_state")}
    out.pop("identity_provenance", None)
    return out


# --------------------------------------------------------------------------- role and identity (12)


def test_nt_e05_01_sessions_differing_only_in_case_or_whitespace_are_rejected(world, tmp_path) -> None:
    """A2-038 / NT-ROW-010: independence rested on raw string comparison of session ids (v1 and v2)."""
    impl = IDS["implementer"]
    variants = {
        "case": impl.upper(),
        "outer whitespace": f"  {impl}\t",
        "case and whitespace": f"\n{impl.title()}  ",
    }
    for name, auditor in variants.items():
        for label, manifest in (("v1", _v1(_fresh(world))), ("v2", _fresh(world))):
            manifest["auditor_session_id"] = auditor
            result = _separation(world, tmp_path, manifest)
            assert result.returncode == 1, f"{label} {name}: {result.stdout}{result.stderr}"
            assert "must differ" in result.stderr, f"{label} {name}: {result.stderr}"


def test_nt_e05_02_one_id_under_two_role_labels_is_rejected(world, tmp_path) -> None:
    """A2-038: a session id may not double as the agent label of another role."""
    manifest = _fresh(world)
    manifest["auditor_agent_id"] = IDS["implementer"]
    result = _separation(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "used as a role label" in result.stderr
    manifest = _v1(_fresh(world))
    manifest["certifier_session_id"] = f" {manifest['implementation_agent_id'].upper()} "
    result = _separation(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "used as a role label" in result.stderr


def test_nt_e05_03_v2_identical_session_uuid_across_roles(world, tmp_path) -> None:
    """A2-038: one host session cannot hold two roles; the recorded uuids must be pairwise different."""
    manifest = _fresh(world)
    manifest["identity_provenance"]["auditor"]["session_uuid"] = IDS["implementer"]
    result = _separation(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "session_uuid" in result.stderr


def test_nt_e05_04_v2_placeholder_identity_is_rejected(world, tmp_path) -> None:
    """A2-006 / A2-038: placeholders such as HISTORICAL_UNRECORDED must not pass as identities of a final manifest."""
    for field, value in (
        ("implementation_session_id", "HISTORICAL_UNRECORDED_IMPLEMENTER"),
        ("certifier_session_id", "PENDING_CERTIFIER"),
    ):
        manifest = _fresh(world)
        manifest[field] = value
        result = _separation(world, tmp_path, manifest)
        assert result.returncode == 1, f"{field}: {result.stderr}"
        assert "placeholder" in result.stderr, f"{field}: {result.stderr}"


def test_nt_e05_05_v2_final_without_identity_provenance(world, tmp_path) -> None:
    """A2-038: a final v2 manifest must carry per-role identity provenance."""
    manifest = _fresh(world)
    del manifest["identity_provenance"]
    result = _separation(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "identity_provenance is required" in result.stderr


def test_nt_e05_06_host_receipt_duplicated_across_stages(world, tmp_path) -> None:
    """A2-038: one host-session export may back only one manifest of the corpus."""
    corpus = tmp_path / "corpus"
    other = _fresh(world)
    other["stage_id"] = "OTHER-STAGE"
    for role in ROLES:
        other["identity_provenance"][role]["session_uuid"] = f"{IDS[role]}-other"
        other[f"{'implementation' if role == 'implementer' else role}_session_id"] = f"{IDS[role]}-other"
    _write_json(corpus / "iiac_audit_session_manifest.other.json", other)
    result = _separation(world, tmp_path / "ours", _fresh(world), "--corpus-root", str(corpus))
    assert result.returncode == 1
    assert "host receipt" in result.stderr and "already used" in result.stderr


def test_nt_e05_07_receipt_not_at_an_ancestor_or_without_the_session_id(world, tmp_path) -> None:
    """A2-038: a receipt must be published before the audit and must contain the recorded session uuid."""
    shas = world["shas"]
    manifest = _fresh(world)
    manifest["identity_provenance"]["auditor"]["host_receipt_ref"] = f"{world['receipts']['auditor']}@{shas['P']}"
    result = _separation(world, tmp_path / "a", manifest)
    assert result.returncode == 1
    assert "not an ancestor of audit_execution_sha" in result.stderr
    manifest = _fresh(world)
    manifest["identity_provenance"]["implementer"]["host_receipt_ref"] = f"src/impl.txt@{shas['A']}"
    result = _separation(world, tmp_path / "b", manifest)
    assert result.returncode == 1
    assert "does not contain the session uuid" in result.stderr


def test_nt_e05_08_v2_auditor_commit_under_governance_scripts(world, tmp_path) -> None:
    """A2-038: the v2 write surface includes scripts/governance, not only the product paths."""
    manifest = _fresh(world)
    manifest["audit_execution_sha"] = world["shas"]["G"]
    result = _separation(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "write-surface" in result.stderr and "scripts/governance/tamper.py" in result.stderr


def test_nt_e05_09_v2_product_diff_fails_without_the_explicit_opt_out(world, tmp_path) -> None:
    """A2-038: for v2 the product diff is a failure by default; --allow-product-diff is the explicit opt-out."""
    manifest = _fresh(world)
    manifest["audit_execution_sha"] = world["shas"]["P"]
    default = _separation(world, tmp_path / "a", manifest)
    assert default.returncode == 1
    assert "intergrax/product.py" in default.stderr
    still_failing = _separation(world, tmp_path / "b", manifest, "--strict-product-diff")
    assert still_failing.returncode == 1
    opted_out = _separation(world, tmp_path / "c", manifest, "--allow-product-diff")
    assert opted_out.returncode == 0, opted_out.stderr


def test_nt_e05_10_missing_countersignature(world, tmp_path) -> None:
    """A2-038 operator line: the countersignature line must exist in the handoff record."""
    manifest = _countersigned(world, _fresh(world), line=False)
    result = _audit(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "Countersigned" in result.stderr


VECTOR_DIGEST = "09ad9fd2fb648cb2f62141215828ea00a62c299db05d20aa9ade2f527a301cc6"


def _audit_raw(world: dict[str, Any], tmp_path: Path, raw: bytes):
    path = tmp_path / "raw_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _run("verify_iiac_audit_manifest.py", str(path), "--repo", str(world["repo"]))


def test_nt_e05_11_countersignature_digest_mismatch(world, tmp_path) -> None:
    """A2-038 operator line: the countersignature must bind to the digest of this manifest (and its canonical form)."""
    manifest = _countersigned(world, _fresh(world), digest="0123456789abcdef" * 4)
    result = _audit(world, tmp_path, manifest)
    assert result.returncode == 1
    assert "does not match the manifest" in result.stderr
    changed = _countersigned(world, _fresh(world))
    changed["limitations"] = ["edited after the countersignature"]
    result = _audit(world, tmp_path / "edited", changed)
    assert result.returncode == 1
    assert "does not match the manifest" in result.stderr
    # The normative test vector of the protocol: the handoff_record_ref is dropped, the rest is canonical JSON.
    vector = {"a": 1, "b": "\u00e9", "handoff_record_ref": "p@" + "0" * 40}
    assert _independent_countersign_digest(vector) == VECTOR_DIGEST
    assert iiac_common.manifest_countersign_digest(vector) == VECTOR_DIGEST
    # Canonical-JSON rules: duplicate keys, floats, a non-object root and a BOM are handled by stable rules.
    good = _countersigned(world, _fresh(world))
    text = json.dumps(good, indent=2, sort_keys=True) + "\n"
    plain = _audit_raw(world, tmp_path / "plain", text.encode("utf-8"))
    assert plain.returncode == 0, plain.stderr
    with_bom = _audit_raw(world, tmp_path / "bom", b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert with_bom.returncode == plain.returncode, with_bom.stderr
    duplicate = _audit_raw(world, tmp_path / "dup", text.replace("{", '{"limitations": [], ', 1).encode("utf-8"))
    assert duplicate.returncode == 1 and "duplicate key" in duplicate.stderr.lower(), duplicate.stderr
    floaty = dict(good, extra_number=1.5)
    floated = _audit_raw(world, tmp_path / "float", (json.dumps(floaty, indent=2) + "\n").encode("utf-8"))
    assert floated.returncode == 1 and "float" in floated.stderr.lower(), floated.stderr
    rooted = _audit_raw(world, tmp_path / "root", b'["not", "an", "object"]\n')
    assert rooted.returncode == 1 and "root must be a json object" in rooted.stderr.lower(), rooted.stderr
    for failing in (duplicate, floated, rooted):
        assert "Traceback" not in failing.stderr, failing.stderr


def _corpus_repo(
    tmp_path: Path, manifests: dict[str, dict[str, Any]], listed: dict[str, str | None]
) -> tuple[Path, Path]:
    """A throw-away repository with manifests under the qualification root and a known-draft allowlist.

    ``listed`` maps a manifest path to the digest recorded for it (``None`` = the true LF digest).
    """
    repo = tmp_path / "corpus_repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    files: dict[str, bytes] = {}
    for rel, manifest in manifests.items():
        files[rel] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    entries = [
        {"path": rel, "sha256": digest or _sha256(files[rel]), "reason": "fixture draft"}
        for rel, digest in listed.items()
    ]
    allowlist = {"schema": "iiac_known_draft_manifests.v1", "entries": entries}
    files["scripts/governance/iiac_known_draft_manifests.v1.json"] = json.dumps(allowlist).encode()
    _commit(repo, files, "corpus fixture")
    return repo, repo / next(iter(manifests))


def _draft_manifest(world: dict[str, Any]) -> dict[str, Any]:
    manifest = _v1(_fresh(world))
    manifest["stage_id"] = "FIXTURE-DRAFT"
    return manifest


QUAL_REL = "docs/project/maintainers/qualification"


def test_nt_e05_12_v2_manifest_listed_in_the_known_draft_allowlist(world, tmp_path) -> None:
    """A2-006: listing a v2 manifest as a known draft would let it dodge placeholder rejection."""
    manifest = _fresh(world)
    manifest["implementation_session_id"] = "HISTORICAL_UNRECORDED_IMPLEMENTER"
    manifest["findings_digest"] = ""  # invalid under the v1 rules as well: only the allowlist could hide it
    problems: list[str] = []
    validate_manifest_dict(manifest, problems)
    assert problems, "the manifest must be invalid on its own, the allowlist is the only thing that could hide it"
    rel = f"{QUAL_REL}/iiac_audit_session_manifest.v2draft.json"
    repo, _path = _corpus_repo(tmp_path, {rel: manifest}, {rel: None})
    errors = iiac_activation_gate._corpus_errors(repo)
    assert any("ACT-N21" in err and "v2 manifest" in err for err in errors), errors


# --------------------------------------------------------------------------- manifest (6)


def test_nt_e05_13_unknown_schema_marker_or_state_fails_closed(world, tmp_path) -> None:
    """A2-006: an unknown marker or state must not fall back to the lenient v1 reading."""
    manifest = _fresh(world)
    manifest["manifest_schema_version"] = "v3"
    result = _audit(world, tmp_path / "a", manifest)
    assert result.returncode == 1 and "manifest_schema_version" in result.stderr
    manifest = _fresh(world)
    manifest["manifest_state"] = "published"
    result = _audit(world, tmp_path / "b", manifest)
    assert result.returncode == 1 and "manifest_state" in result.stderr
    manifest = _v1(_fresh(world))
    manifest["manifest_state"] = "final"
    result = _audit(world, tmp_path / "c", manifest)
    assert result.returncode == 1 and "needs manifest_schema_version" in result.stderr


def _gate_call_count(function: str, callee: str) -> int:
    """How often the body of a function of the gate calls ``callee`` (parsed, so a comment cannot fake it)."""
    tree = ast.parse((SCRIPTS / "iiac_activation_gate.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return sum(
                1
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and getattr(call.func, "id", getattr(call.func, "attr", "")) == callee
            )
    raise AssertionError(f"function {function} not found in the gate")


def test_nt_e05_14_draft_used_as_certifying(world, tmp_path) -> None:
    """A2-006: a draft may exist but cannot back an activation (ACT-N20), and the gate calls that check once."""
    manifest = _countersigned(world, _fresh(world))
    manifest["manifest_state"] = "draft"
    manifest.pop("handoff_record_ref")
    accepted = _audit(world, tmp_path / "a", manifest)
    assert accepted.returncode == 0, accepted.stderr
    errors = iiac_activation_gate._manifest_state_errors(manifest)
    assert any("ACT-N20" in err and "draft" in err for err in errors), errors
    unsupported = dict(manifest, manifest_schema_version="v3", manifest_state="final")
    errors = iiac_activation_gate._manifest_state_errors(unsupported)
    assert any("ACT-N20" in err for err in errors), errors
    assert iiac_activation_gate._manifest_state_errors(dict(manifest, manifest_state="final")) == []
    assert _gate_call_count("evaluate_activation", "_manifest_state_errors") == 1


def test_nt_e05_15_final_with_a_non_passing_verdict(world, tmp_path) -> None:
    """A2-006: a final v2 manifest needs PASS or PASS_SCOPED."""
    for verdict in ("FAIL", "CONDITIONAL"):
        manifest = _countersigned(world, _fresh(world))
        manifest["verdict"] = verdict
        result = _audit(world, tmp_path / verdict, manifest)
        assert result.returncode == 1, verdict
        assert "verdict" in result.stderr, verdict


def test_nt_e05_16_final_with_a_placeholder_string(world, tmp_path) -> None:
    """A2-006: placeholders in digest or identity fields cannot pass as a final manifest."""
    for field, value in (("findings_digest", "TBD_findings_digest_value"), ("auditor_agent_id", "PENDING_AUDITOR")):
        manifest = _countersigned(world, _fresh(world))
        manifest[field] = value
        result = _audit(world, tmp_path / field, manifest)
        assert result.returncode == 1, field
        assert "placeholder" in result.stderr.lower(), f"{field}: {result.stderr}"


def _artifact_run(world: dict[str, Any], tmp_path: Path, manifest: dict[str, Any]):
    path = _write_json(tmp_path / "manifest.json", manifest)
    args = (str(path), "--repo", str(world["repo"]), "--profile", "final")
    return path, _run("verify_iiac_manifest_artifacts.py", *args)


def _artifact_failures(world: dict[str, Any], path: Path) -> list[str]:
    errors = iiac_activation_gate._companion_errors(world["repo"], path)
    return [err for err in errors if "verify_iiac_manifest_artifacts.py failed" in err]


def test_nt_e05_17_final_with_a_missing_sha_object_or_a_strict_chain_violation(world, tmp_path) -> None:
    """A2-006: SHAs must be commit objects and follow audited <= test <= audit <= evidence <= certification."""
    for name, field, value, code in (
        ("missing", "certification_publication_sha", "ab" * 20, "SHA_MISSING"),
        ("order", "evidence_publication_sha", world["shas"]["R"], "SHA_ORDER"),  # before the audit execution commit
    ):
        manifest = _countersigned(world, _fresh(world))
        manifest[field] = value
        path, direct = _artifact_run(world, tmp_path / name, manifest)
        assert _artifact_failures(world, path), f"{name}: the gate must run the artifact verifier and report it"
        assert direct.returncode == 1 and code in direct.stderr, f"{name}: {direct.stderr}"
    good = _countersigned(world, _fresh(world))
    path, _direct = _artifact_run(world, tmp_path / "good", good)
    assert _artifact_failures(world, path) == []


def test_nt_e05_18_final_with_a_digest_not_bound_to_file_bytes(world, tmp_path) -> None:
    """A2-006: the digests are bound to the bytes of the files they name."""
    manifest = _countersigned(world, _fresh(world))
    manifest["audit_report_digest"] = _sha256(b"some other report")
    path, direct = _artifact_run(world, tmp_path / "a", manifest)
    assert _artifact_failures(world, path), "the gate must run the artifact verifier and report it"
    assert direct.returncode == 1 and "DIGEST_MISMATCH" in direct.stderr, direct.stderr
    manifest = _countersigned(world, _fresh(world))
    del manifest["findings_path"]
    path, direct = _artifact_run(world, tmp_path / "b", manifest)
    assert _artifact_failures(world, path), "the gate must run the artifact verifier and report it"
    assert direct.returncode == 1 and "DIGEST_UNBOUND" in direct.stderr, direct.stderr


# --------------------------------------------------------------------------- gate and delegation (6)

DECISION_DIGEST = "a" * 64
IMPLEMENTER = "local_impl-aaaa"
RECORD_GOOD = f"# Delegation record\nStage: {STAGE}\nDelegate session: local_deleg-1111\n"
RECORD_IMPL = f"# Delegation record\nStage: {STAGE}\nDelegate session: {IMPLEMENTER.upper()} \n"


@pytest.fixture(scope="module")
def deleg(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    repo = tmp_path_factory.mktemp("e05_deleg")
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(repo, {"README.md": b"base\n"}, "base")
    records = _commit(
        repo,
        {
            "records/good.md": RECORD_GOOD.encode(),
            "records/other.md": b"# Delegation record\nStage: OTHER-STAGE-9\nDelegate session: local_deleg-1111\n",
            "records/impl.md": RECORD_IMPL.encode(),
        },
        "delegation records",
    )
    _git(repo, "checkout", "-q", "-b", "side")
    side = _commit(repo, {"records/side.md": RECORD_GOOD.encode()}, "side delegation")
    _git(repo, "checkout", "-q", "main")
    tip = _commit(repo, {"tip.txt": b"tip\n"}, "tip")
    state = {"repo": repo, "base": base, "records": records, "side": side, "tip": tip}
    # Set-up guard: the good delegate record resolves, so the negative tests fail for their own reason only.
    assert _operator(state, "delegate", f"records/good.md@{records}") == []
    return state


def _operator(
    state: dict[str, Any],
    kind: str | None,
    ref: str | None,
    *,
    schema: str | None = "v2",
    text_extra: str = "",
) -> list[str]:
    who = "operator-delegate" if kind == "delegate" else "operator"
    line = f"Approved IIAC policy activation {POLICY}: {who} @ 2026-11-17T12:00:00+01:00"
    record: dict[str, Any] = {"operator_decision_id": "E05-001", "operator_activation_line": line}
    if schema:
        record["activation_record_schema_version"] = schema
    if kind:
        record["operator_line_kind"] = kind
    if ref:
        record["operator_delegation_ref"] = ref
    text = (
        "Decision ID: E05-001\n"
        f"Policy version: {POLICY}\n"
        "Decision scope: IIAC policy activation\n"
        "Decision status: AUTHORIZED\n"
        f"Audit report digest: {DECISION_DIGEST}\n"
        f"{line}\n{text_extra}"
    )
    manifest = {"stage_id": STAGE, "implementation_session_id": IMPLEMENTER}
    return iiac_activation_gate._operator_errors(
        record, text, DECISION_DIGEST, repo=state["repo"], head=state["tip"], manifest=manifest
    )


def test_nt_e05_19_delegate_line_without_a_record(deleg) -> None:
    """A2-038 operator line: a delegate line was only a format check; it must resolve to a record."""
    errors = _operator(deleg, "delegate", None)
    assert any("ACT-N19" in err and "operator_delegation_ref" in err for err in errors), errors
    errors = _operator(deleg, "delegate", f"records/missing.md@{deleg['records']}")
    assert any("ACT-N19" in err and "does not exist" in err for err in errors), errors


def test_nt_e05_20_delegation_record_not_an_ancestor_of_head(deleg) -> None:
    """A2-038 operator line: a record that is not in the history of HEAD proves nothing."""
    errors = _operator(deleg, "delegate", f"records/side.md@{deleg['side']}")
    assert any("not an ancestor" in err for err in errors), errors


def test_nt_e05_21_delegation_record_naming_another_stage(deleg) -> None:
    """A2-038 operator line: the record must name the stage of the manifest it delegates."""
    errors = _operator(deleg, "delegate", f"records/other.md@{deleg['records']}")
    assert any("does not name the stage" in err for err in errors), errors


def test_nt_e05_22_personal_form_line_pointing_at_a_delegation_file(deleg) -> None:
    """A2-038 operator line: a personal-form line may not borrow a delegation record."""
    ref = f"records/good.md@{deleg['records']}"
    errors = _operator(deleg, "personal", ref)
    assert any("may not point at a delegation record" in err for err in errors), errors
    errors = _operator(deleg, "personal", None, text_extra=f"Delegation: {ref}\n")
    assert any("may not point at a delegation record" in err for err in errors), errors


def test_nt_e05_23_new_record_without_operator_line_kind(deleg) -> None:
    """A2-038 operator line: a new (v2) activation record must say which kind of operator line it carries."""
    errors = _operator(deleg, None, None)
    assert any("must carry operator_line_kind" in err for err in errors), errors
    errors = _operator(deleg, "witness", None)
    assert any("operator_line_kind must be one of" in err for err in errors), errors


def test_nt_e05_24_delegate_equal_to_the_implementer(deleg) -> None:
    """A2-038 operator line: the delegate must not be the implementer (compared after normalisation)."""
    errors = _operator(deleg, "delegate", f"records/impl.md@{deleg['records']}")
    assert any("names the implementer as the delegate" in err for err in errors), errors


# --------------------------------------------------------------------------- wiring (4)


def test_nt_e05_25_unlisted_invalid_manifest_makes_the_corpus_check_fail(world, tmp_path) -> None:
    """A2-006: the corpus verifier is wired into the gate once, so an unlisted invalid manifest fails it."""
    rel = f"{QUAL_REL}/iiac_audit_session_manifest.invalid.json"
    invalid = {"stage_id": "INVALID-FIXTURE"}
    repo, _path = _corpus_repo(tmp_path / "listed", {rel: invalid}, {rel: None})
    listed_errors = iiac_activation_gate._corpus_errors(repo)
    assert not any("verify_iiac_manifest_corpus.py failed" in err for err in listed_errors), listed_errors
    repo, _path = _corpus_repo(tmp_path / "unlisted", {rel: invalid}, {})
    errors = iiac_activation_gate._corpus_errors(repo)
    assert any("ACT-N10 companion verify_iiac_manifest_corpus.py failed" in err for err in errors), errors
    assert _gate_call_count("evaluate_activation", "_corpus_errors") == 1


def test_nt_e05_26_changed_known_draft_digest_makes_the_corpus_check_fail(world, tmp_path) -> None:
    """A2-006: a listed draft whose bytes changed after it was listed fails the wired corpus call."""
    rel = f"{QUAL_REL}/iiac_audit_session_manifest.invalid.json"
    repo, _path = _corpus_repo(tmp_path, {rel: {"stage_id": "INVALID-FIXTURE"}}, {rel: "f" * 64})
    errors = iiac_activation_gate._corpus_errors(repo)
    assert any("verify_iiac_manifest_corpus.py failed" in err for err in errors), errors


def test_nt_e05_27_each_wired_file_is_a_material_governance_path() -> None:
    """The gate's meaning changed: every file the wired calls execute or read must be covered by ACT-N11."""
    wired = (
        "scripts/governance/iiac_verifier_common.py",
        "scripts/governance/verify_iiac_manifest_corpus.py",
        "scripts/governance/verify_iiac_manifest_artifacts.py",
        "scripts/governance/iiac_known_draft_manifests.v1.json",
    )
    gate_text = (SCRIPTS / "iiac_activation_gate.py").read_text(encoding="utf-8")
    for rel in wired:
        assert rel in iiac_activation_gate.MATERIAL_GOVERNANCE_PATHS, rel
        assert (REPO / rel).is_file(), rel
    assert "verify_iiac_manifest_corpus.py" in gate_text and "verify_iiac_manifest_artifacts.py" in gate_text
    assert len(iiac_activation_gate.MATERIAL_GOVERNANCE_PATHS) == 10


def test_nt_e05_28_companion_errors_alone_runs_no_corpus_check(world, tmp_path) -> None:
    """Regression guard: the EVOL-X preflight imports _companion_errors, which must not start a corpus run."""
    rel = f"{QUAL_REL}/iiac_audit_session_manifest.subject.json"
    other = f"{QUAL_REL}/iiac_audit_session_manifest.unlisted_invalid.json"
    subject = _v1(_fresh(world))
    subject["stage_id"] = "SUBJECT"
    clean_repo, clean_path = _corpus_repo(tmp_path / "clean", {rel: subject}, {})
    noisy_repo, noisy_path = _corpus_repo(tmp_path / "noisy", {rel: subject, other: {"stage_id": "BROKEN"}}, {})
    clean = iiac_activation_gate._companion_errors(clean_repo, clean_path)
    noisy = iiac_activation_gate._companion_errors(noisy_repo, noisy_path)
    assert not any("verify_iiac_manifest_corpus" in err for err in noisy), noisy

    def normal(errors: list[str], repo: Path, path: Path) -> list[str]:
        return [e.replace(str(path), "<M>").replace(str(repo), "<R>") for e in errors]

    assert normal(noisy, noisy_repo, noisy_path) == normal(clean, clean_repo, clean_path)
    assert _gate_call_count("_companion_errors", "_corpus_errors") == 0


# --------------------------------------------------------------------------- controls (3)


def test_c1_published_manifests_keep_their_verdicts() -> None:
    """Control: the published manifests keep today's verdict; exactly the listed drafts are invalid."""
    allowlist = json.loads((SCRIPTS / "iiac_known_draft_manifests.v1.json").read_text(encoding="utf-8"))
    listed = {entry["path"] for entry in allowlist["entries"]}
    invalid: set[str] = set()
    valid = 0
    for path in sorted(QUAL.rglob("iiac_audit_session_manifest*.json")):
        if path.name.endswith(".schema.json"):
            continue
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        assert "manifest_schema_version" not in manifest, f"{path} unexpectedly carries the v2 marker"
        errors: list[str] = []
        validate_manifest_dict(manifest, errors)
        if errors:
            invalid.add(path.relative_to(REPO).as_posix())
        else:
            valid += 1
    assert invalid == listed
    assert valid >= 38


def test_c2_v1_activation_record_still_passes_its_own_companion_calls() -> None:
    """Control: the companion calls of the current (v1) activation record, now with the wired verifiers, pass."""
    record = json.loads((QUAL / "IIAC_POLICY_ACTIVATION_RECORD.json").read_text(encoding="utf-8"))
    manifest = REPO / record["session_b_manifest_path"]
    assert manifest.is_file()
    assert iiac_activation_gate._companion_errors(REPO, manifest) == []
    assert iiac_activation_gate._corpus_errors(REPO) == []  # last statement: the live corpus passes the corpus check


def test_c3_corpus_check_is_skipped_without_the_allowlist_or_the_qualification_root(tmp_path) -> None:
    """Control: the corpus verifier runs only when the evaluated repository has its allowlist and its corpus root."""
    broken = json.dumps({"stage_id": "BROKEN"}).encode()
    no_allowlist = tmp_path / "no_allowlist"
    no_allowlist.mkdir()
    _git(no_allowlist, "init", "-q", "-b", "main")
    _commit(no_allowlist, {f"{QUAL_REL}/iiac_audit_session_manifest.broken.json": broken}, "no allowlist")
    assert iiac_activation_gate._corpus_errors(no_allowlist) == []
    no_root = tmp_path / "no_root"
    no_root.mkdir()
    _git(no_root, "init", "-q", "-b", "main")
    allowlist = json.dumps({"schema": "iiac_known_draft_manifests.v1", "entries": []}).encode()
    _commit(
        no_root,
        {
            "scripts/governance/iiac_known_draft_manifests.v1.json": allowlist,
            "elsewhere/iiac_audit_session_manifest.broken.json": broken,
        },
        "no qualification root",
    )
    assert iiac_activation_gate._corpus_errors(no_root) == []
