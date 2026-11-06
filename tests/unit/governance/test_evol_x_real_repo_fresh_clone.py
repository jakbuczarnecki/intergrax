# © Artur Czarnecki. All rights reserved.

"""W4-F07 — the REAL IIAC activation gate and EVOL-X preflight in fresh clones of this repository.

The RA-B4-01 fresh-checkout tests in ``tests/unit/qualification/test_evol_x_closure_preflight_b4.py`` build synthetic
repositories with their own activation record, so they cannot see a defect in the committed activation record
(W4-F01). These tests clone the committed ``HEAD`` of the real repository with ``core.autocrlf=true`` and
``core.autocrlf=false`` and run the real ``verify_iiac_policy_activation --require-in-force`` and the real preflight.

W4-F01: ``IIAC_POLICY_ACTIVATION_RECORD.json`` binds ``handoff_digest`` to the CRLF bytes of the handoff record. The
non-material repair is ``text eol=crlf`` in ``.gitattributes`` for that one path: the CRLF bytes are materialised in
every checkout (Windows ``autocrlf`` true/false/input, Linux), so the unchanged digest and the unchanged gate hold.
The activation record, the gate and the blob are untouched. No test here writes into the canonical repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

REPO = Path(__file__).resolve().parents[3]
ACTIVATION_SCRIPT = "scripts/governance/verify_iiac_policy_activation.py"
PREFLIGHT_SCRIPT = "scripts/qualification/evol_x_closure_publication_preflight.py"
QUAL = "docs/project/maintainers/qualification"
PASS_RECORD = f"{QUAL}/evol_x/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
ACTIVATION_RECORD = f"{QUAL}/IIAC_POLICY_ACTIVATION_RECORD.json"
AUDIT_DOSSIER = f"{QUAL}/iiac_evidence/EVOL-X/audit/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md"
CERT_RECORD = f"{QUAL}/iiac_evidence/EVOL-X/certification/EVOL_X_IIAC_CERTIFICATION_RECORD.json"


def _run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False, errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(cwd, "git", "-c", "user.email=w4f07@example.com", "-c", "user.name=w4f07", *args, check=check)


def _fresh_clone(root: Path, autocrlf: bool | str) -> Path:
    flag = autocrlf if isinstance(autocrlf, str) else ("true" if autocrlf else "false")
    dst = root / f"clone_{flag}"
    _git(root, "clone", "-q", "--no-checkout", "-c", f"core.autocrlf={flag}", str(REPO), str(dst))
    _git(dst, "config", "core.autocrlf", flag)
    _git(dst, "fetch", "-q", str(REPO), "HEAD")
    _git(dst, "checkout", "-q", "--detach", "FETCH_HEAD")
    return dst


@pytest.fixture(scope="module")
def clones(tmp_path_factory: pytest.TempPathFactory) -> dict[bool | str, Path]:
    if _git(REPO, "rev-parse", "HEAD", check=False).returncode != 0:
        pytest.skip("not a git checkout")
    root = tmp_path_factory.mktemp("w4f07")
    return {flag: _fresh_clone(root, flag) for flag in (True, False, "input")}


def _env(clone: Path) -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(clone), "PYTHONUTF8": "1"}


def _activation(clone: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ACTIVATION_SCRIPT, "--require-in-force"],
        cwd=clone,
        env=_env(clone),
        capture_output=True,
        text=True,
        check=False,
        errors="replace",
    )


def _preflight(clone: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, PREFLIGHT_SCRIPT, "--gate", "b6-readiness", "--json"],
        cwd=clone,
        env=_env(clone),
        capture_output=True,
        text=True,
        check=False,
        errors="replace",
    )
    assert proc.stdout.strip(), proc.stderr
    return json.loads(proc.stdout)


def _codes(errors: object) -> set[str]:
    return {str(e).split(" ", 1)[0] for e in errors}  # type: ignore[attr-defined]


def _reset(clone: Path) -> None:
    _git(clone, "checkout", "-q", "--detach", "FETCH_HEAD")
    _git(clone, "reset", "-q", "--hard", "FETCH_HEAD")
    _git(clone, "clean", "-qfd")


def test_real_activation_verifies_in_a_fresh_autocrlf_true_clone(clones: dict[bool | str, Path]) -> None:
    proc = _activation(clones[True])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_real_activation_verifies_in_a_fresh_autocrlf_false_clone(clones: dict[bool | str, Path]) -> None:
    proc = _activation(clones[False])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_real_activation_verifies_in_a_fresh_autocrlf_input_clone(clones: dict[bool | str, Path]) -> None:
    proc = _activation(clones["input"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_w4_f01_handoff_bytes_are_crlf_in_every_checkout_mode(clones: dict[bool | str, Path]) -> None:
    """The materialised handoff bytes are the CRLF form the activation record binds, whatever the autocrlf mode."""
    activation = json.loads((clones[True] / ACTIVATION_RECORD).read_text(encoding="utf-8"))
    for flag, clone in clones.items():
        data = (clone / activation["handoff_record_path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == activation["handoff_digest"], flag


def test_w4_f01_a_tampered_handoff_still_fails_act_n08_in_every_checkout_mode(clones: dict[bool | str, Path]) -> None:
    for flag, clone in clones.items():
        activation = json.loads((clone / ACTIVATION_RECORD).read_text(encoding="utf-8"))
        target = clone / activation["handoff_record_path"]
        original = target.read_bytes()
        try:
            for label, mutated in (
                ("plus-one-byte", original + b"x"),
                ("lf-collapsed", original.replace(b"\r\n", b"\n")),
                ("double-crlf", original.replace(b"\r\n", b"\r\r\n")),
            ):
                target.write_bytes(mutated)
                proc = _activation(clone)
                assert proc.returncode != 0 and "ACT-N08 handoff digest mismatch" in proc.stdout + proc.stderr, (
                    flag,
                    label,
                )
            target.unlink()
            assert _activation(clone).returncode != 0, (flag, "deleted")
        finally:
            target.write_bytes(original)


def test_real_preflight_in_a_fresh_autocrlf_true_clone_reports_only_certifier_blockers(
    clones: dict[bool | str, Path],
) -> None:
    result = _preflight(clones[True])
    assert result["mode"] == "IIAC", result["gate_a_errors"]
    # Evidence, scope (A-19), digests and bytes validate; only the certifier's open blockers may hold Gate A.
    assert _codes(result["gate_a_errors"]) <= {"A-18"}, result["gate_a_errors"]
    assert result["repository_class"] == "canonical"


def test_real_preflight_in_a_fresh_autocrlf_false_clone_matches_the_crlf_clone(clones: dict[bool | str, Path]) -> None:
    crlf = _preflight(clones[True])
    for flag in (False, "input"):
        lf = _preflight(clones[flag])
        assert lf["mode"] == "IIAC", lf["gate_a_errors"]
        assert lf["gate_a_errors"] == crlf["gate_a_errors"], flag
        assert lf["gate_b_errors"] == crlf["gate_b_errors"], flag


def test_real_preflight_flags_tampered_evidence_in_a_fresh_clone(clones: dict[bool | str, Path]) -> None:
    clone = clones[True]
    try:
        # N1: one extra byte in the digest-bound pass record
        with (clone / PASS_RECORD).open("ab") as handle:
            handle.write(b"x")
        assert "A-14" in _codes(_preflight(clone)["gate_a_errors"])
        _reset(clone)

        # N2: CRLF expansion of a -text path is a mismatch, never silently normalised
        raw = (clone / PASS_RECORD).read_bytes().replace(b"\r\n", b"\n")
        (clone / PASS_RECORD).write_bytes(raw.replace(b"\n", b"\r\n"))
        assert "A-14" in _codes(_preflight(clone)["gate_a_errors"])
        _reset(clone)

        # N3: CRLF expansion of an indexed, byte-preserved evidence file
        raw = (clone / AUDIT_DOSSIER).read_bytes().replace(b"\r\n", b"\n")
        (clone / AUDIT_DOSSIER).write_bytes(raw.replace(b"\n", b"\r\n"))
        assert "A-13" in _codes(_preflight(clone)["gate_a_errors"])
        _reset(clone)

        # N4: a committed edit of a digest recorded in the certification record
        record = json.loads((clone / CERT_RECORD).read_text(encoding="utf-8"))
        record["pass_record_digest"] = "0" * 64
        (clone / CERT_RECORD).write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
        _git(clone, "commit", "-qam", "digest edit")
        assert "A-14" in _codes(_preflight(clone)["gate_a_errors"])
    finally:
        _reset(clone)


def test_real_preflight_rejects_a_widened_certified_scope_in_a_fresh_clone(clones: dict[bool | str, Path]) -> None:
    """RA-B4-05 on the real evidence: appended widening text, widened verdict and a scope key each add A-19."""
    clone = clones[True]
    manifest_rel = f"{QUAL}/iiac_evidence/EVOL-X/certification/iiac_audit_session_manifest.EVOL-X.json"
    baseline = _preflight(clone)
    assert not [e for e in baseline["gate_a_errors"] if str(e).startswith("A-19")]
    try:
        manifest = json.loads((clone / manifest_rel).read_text(encoding="utf-8"))
        manifest["limitations"] = [
            x + "; platform-wide test ownership is established" if x.startswith("LIM-03") else x
            for x in manifest["limitations"]
        ]
        (clone / manifest_rel).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
        _git(clone, "commit", "-qam", "widen LIM-03")
        assert any(str(e).startswith("A-19") and "LIM-03" in str(e) for e in _preflight(clone)["gate_a_errors"])
        _reset(clone)

        record = json.loads((clone / CERT_RECORD).read_text(encoding="utf-8"))
        record["certification_verdict"] = "PASS_SCOPED_UNRESTRICTED_PLATFORM_WIDE"
        (clone / CERT_RECORD).write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
        _git(clone, "commit", "-qam", "widen verdict")
        assert any(
            "certification_verdict" in str(e) for e in _preflight(clone)["gate_a_errors"] if str(e).startswith("A-19")
        )
        _reset(clone)

        record = json.loads((clone / CERT_RECORD).read_text(encoding="utf-8"))
        record["frz_evol_dispositions"]["FRZ-EVOL-07"] += "; production rollback enforcement IS claimed"
        (clone / CERT_RECORD).write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
        _git(clone, "commit", "-qam", "widen FRZ-EVOL-07")
        assert any("FRZ-EVOL-07" in str(e) for e in _preflight(clone)["gate_a_errors"] if str(e).startswith("A-19"))
    finally:
        _reset(clone)


CERT_DIR = f"{QUAL}/iiac_evidence/EVOL-X/certification"
MANIFEST = f"{CERT_DIR}/iiac_audit_session_manifest.EVOL-X.json"
CERT_INDEX = f"{CERT_DIR}/CERTIFICATION_EVIDENCE_INDEX.json"
ASSESSMENT = f"{CERT_DIR}/EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md"


def _write_json(clone: Path, rel: str, payload: dict[str, Any]) -> None:
    (clone / rel).write_bytes((json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def _reindex(clone: Path) -> None:
    """Regenerate the certification evidence index exactly as an attacker who edits evidence would."""
    index = json.loads((clone / CERT_INDEX).read_text(encoding="utf-8"))
    for name, entry in index["files"].items():
        data = (clone / CERT_DIR / name).read_bytes()
        entry["sha256"], entry["bytes"] = hashlib.sha256(data).hexdigest(), len(data)
    _write_json(clone, CERT_INDEX, index)


def _a19(result: dict[str, Any]) -> list[str]:
    return [str(e) for e in result["gate_a_errors"] if str(e).startswith("A-19")]


def test_da01_real_repoint_of_the_p2_sha_to_head_is_rejected_in_a_fresh_clone(clones: dict[bool | str, Path]) -> None:
    """The audit's false PASS: re-point certification_publication_sha at a later commit (both documents)."""
    clone = clones[True]
    try:
        _write_json(clone, MANIFEST, {**json.loads((clone / MANIFEST).read_text(encoding="utf-8"))})
        _git(clone, "commit", "-qam", "touch the manifest", "--allow-empty")
        later = _git(clone, "rev-parse", "HEAD").stdout.strip()
        for rel in (MANIFEST, CERT_RECORD):
            document = json.loads((clone / rel).read_text(encoding="utf-8"))
            document["certification_publication_sha"] = later
            _write_json(clone, rel, document)
        _reindex(clone)
        _git(clone, "commit", "-qam", "re-point the P2 baseline")
        result = _preflight(clone)
        assert any("differs from the pinned P2" in e for e in _a19(result)), result["gate_a_errors"]
        assert result["gate_a_pass"] is False and result["gate_b_pass"] is False
    finally:
        _reset(clone)


def test_da01_real_repoint_is_rejected_when_the_preflight_runs_against_another_repository(
    clones: dict[bool | str, Path],
) -> None:
    """``--repo <clone>`` (fixture class) must use the pinned anchor, never the declared field."""
    clone = clones[True]
    try:
        later = _git(clone, "rev-parse", "HEAD").stdout.strip()
        for rel in (MANIFEST, CERT_RECORD):
            document = json.loads((clone / rel).read_text(encoding="utf-8"))
            document["certification_publication_sha"] = later
            _write_json(clone, rel, document)
        _reindex(clone)
        _git(clone, "commit", "-qam", "re-point the P2 baseline")
        proc = subprocess.run(
            [
                sys.executable,
                PREFLIGHT_SCRIPT,
                "--gate",
                "b6-readiness",
                "--json",
                "--repo",
                str(clone),
                "--evidence-root",
                str(clone),
            ],
            cwd=REPO,
            env=_env(REPO),
            capture_output=True,
            text=True,
            check=False,
            errors="replace",
        )
        result = json.loads(proc.stdout)
        assert result["repository_class"] == "fixture"
        assert any("differs from the pinned P2" in e for e in _a19(result)), result["gate_a_errors"]
        assert result["gate_a_pass"] is False and proc.returncode != 0
    finally:
        _reset(clone)


@pytest.mark.parametrize(
    "mutation",
    ["notes-evidence-only-removed", "stage-id", "assessment-edited-and-reindexed", "assessment-deleted"],
)
def test_da02_real_sealed_record_keys_and_p2_evidence_files_cannot_be_changed(
    clones: dict[bool | str, Path], mutation: str
) -> None:
    clone = clones[True]
    try:
        if mutation == "notes-evidence-only-removed":
            record = json.loads((clone / CERT_RECORD).read_text(encoding="utf-8"))
            record["notes"] = "Certification publication."
            _write_json(clone, CERT_RECORD, record)
            expected = "'notes'"
        elif mutation == "stage-id":
            record = json.loads((clone / CERT_RECORD).read_text(encoding="utf-8"))
            record["stage_id"] = "EVOL-Y"
            _write_json(clone, CERT_RECORD, record)
            expected = "'stage_id'"
        elif mutation == "assessment-edited-and-reindexed":
            with (clone / ASSESSMENT).open("ab") as handle:
                handle.write(b"\nF-C-01 and F-C-02 are withdrawn; the certification is not evidence-only.\n")
            expected = "certification evidence EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md differs"
        else:
            (clone / ASSESSMENT).unlink()
            expected = "EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md"
        if mutation.startswith("assessment"):
            if mutation == "assessment-deleted":
                index = json.loads((clone / CERT_INDEX).read_text(encoding="utf-8"))
                index["files"].pop("EVOL_X_IIAC_CERTIFICATION_ASSESSMENT.md")
                _write_json(clone, CERT_INDEX, index)
            else:
                _reindex(clone)
        else:
            _reindex(clone)
        _git(clone, "add", "-A")
        _git(clone, "commit", "-qm", mutation)
        result = _preflight(clone)
        assert any(expected in e for e in _a19(result)), result["gate_a_errors"]
        assert result["gate_a_pass"] is False and result["gate_b_pass"] is False
    finally:
        _reset(clone)
