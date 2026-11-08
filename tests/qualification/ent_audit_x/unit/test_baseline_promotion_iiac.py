"""D-2: OD-ENT-01 promotion accepts a record signed by an operator-designated signing session (IIAC Amendment 01).

Every test builds a small real git repository (ladder, dossier, IIAC manifest, delegation, record), so ancestry and
published-blob reads are exercised without mocking git. Negative cases must all fail closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from intergrax.platform.enterprise_audit.baseline import (
    EnterpriseAuditBaseline,
    baseline_path,
    candidate_baseline_path,
    save_candidate_baseline,
)
from scripts.enterprise_audit.iiac_baseline_promotion import (
    main as promotion_main,
)
from scripts.enterprise_audit.iiac_baseline_promotion import (
    promote_accepted_baseline,
    signed_text_digest,
    verify_record,
)

POLICY = "IIAC-2026-11-05.1"
QUAL = Path("docs/project/maintainers/qualification")
ENT = QUAL / "ent_audit_x"
DECISIONS = QUAL / "iiac_evidence" / "ENT-AUDIT-X" / "operator_decisions"
CERT = QUAL / "iiac_evidence" / "ENT-AUDIT-X" / "certification"
OPERATOR_RECORDS = ENT / "operator_records"
ACCEPTED = "a" * 40
SIGNED_AT = "2026-11-08T12:00:00+01:00"
OPERATOR_SIGNED = "2026-11-08T11:00:00+01:00"


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _write(root: Path, rel: Path | str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass
class World:
    root: Path
    record_path: Path
    tes: str
    evid: str
    b6: str
    delegation_sha: str
    manifest_rel: str


def build_world(root: Path, **opt: Any) -> World:  # noqa: C901 - one fixture, many switches
    origin = opt.get("record_origin", "operator-delegate")
    delegated = origin == "operator-delegate"
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "user.email", "t@example.invalid")

    _write(root, "seed.txt", "tes")
    tes = _commit(root, "tes")
    _write(root, "seed.txt", "evid")
    evid = _commit(root, "evid")
    _write(root, "seed.txt", "b6")
    b6 = _commit(root, "b6")

    _write(
        root,
        QUAL / "IIAC_POLICY_ACTIVATION_RECORD.json",
        json.dumps({"policy_version": POLICY, "status": opt.get("policy_status", "IN_FORCE")}),
    )
    b6_cell = f"`{b6}`" if opt.get("ladder_b6", True) else "*unset*"
    _write(
        root,
        ENT / "ENT_AUDIT_X_SHA_LADDER.md",
        "\n".join(
            [
                "| Role | SHA | Note |",
                "| --- | --- | --- |",
                f"| TEST_EXECUTION_SHA | `{tes}` | tes |",
                f"| EVIDENCE_PUBLICATION_SHA | `{evid}` | evid |",
                f"| B6_ACCEPTANCE_PUBLICATION_SHA | {b6_cell} | b6 |",
                "| ACCEPTED_BASELINE_SHA | *unset* | baseline |",
                "",
            ]
        ),
    )
    _write(root, ENT / "ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md", f"{tes}\n{evid}\n{b6}\n")

    manifest = {
        "stage_id": "ENT-AUDIT-X",
        "verdict": "PASS_SCOPED",
        "test_execution_sha": tes,
        "evidence_publication_sha": evid,
        "implementation_session_id": "IMPL-1",
        "auditor_session_id": "AUD-1",
        "certifier_session_id": "CERT-1",
    }
    manifest.update(opt.get("manifest", {}))
    manifest_rel = (CERT / "iiac_audit_session_manifest.ENT-AUDIT-X.json").as_posix()
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    _write(root, manifest_rel, manifest_bytes.decode("utf-8"))
    manifest_digest = opt.get("manifest_digest") or hashlib.sha256(manifest_bytes).hexdigest()
    _commit(root, "publish ladder, dossier, manifest")

    record_dir = opt.get("record_dir") or (DECISIONS if delegated else OPERATOR_RECORDS)
    record_rel = (record_dir / "OD_ENT_01.json").as_posix()
    record: dict[str, Any] = {
        "decision_id": "OD-ENT-01",
        "record_origin": origin,
        "b6_acceptance_sha": b6,
        "evidence_publication_sha": evid,
        "test_execution_sha": tes,
        "accepted_baseline_sha": ACCEPTED,
    }
    if not delegated:
        record["operator_origin"] = True
    else:
        record.update(
            {
                "policy_version": POLICY,
                "designated_session_id": "SIGN-1",
                "iiac_manifest_path": manifest_rel,
                "iiac_manifest_digest": manifest_digest,
            }
        )
    record.update(opt.get("record", {}))

    delegation_sha = "0" * 40
    if delegated:
        digest = opt.get("digest") or signed_text_digest(record)
        fields = {
            "id": "DEL-ENT-01",
            "policy": POLICY,
            "decision": "OD-ENT-01",
            "record_path": record_rel,
            "designated": "SIGN-1",
            "implementers": "IMPL-1, IMPL-2",
            "expires": "2026-11-30",
        }
        fields.update(opt.get("delegation", {}))
        sig = ""
        if opt.get("operator_line", True):
            sig = f"Approved delegation {fields['id']}: operator @ {opt.get('operator_ts', OPERATOR_SIGNED)}"
        deleg_rel = (DECISIONS / "DELEGATION_ENT_01.md").as_posix()
        _write(
            root,
            deleg_rel,
            "\n".join(
                [
                    "# IIAC operator signing delegation record",
                    f"**Delegation ID:** `{fields['id']}`",
                    f"**Policy version:** `{fields['policy']}`",
                    "**Date (ISO-8601):** `2026-11-08`",
                    "",
                    "## Delegated act",
                    "- Stage: `ENT-AUDIT-X`",
                    f"- Decision ID (decision to be signed): `{fields['decision']}`",
                    f"- Record path (repo-relative path of the record to be signed): `{fields['record_path']}`",
                    f"- SHA-256 of the text to be signed: `{digest}`",
                    "",
                    "## Designated signing session",
                    f"- Designated session id: `{fields['designated']}`",
                    "- Implementer session id(s) of this stage (must differ from the designated session id): "
                    f"`{fields['implementers']}`",
                    "",
                    "## Limits",
                    f"- Expires (ISO-8601): `{fields['expires']}`",
                    "",
                    "## Operator signature line",
                    sig,
                    "",
                ]
            ),
        )
        delegation_sha = _commit(root, "delegation")
        record["signature_line"] = (
            opt.get("signature_line") or f"Approved OD-ENT-01: operator-delegate @ {opt.get('sig_ts', SIGNED_AT)}"
        )
        record["delegation_line"] = (
            f"Delegation: {opt.get('delegation_path', deleg_rel)}@{opt.get('delegation_sha', delegation_sha)}"
        )
        if opt.get("tamper_after"):
            record["accepted_baseline_sha"] = "b" * 40

    _write(root, record_rel, json.dumps(record, indent=2))
    _commit(root, "record")
    return World(root, root / record_rel, tes, evid, b6, delegation_sha, manifest_rel)


def _verify(world: World) -> dict:
    return verify_record(world.record_path, for_promotion=True, repo_root=world.root)


def test_valid_delegate_record_verifies(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    record = _verify(world)
    assert record["record_origin"] == "operator-delegate"
    assert record["accepted_baseline_sha"] == ACCEPTED


def test_signature_on_the_expiry_day_is_still_valid(tmp_path: Path) -> None:
    """A date-only Expires means the end of that day (UTC); 2026-11-30T23:00Z is inside the window."""
    world = build_world(tmp_path, sig_ts="2026-12-01T00:00:00+01:00")
    assert _verify(world)["record_origin"] == "operator-delegate"


def test_human_operator_record_still_verifies(tmp_path: Path) -> None:
    world = build_world(tmp_path, record_origin="operator")
    assert _verify(world)["operator_origin"] is True


def test_valid_delegate_record_promotes_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    world = build_world(repo)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evidence_manifest.v1.json").write_text(json.dumps({"qualification_sha": world.tes}), encoding="utf-8")
    monkeypatch.setattr("intergrax.platform.enterprise_audit.validators.verify_closed_evidence_bundle", lambda _d: [])
    base = tmp_path / "baseline"
    save_candidate_baseline(
        candidate_baseline_path(base), EnterpriseAuditBaseline(accepted_baseline_sha=ACCEPTED, findings={})
    )
    out = promote_accepted_baseline(
        base, od_record_path=world.record_path, verify_closed_bundle_dir=bundle, repo_root=world.root
    )
    assert json.loads(out.read_text(encoding="utf-8"))["accepted_baseline_sha"] == ACCEPTED


def _cli_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **opt: Any) -> tuple[World, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    world = build_world(repo, **opt)
    baseline_dir = repo / "tests" / "qualification" / "ent_audit_x"
    save_candidate_baseline(
        candidate_baseline_path(baseline_dir), EnterpriseAuditBaseline(accepted_baseline_sha=ACCEPTED, findings={})
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evidence_manifest.v1.json").write_text(json.dumps({"qualification_sha": world.tes}), encoding="utf-8")
    monkeypatch.setattr("intergrax.platform.enterprise_audit.validators.verify_closed_evidence_bundle", lambda _d: [])
    return world, bundle, baseline_dir


def _cli(world: World, bundle: Path, *extra: str) -> int:
    return promotion_main(
        ["--od-record", str(world.record_path), "--evidence-dir", str(bundle), "--repo-root", str(world.root), *extra]
    )


def test_cli_promotes_a_delegated_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    world, bundle, baseline_dir = _cli_world(tmp_path, monkeypatch)
    assert _cli(world, bundle) == 0
    assert json.loads(baseline_path(baseline_dir).read_text(encoding="utf-8"))["accepted_baseline_sha"] == ACCEPTED


def test_cli_rejects_a_record_changed_after_delegation_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world, bundle, baseline_dir = _cli_world(tmp_path, monkeypatch, tamper_after=True)
    assert _cli(world, bundle) == 1
    assert "SHA-256" in capsys.readouterr().err
    assert not baseline_path(baseline_dir).exists()


def test_cli_rejects_a_baseline_dir_outside_the_jail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    world, bundle, _ = _cli_world(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    assert _cli(world, bundle, "--baseline-dir", str(elsewhere)) == 1
    assert "baseline_dir must stay under" in capsys.readouterr().err
    assert not baseline_path(elsewhere).exists()


REJECTED = {
    "agent_origin": ({"record_origin": "agent"}, "agent-origin"),
    "no_operator_signature_in_delegation": ({"operator_line": False}, "no operator signature line"),
    "template_blank_signature": ({"operator_ts": "________________"}, "no operator signature line"),
    "plain_operator_signature_in_record": (
        {"signature_line": "Approved OD-ENT-01: operator @ " + SIGNED_AT},
        "signature_line must be",
    ),
    "signature_for_other_decision": (
        {"signature_line": "Approved OD-ENT-08: operator-delegate @ " + SIGNED_AT},
        "signature_line must be",
    ),
    "delegation_for_other_decision": ({"delegation": {"decision": "OD-ENT-08"}}, "covers decision"),
    "policy_mismatch_in_delegation": ({"delegation": {"policy": "IIAC-0000"}}, "policy version"),
    "policy_not_in_force": ({"policy_status": "NOT_IN_FORCE"}, "not IN_FORCE"),
    "digest_mismatch": ({"digest": "0" * 64}, "SHA-256"),
    "record_changed_after_delegation": ({"tamper_after": True}, "SHA-256"),
    "expired_delegation": ({"sig_ts": "2026-12-02T00:00:00+01:00"}, "validity window"),
    "signed_before_operator": ({"sig_ts": "2026-11-08T10:00:00+01:00"}, "validity window"),
    "designated_is_implementer": ({"delegation": {"implementers": "IMPL-1, SIGN-1"}}, "implementer"),
    "designated_mismatch": ({"delegation": {"designated": "SIGN-OTHER"}}, "designated_session_id"),
    "blank_delegation_field": ({"delegation": {"expires": "________________"}}, "blank or missing"),
    "delegation_commit_unknown": ({"delegation_sha": "1" * 40}, "ancestor"),
    "delegation_outside_decisions_dir": ({"delegation_path": "elsewhere/DELEGATION.md"}, "must live under"),
    "delegation_binds_other_record_path": ({"delegation": {"record_path": "x/other.json"}}, "binds record path"),
    "record_outside_decisions_dir": ({"record_dir": QUAL / "elsewhere"}, "must live under"),
    "ladder_b6_unset": ({"ladder_b6": False}, "B6_ACCEPTANCE_PUBLICATION_SHA unset"),
    "manifest_verdict_fail": ({"manifest": {"verdict": "FAIL"}}, "verdict"),
    "manifest_wrong_stage": ({"manifest": {"stage_id": "EVOL-X"}}, "stage_id"),
    "manifest_tes_mismatch": ({"manifest": {"test_execution_sha": "c" * 40}}, "test_execution_sha"),
    "manifest_sessions_not_distinct": ({"manifest": {"auditor_session_id": "IMPL-1"}}, "pairwise distinct"),
    "manifest_digest_mismatch": ({"manifest_digest": "0" * 64}, "iiac_manifest_digest"),
    "designated_is_auditor": ({"manifest": {"auditor_session_id": "SIGN-1"}}, "auditor session"),
}


@pytest.mark.parametrize(("opt", "match"), REJECTED.values(), ids=REJECTED.keys())
def test_delegate_record_rejected(tmp_path: Path, opt: dict[str, Any], match: str) -> None:
    world = build_world(tmp_path, **opt)
    with pytest.raises(ValueError, match=match):
        _verify(world)


def test_delegate_origin_requires_valid_signature_lines_even_without_promotion(tmp_path: Path) -> None:
    world = build_world(tmp_path, signature_line="Approved OD-ENT-01: operator @ " + SIGNED_AT)
    with pytest.raises(ValueError, match="signature_line must be"):
        verify_record(world.record_path)


def test_signature_lines_are_not_part_of_the_signed_text() -> None:
    base = {"decision_id": "OD-ENT-01", "x": 1}
    signed = {**base, "signature_line": "a", "delegation_line": "b"}
    assert signed_text_digest(base) == signed_text_digest(signed)
    assert signed_text_digest(base) != signed_text_digest({**base, "x": 2})
