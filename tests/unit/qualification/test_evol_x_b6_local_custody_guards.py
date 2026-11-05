# © Artur Czarnecki. All rights reserved.

"""Guards for local B6 attestor/custodian separation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parents[3]
_ATTESTOR = _ROOT / "scripts/qualification/evol_x_b6_local_attestor.py"
_CUSTODIAN = _ROOT / "scripts/qualification/evol_x_b6_custodian_sign.py"
_R2_REPORT = _ROOT / ".tmp/session/evol-x-r2-final-reconciliation/run-2026-11-04/independent_verification_report.json"
_REPLAY = _ROOT / "scripts/qualification/evol_x_b6_replay_ledger.py"


def test_replay_ledger_rejects_repo_internal_path() -> None:
    from scripts.qualification.evol_x_b6_replay_ledger import ReplayLedgerError, append_binding

    with pytest.raises(ReplayLedgerError, match="must not live inside"):
        append_binding(_ROOT / ".tmp/session/ledger.json", "a" * 64, repo_root=_ROOT)


def test_attestor_refuses_without_readonly_env() -> None:
    env = os.environ.copy()
    env.pop("EVOL_B6_ATTESTOR_READONLY", None)
    env["EVOL_B6_ATTESTOR_IDENTITY"] = "external-attestor-1"
    proc = subprocess.run(
        [
            sys.executable,
            str(_ATTESTOR),
            "--session-root",
            str(_ROOT / ".tmp/session/pytest-attestor-guard"),
            "--verification-report",
            str(_R2_REPORT),
            "--trust-root-ref",
            "tr-test",
            "--key-id",
            "k-test",
            "--output",
            str(_ROOT / ".tmp/session/pytest-attestor-guard/unsigned.json"),
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1


@pytest.mark.skipif(not _R2_REPORT.is_file(), reason="R2 session report not present locally")
def test_attestor_builds_unsigned_from_r2_report() -> None:
    env = os.environ.copy()
    env["EVOL_B6_ATTESTOR_READONLY"] = "1"
    env["EVOL_B6_ATTESTOR_IDENTITY"] = "external-attestor-1"
    out = _ROOT / ".tmp/session/pytest-attestor-guard/unsigned-from-r2.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_ATTESTOR),
            "--verification-report",
            str(_R2_REPORT),
            "--trust-root-ref",
            "tr-test",
            "--key-id",
            "k-test",
            "--output",
            str(out),
            "--credential-class",
            "PRODUCTION_INDEPENDENT",
            "--formal-disposition",
            "ACCEPTED_AUTOMATED_B6",
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload.get("signature") in ("", None) or payload.get("signature") == ""
    assert payload.get("replay_binding") == ""


def test_custodian_refuses_without_mode_flag() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(_CUSTODIAN),
            "--unsigned",
            str(_ROOT / ".tmp/session/pytest-attestor-guard/unsigned-from-r2.json"),
            "--private-key",
            str(_ROOT / ".tmp/session/pytest-attestor-guard/fake.key"),
            "--output",
            str(_ROOT / ".tmp/session/pytest-attestor-guard/signed.json"),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
