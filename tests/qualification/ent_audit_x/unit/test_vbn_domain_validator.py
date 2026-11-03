from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.vbn import validate_domain_report


def test_vbn_validator_31_ids() -> None:
    root = Path(__file__).resolve().parents[4]
    script = root / "scripts/enterprise_audit/vbn_domain_report.py"
    out = root / ".tmp/session/ent-audit-x-vbn-test/vbn_domains.v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(out)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = __import__("json").loads(out.read_text(encoding="utf-8"))
    errors = validate_domain_report(report)
    assert errors == ()


def test_vbn_forged_sha_rejected() -> None:
    report = {
        "schema_version": 1,
        "domains_total": 31,
        "qualification_sha": "b" * 40,
        "domains": [{"domain_id": f"D{i:02d}", "status": "QUALIFIED"} for i in range(1, 32)],
    }
    errors = validate_domain_report(report, expected_sha="a" * 40)
    assert any("stale" in e for e in errors)
