# © Artur Czarnecki. All rights reserved.

"""CLI verification hardening (FH-1…FH-14) — offline, no network."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from governed_contractor_application.host.offline_demo import (
    run_offline_governed_contractor_demo,
)
from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.execution_evidence.key_store import (
    DEMO_OFFLINE_KEY_ID,
    FilesystemHostKeyResolver,
    write_verification_key_artifact,
)
from intergrax.runtime.execution_evidence.verify import verify_proof_receipt

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]


def _uv_intergrax(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    # Force a narrow stdout encoding to catch UnicodeEncodeError in help paths.
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        ["uv", "run", "intergrax", *args],
        cwd=str(cwd or _REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


def test_filesystem_key_resolver_roundtrip(tmp_path: Path) -> None:
    attestor = build_deterministic_test_attestor(key_id=DEMO_OFFLINE_KEY_ID)
    path = write_verification_key_artifact(
        tmp_path,
        key_id=attestor.key_id,
        public_key_bytes=attestor.public_key_bytes,
    )
    text = path.read_text(encoding="utf-8")
    assert "private" not in text.lower() or "private_key" not in text
    assert "seed" not in text.lower()
    resolver = FilesystemHostKeyResolver(tmp_path)
    assert resolver.resolve_public_key(attestor.key_id) == attestor.public_key_bytes
    assert resolver.resolve_public_key("unknown") is None


def test_filesystem_key_resolver_malformed_and_deprecated(tmp_path: Path) -> None:
    bad = tmp_path / "keys" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed_verification_key"):
        FilesystemHostKeyResolver(tmp_path).resolve_public_key("bad")

    attestor = build_deterministic_test_attestor(key_id="dep-1")
    write_verification_key_artifact(
        tmp_path,
        key_id="dep-1",
        public_key_bytes=attestor.public_key_bytes,
        deprecated=True,
        status="active",
    )
    assert (
        FilesystemHostKeyResolver(tmp_path, allow_deprecated=True).resolve_public_key(
            "dep-1"
        )
        == attestor.public_key_bytes
    )
    assert (
        FilesystemHostKeyResolver(tmp_path, allow_deprecated=False).resolve_public_key(
            "dep-1"
        )
        is None
    )

    write_verification_key_artifact(
        tmp_path,
        key_id="algo",
        public_key_bytes=attestor.public_key_bytes,
    )
    algo_path = tmp_path / "keys" / "algo.json"
    payload = json.loads(algo_path.read_text(encoding="utf-8"))
    payload["algorithm"] = "RSA"
    algo_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="algorithm_not_allowed"):
        FilesystemHostKeyResolver(tmp_path).resolve_public_key("algo")


def test_cli_help_ascii_safe() -> None:
    commands = [
        ["--help"],
        ["demo", "--help"],
        ["demo", "governed-contractor", "--help"],
        ["receipt", "--help"],
        ["receipt", "verify", "--help"],
        ["external-work", "--help"],
        ["external-work", "retry-attestation", "--help"],
    ]
    banned = ("\u2192", "\u2713", "\u2717", "\u2022", "\u2014", "\u2013")
    for cmd in commands:
        proc = _uv_intergrax(*cmd)
        assert proc.returncode == 0, (cmd, proc.stderr)
        combined = (proc.stdout or "") + (proc.stderr or "")
        for ch in banned:
            assert ch not in combined, f"{cmd} contains {ch!r}"


def test_verify_requires_explicit_key_source(tmp_path: Path) -> None:
    report = run_offline_governed_contractor_demo(store_root=tmp_path / "demo")
    receipt = Path(report.receipt_absolute_path)
    proc = _uv_intergrax("receipt", "verify", str(receipt))
    assert proc.returncode != 0
    assert "verification_key_source_required" in proc.stdout


def test_verify_store_backed_subprocess(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    assert report.verification_valid is True
    proc = _uv_intergrax(
        "receipt",
        "verify",
        str(Path(report.receipt_absolute_path)),
        "--store",
        str(store),
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["valid"] is True
    assert payload["key_source"] == "store"
    assert payload["signature_valid"] is True


def test_verify_public_key_hex_and_conflict(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    attestor = build_deterministic_test_attestor(key_id=report.key_id)
    receipt = str(Path(report.receipt_absolute_path))
    ok = _uv_intergrax(
        "receipt",
        "verify",
        receipt,
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        report.key_id,
    )
    assert ok.returncode == 0
    assert json.loads(ok.stdout)["key_source"] == "public_key_hex"

    conflict = _uv_intergrax(
        "receipt",
        "verify",
        receipt,
        "--store",
        str(store),
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        report.key_id,
    )
    assert conflict.returncode != 0
    assert "exactly_one_verification_key_source_required" in conflict.stdout


def test_verify_public_key_file(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    attestor = build_deterministic_test_attestor(key_id=report.key_id)
    key_file = tmp_path / "pub.hex"
    key_file.write_text(attestor.public_key_bytes.hex(), encoding="utf-8")
    proc = _uv_intergrax(
        "receipt",
        "verify",
        str(Path(report.receipt_absolute_path)),
        "--public-key-file",
        str(key_file),
        "--key-id",
        report.key_id,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["key_source"] == "public_key_file"


def test_verify_demo_key_and_mismatch(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    ok = _uv_intergrax(
        "receipt",
        "verify",
        str(Path(report.receipt_absolute_path)),
        "--demo-key",
    )
    assert ok.returncode == 0
    assert json.loads(ok.stdout)["key_source"] == "demo_key"

    receipt = ProofReceipt.model_validate_json(
        Path(report.receipt_absolute_path).read_text(encoding="utf-8")
    )
    mutated = receipt.model_copy(
        update={
            "host_attestation": receipt.host_attestation.model_copy(
                update={"key_id": "other-key"}
            )
        }
    )
    bad_path = tmp_path / "bad_key.json"
    bad_path.write_text(mutated.model_dump_json(), encoding="utf-8")
    bad = _uv_intergrax("receipt", "verify", str(bad_path), "--demo-key")
    assert bad.returncode != 0
    assert "demo_key_id_mismatch" in bad.stdout


def test_verify_wrong_key_and_mutated_receipt(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    receipt_path = Path(report.receipt_absolute_path)
    other = build_deterministic_test_attestor(
        key_id=report.key_id,
        seed=b"\x11" * 32,
    )
    wrong = _uv_intergrax(
        "receipt",
        "verify",
        str(receipt_path),
        "--public-key-hex",
        other.public_key_bytes.hex(),
        "--key-id",
        report.key_id,
    )
    assert wrong.returncode != 0
    assert json.loads(wrong.stdout)["valid"] is False

    receipt = ProofReceipt.model_validate_json(receipt_path.read_text(encoding="utf-8"))
    event = receipt.execution_boundary_event.model_copy(update={"actor": "mutated"})
    mutated = receipt.model_copy(update={"execution_boundary_event": event})
    mut_path = store / "export" / "accept_receipt_mutated.json"
    mut_path.write_text(mutated.model_dump_json(indent=2), encoding="utf-8")
    mut = _uv_intergrax("receipt", "verify", str(mut_path), "--store", str(store))
    assert mut.returncode != 0
    payload = json.loads(mut.stdout)
    assert payload["valid"] is False
    assert "digest_mismatch" in payload["errors"]


def test_verify_key_id_mismatch_flag(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    attestor = build_deterministic_test_attestor(key_id=report.key_id)
    proc = _uv_intergrax(
        "receipt",
        "verify",
        str(Path(report.receipt_absolute_path)),
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        "wrong-id",
    )
    assert proc.returncode != 0
    assert "key_id_mismatch" in proc.stdout


def test_verify_missing_store_and_missing_key(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    receipt = str(Path(report.receipt_absolute_path))
    missing_store = _uv_intergrax(
        "receipt", "verify", receipt, "--store", str(tmp_path / "nope")
    )
    assert missing_store.returncode != 0
    assert "store_not_found" in missing_store.stdout

    empty = tmp_path / "empty_store"
    empty.mkdir()
    no_key = _uv_intergrax("receipt", "verify", receipt, "--store", str(empty))
    assert no_key.returncode != 0
    payload = json.loads(no_key.stdout)
    assert payload["valid"] is False
    assert "unknown_key_id" in payload["errors"]


def test_signer_failure_recovery_cli_subprocess(tmp_path: Path) -> None:
    store = tmp_path / "recovery"
    fail = _uv_intergrax(
        "demo",
        "governed-contractor",
        "--offline",
        "--simulate-signing-failure",
        "--store",
        str(store),
    )
    assert fail.returncode == 0, fail.stderr
    fail_payload = json.loads(fail.stdout)
    assert fail_payload["state"] == "EXECUTION_SUCCEEDED_ATTESTATION_FAILED"
    assert fail_payload["provider_execution_succeeded"] is True
    assert fail_payload["attestation_succeeded"] is False
    assert fail_payload.get("receipt_path") in (None, "")
    assert (store / "provider_calls.json").is_file()
    ledger = json.loads((store / "provider_calls.json").read_text(encoding="utf-8"))
    assert ledger["create_calls"] == 1
    assert ledger["accept_calls"] == 1

    retry = _uv_intergrax(
        "external-work",
        "retry-attestation",
        "exec-offline-accept",
        "--store",
        str(store),
    )
    assert retry.returncode == 0, retry.stderr
    retry_payload = json.loads(retry.stdout)
    assert retry_payload["state"] == "EXECUTION_SUCCEEDED_ATTESTED"
    assert retry_payload["provider_invoked"] is False
    assert retry_payload["verification_valid"] is True
    export = store / "export" / "accept_receipt.json"
    assert export.is_file()

    verify = _uv_intergrax(
        "receipt", "verify", str(export), "--store", str(store)
    )
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["valid"] is True

    again = _uv_intergrax(
        "external-work",
        "retry-attestation",
        "exec-offline-accept",
        "--store",
        str(store),
    )
    assert again.returncode == 0
    again_payload = json.loads(again.stdout)
    assert again_payload["reason"] == "attested_idempotent"
    assert again_payload["provider_invoked"] is False

    ledger2 = json.loads((store / "provider_calls.json").read_text(encoding="utf-8"))
    assert ledger2 == ledger

    # Security: no private material in store keys / receipts.
    for path in store.rglob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "private_key" not in text
        assert '"seed"' not in text


def test_retry_invalid_execution_and_without_signer(tmp_path: Path) -> None:
    store = tmp_path / "empty"
    store.mkdir()
    missing = _uv_intergrax(
        "external-work",
        "retry-attestation",
        "no-such-exec",
        "--store",
        str(store),
    )
    # No demo_mode marker -> signing source required
    assert missing.returncode != 0
    assert "signing_key_source_required" in missing.stdout

    run_offline_governed_contractor_demo(
        store_root=store,
        simulate_signing_failure=True,
    )
    bad_id = _uv_intergrax(
        "external-work",
        "retry-attestation",
        "missing-exec",
        "--store",
        str(store),
    )
    assert bad_id.returncode != 0
    assert "execution_result_missing" in bad_id.stdout


def test_offline_demo_exports_verification_key(tmp_path: Path) -> None:
    store = tmp_path / "demo"
    report = run_offline_governed_contractor_demo(store_root=store)
    key_path = store / "keys" / f"{DEMO_OFFLINE_KEY_ID}.json"
    assert key_path.is_file()
    assert "verification_key_path" in report.as_dict()
    assert "verification_command" in report.as_dict()
    assert "--store" in report.verification_command
    resolver = FilesystemHostKeyResolver(store)
    receipt = ProofReceipt.model_validate_json(
        Path(report.receipt_absolute_path).read_text(encoding="utf-8")
    )
    assert verify_proof_receipt(
        receipt, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid


def test_cli_demo_subprocess_then_verify(tmp_path: Path) -> None:
    store = tmp_path / "cli_demo"
    demo = _uv_intergrax(
        "demo",
        "governed-contractor",
        "--offline",
        "--store",
        str(store),
    )
    assert demo.returncode == 0, demo.stderr
    payload = json.loads(demo.stdout)
    assert payload["verification_valid"] is True
    assert "verification_key_path" in payload
    # Fresh process verifies via store.
    verify = _uv_intergrax(
        "receipt",
        "verify",
        str(store / "export" / "accept_receipt.json"),
        "--store",
        str(store),
    )
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["valid"] is True
