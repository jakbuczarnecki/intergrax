# © Artur Czarnecki. All rights reserved.

"""CLI verification hardening (FH-1…FH-14) — offline, no network.

Most coverage runs in-process via ``intergrax.cli.main`` (fast).
A small subprocess smoke set proves fresh-process portability without
paying ``uv run`` startup on every case.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import pytest

from governed_contractor_application.host.offline_demo import (
    run_offline_governed_contractor_demo,
)
from intergrax.cli.main import main as cli_main
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
_BANNED_HELP_CHARS = ("\u2192", "\u2713", "\u2717", "\u2022", "\u2014", "\u2013")


@dataclass(frozen=True, slots=True)
class _CliResult:
    returncode: int
    stdout: str
    stderr: str


def _cli(*args: str) -> _CliResult:
    """Fast in-process CLI invocation (same process as pytest)."""
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli_main(list(args))
        except SystemExit as exc:
            code = int(exc.code or 0)
    return _CliResult(returncode=int(code), stdout=out.getvalue(), stderr=err.getvalue())


def _cli_subprocess(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """True fresh process — uses venv python, not ``uv run``."""
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env["PYTHONIOENCODING"] = "cp1252"
    return subprocess.run(
        [sys.executable, "-m", "intergrax.cli", *args],
        cwd=str(cwd or _REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


@pytest.fixture(scope="module")
def demo_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    """One offline demo for the module — avoid re-signing for every negative case."""
    store = tmp_path_factory.mktemp("demo")
    report = run_offline_governed_contractor_demo(store_root=store)
    assert report.verification_valid is True
    return store, report


def test_filesystem_key_resolver_roundtrip(tmp_path: Path) -> None:
    attestor = build_deterministic_test_attestor(key_id=DEMO_OFFLINE_KEY_ID)
    path = write_verification_key_artifact(
        tmp_path,
        key_id=attestor.key_id,
        public_key_bytes=attestor.public_key_bytes,
    )
    text = path.read_text(encoding="utf-8")
    assert "private_key" not in text
    assert '"seed"' not in text.lower()
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


def test_cli_help_ascii_safe_inprocess() -> None:
    commands = [
        ["--help"],
        ["demo", "--help"],
        ["demo", "governed-contractor", "--help"],
        ["receipt", "--help"],
        ["receipt", "verify", "--help"],
        ["external-work", "--help"],
        ["external-work", "retry-attestation", "--help"],
    ]
    for cmd in commands:
        result = _cli(*cmd)
        assert result.returncode == 0, (cmd, result.stderr)
        combined = result.stdout + result.stderr
        for ch in _BANNED_HELP_CHARS:
            assert ch not in combined, f"{cmd} contains {ch!r}"


def test_cli_help_cp1252_subprocess_smoke() -> None:
    """One real process under narrow encoding — Windows help contract."""
    proc = _cli_subprocess("demo", "governed-contractor", "--help")
    assert proc.returncode == 0, proc.stderr
    combined = (proc.stdout or "") + (proc.stderr or "")
    for ch in _BANNED_HELP_CHARS:
        assert ch not in combined


def test_verify_key_sources_and_negatives(demo_store: tuple[Path, object]) -> None:
    store, report = demo_store
    receipt = str(Path(report.receipt_absolute_path))  # type: ignore[attr-defined]
    attestor = build_deterministic_test_attestor(key_id=report.key_id)  # type: ignore[attr-defined]

    missing = _cli("receipt", "verify", receipt)
    assert missing.returncode != 0
    assert "verification_key_source_required" in missing.stdout

    store_ok = _cli("receipt", "verify", receipt, "--store", str(store))
    assert store_ok.returncode == 0
    assert json.loads(store_ok.stdout)["key_source"] == "store"

    hex_ok = _cli(
        "receipt",
        "verify",
        receipt,
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        report.key_id,  # type: ignore[attr-defined]
    )
    assert hex_ok.returncode == 0
    assert json.loads(hex_ok.stdout)["key_source"] == "public_key_hex"

    key_file = store / "pub.hex"
    key_file.write_text(attestor.public_key_bytes.hex(), encoding="utf-8")
    file_ok = _cli(
        "receipt",
        "verify",
        receipt,
        "--public-key-file",
        str(key_file),
        "--key-id",
        report.key_id,  # type: ignore[attr-defined]
    )
    assert file_ok.returncode == 0
    assert json.loads(file_ok.stdout)["key_source"] == "public_key_file"

    demo_ok = _cli("receipt", "verify", receipt, "--demo-key")
    assert demo_ok.returncode == 0
    assert json.loads(demo_ok.stdout)["key_source"] == "demo_key"

    conflict = _cli(
        "receipt",
        "verify",
        receipt,
        "--store",
        str(store),
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        report.key_id,  # type: ignore[attr-defined]
    )
    assert conflict.returncode != 0
    assert "exactly_one_verification_key_source_required" in conflict.stdout

    mismatch = _cli(
        "receipt",
        "verify",
        receipt,
        "--public-key-hex",
        attestor.public_key_bytes.hex(),
        "--key-id",
        "wrong-id",
    )
    assert mismatch.returncode != 0
    assert "key_id_mismatch" in mismatch.stdout

    missing_store = _cli("receipt", "verify", receipt, "--store", str(store / "nope"))
    assert missing_store.returncode != 0
    assert "store_not_found" in missing_store.stdout

    empty = store / "empty_keys"
    empty.mkdir()
    no_key = _cli("receipt", "verify", receipt, "--store", str(empty))
    assert no_key.returncode != 0
    assert "unknown_key_id" in json.loads(no_key.stdout)["errors"]

    other = build_deterministic_test_attestor(
        key_id=report.key_id,  # type: ignore[attr-defined]
        seed=b"\x11" * 32,
    )
    wrong = _cli(
        "receipt",
        "verify",
        receipt,
        "--public-key-hex",
        other.public_key_bytes.hex(),
        "--key-id",
        report.key_id,  # type: ignore[attr-defined]
    )
    assert wrong.returncode != 0
    assert json.loads(wrong.stdout)["valid"] is False

    receipt_obj = ProofReceipt.model_validate_json(Path(receipt).read_text(encoding="utf-8"))
    event = receipt_obj.execution_boundary_event.model_copy(update={"actor": "mutated"})
    mutated = receipt_obj.model_copy(update={"execution_boundary_event": event})
    mut_path = store / "export" / "accept_receipt_mutated.json"
    mut_path.write_text(mutated.model_dump_json(indent=2), encoding="utf-8")
    mut = _cli("receipt", "verify", str(mut_path), "--store", str(store))
    assert mut.returncode != 0
    mut_payload = json.loads(mut.stdout)
    assert mut_payload["valid"] is False
    assert "digest_mismatch" in mut_payload["errors"]

    bad_key_receipt = receipt_obj.model_copy(
        update={
            "host_attestation": receipt_obj.host_attestation.model_copy(
                update={"key_id": "other-key"}
            )
        }
    )
    bad_path = store / "bad_key.json"
    bad_path.write_text(bad_key_receipt.model_dump_json(), encoding="utf-8")
    demo_mismatch = _cli("receipt", "verify", str(bad_path), "--demo-key")
    assert demo_mismatch.returncode != 0
    assert "demo_key_id_mismatch" in demo_mismatch.stdout


def test_offline_demo_exports_verification_key(demo_store: tuple[Path, object]) -> None:
    store, report = demo_store
    key_path = store / "keys" / f"{DEMO_OFFLINE_KEY_ID}.json"
    assert key_path.is_file()
    assert "--store" in report.verification_command  # type: ignore[attr-defined]
    resolver = FilesystemHostKeyResolver(store)
    receipt = ProofReceipt.model_validate_json(
        Path(report.receipt_absolute_path).read_text(encoding="utf-8")  # type: ignore[attr-defined]
    )
    assert verify_proof_receipt(
        receipt, key_resolver=resolver, require_policy_bundle_artifact=True
    ).valid


def test_retry_invalid_execution_and_without_signer(tmp_path: Path) -> None:
    store = tmp_path / "empty"
    store.mkdir()
    missing = _cli(
        "external-work",
        "retry-attestation",
        "no-such-exec",
        "--store",
        str(store),
    )
    assert missing.returncode != 0
    assert "signing_key_source_required" in missing.stdout

    run_offline_governed_contractor_demo(
        store_root=store,
        simulate_signing_failure=True,
    )
    bad_id = _cli(
        "external-work",
        "retry-attestation",
        "missing-exec",
        "--store",
        str(store),
    )
    assert bad_id.returncode != 0
    assert "execution_result_missing" in bad_id.stdout


def test_signer_failure_recovery_fresh_process(tmp_path: Path) -> None:
    """Fresh-process proof: demo fail -> retry -> verify -> idempotent retry."""
    store = tmp_path / "recovery"
    fail = _cli_subprocess(
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
    ledger = json.loads((store / "provider_calls.json").read_text(encoding="utf-8"))
    assert ledger["create_calls"] == 1
    assert ledger["accept_calls"] == 1

    retry = _cli_subprocess(
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

    verify = _cli_subprocess(
        "receipt", "verify", str(export), "--store", str(store)
    )
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["valid"] is True

    again = _cli_subprocess(
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
    assert json.loads((store / "provider_calls.json").read_text(encoding="utf-8")) == ledger

    for path in store.rglob("*.json"):
        text = path.read_text(encoding="utf-8").lower()
        assert "private_key" not in text
        assert '"seed"' not in text


def test_demo_then_verify_fresh_process(tmp_path: Path) -> None:
    """Fresh-process proof: demo process exits, verifier process validates via --store."""
    store = tmp_path / "cli_demo"
    demo = _cli_subprocess(
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

    verify = _cli_subprocess(
        "receipt",
        "verify",
        str(store / "export" / "accept_receipt.json"),
        "--store",
        str(store),
    )
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["valid"] is True
