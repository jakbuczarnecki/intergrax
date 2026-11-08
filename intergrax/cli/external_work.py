# © Artur Czarnecki. All rights reserved.

"""Tier-0 proof-receipt verification CLI (``intergrax receipt verify``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.contracts.execution_evidence.receipt import (
    parse_execution_evidence_proof_receipt_json,
)
from intergrax.runtime.execution_evidence.attestor import (
    ALGORITHM_ED25519,
    build_deterministic_test_attestor,
)
from intergrax.runtime.execution_evidence.key_store import (
    DEMO_OFFLINE_KEY_ID,
    FilesystemHostKeyResolver,
)
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_execution_evidence_proof_receipt,
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_parser(sub: argparse._SubParsersAction) -> None:
    rcpt = sub.add_parser("receipt", help="ProofReceipt tools.")
    rcpt_sub = rcpt.add_subparsers(dest="receipt_command", required=True)
    verify = rcpt_sub.add_parser(
        "verify",
        help="Offline-verify a receipt JSON file (requires explicit key source).",
    )
    verify.add_argument("receipt_json", type=Path)
    verify.add_argument(
        "--store",
        type=Path,
        default=None,
        help="Store root with keys/<key_id>.json public verification material.",
    )
    verify.add_argument(
        "--public-key-hex",
        default=None,
        help="Ed25519 public key hex (32 bytes). Requires --key-id.",
    )
    verify.add_argument(
        "--public-key-file",
        type=Path,
        default=None,
        help="File with Ed25519 public key (PEM or hex). Requires --key-id.",
    )
    verify.add_argument(
        "--key-id",
        default=None,
        help="key_id expected in receipt (required with --public-key-hex/file).",
    )
    verify.add_argument(
        "--demo-key",
        action="store_true",
        help=(
            "Local/test only: reconstruct known offline demo verification key. "
            "Fails when receipt key_id is not the demo key."
        ),
    )


def run_receipt(args: argparse.Namespace) -> int:
    if args.receipt_command == "verify":
        return _verify_receipt_file(
            Path(args.receipt_json),
            store=Path(args.store) if args.store is not None else None,
            public_key_hex=args.public_key_hex,
            public_key_file=Path(args.public_key_file) if args.public_key_file is not None else None,
            key_id=args.key_id,
            demo_key=bool(args.demo_key),
        )
    print(f"Unknown receipt command: {args.receipt_command}", file=sys.stderr)
    return 2


def _load_public_key_file(path: Path) -> bytes:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore").strip()
    if "BEGIN PUBLIC KEY" in text or "BEGIN PRIVATE KEY" in text:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key

        if "BEGIN PRIVATE KEY" in text:
            raise ValueError("private_key_file_not_allowed")
        key = load_pem_public_key(raw)
        public_bytes = key.public_bytes_raw()  # type: ignore[attr-defined]
        if len(public_bytes) != 32:
            raise ValueError("unsupported_public_key_length")
        return public_bytes
    hex_text = "".join(text.split())
    try:
        key_bytes = bytes.fromhex(hex_text)
    except ValueError as exc:
        raise ValueError("malformed_public_key_file") from exc
    if len(key_bytes) != 32:
        raise ValueError("unsupported_public_key_length")
    return key_bytes


def _build_key_resolver(
    *,
    store: Path | None,
    public_key_hex: str | None,
    public_key_file: Path | None,
    key_id: str | None,
    demo_key: bool,
    receipt_key_id: str,
) -> tuple[object, str] | tuple[None, str]:
    sources: list[str] = []
    if store is not None:
        sources.append("store")
    if public_key_hex:
        sources.append("public_key_hex")
    if public_key_file is not None:
        sources.append("public_key_file")
    if demo_key:
        sources.append("demo_key")

    if not sources:
        return None, "verification_key_source_required"
    if len(sources) > 1:
        return None, "exactly_one_verification_key_source_required"

    source = sources[0]
    if source == "store":
        assert store is not None
        if not store.exists():
            return None, "store_not_found"
        return FilesystemHostKeyResolver(store), "store"

    if source == "demo_key":
        if receipt_key_id != DEMO_OFFLINE_KEY_ID:
            return None, "demo_key_id_mismatch"
        attestor = build_deterministic_test_attestor(key_id=DEMO_OFFLINE_KEY_ID)
        return (
            StaticKeyResolver(
                {attestor.key_id: attestor.public_key_bytes},
                current_key_id=attestor.key_id,
            ),
            "demo_key",
        )

    expected_key_id = (key_id or "").strip()
    if not expected_key_id:
        return None, "key_id_required"
    if expected_key_id != receipt_key_id:
        return None, "key_id_mismatch"

    if source == "public_key_hex":
        assert public_key_hex is not None
        try:
            key_bytes = bytes.fromhex(public_key_hex.strip())
        except ValueError:
            return None, "malformed_public_key_hex"
        if len(key_bytes) != 32:
            return None, "unsupported_public_key_length"
        return (
            StaticKeyResolver({expected_key_id: key_bytes}, current_key_id=expected_key_id),
            "public_key_hex",
        )

    assert public_key_file is not None
    if not public_key_file.is_file():
        return None, "public_key_file_not_found"
    try:
        key_bytes = _load_public_key_file(public_key_file)
    except ValueError as exc:
        return None, str(exc)
    return (
        StaticKeyResolver({expected_key_id: key_bytes}, current_key_id=expected_key_id),
        "public_key_file",
    )


def _verify_receipt_file(
    receipt_path: Path,
    *,
    store: Path | None,
    public_key_hex: str | None,
    public_key_file: Path | None,
    key_id: str | None,
    demo_key: bool,
) -> int:
    if not receipt_path.is_file():
        _print_json(
            {
                "valid": False,
                "errors": ["receipt_file_not_found"],
                "receipt_path": _display_path(receipt_path),
            }
        )
        return 1
    receipt = parse_execution_evidence_proof_receipt_json(receipt_path.read_text(encoding="utf-8"))
    receipt_key_id = receipt.host_attestation.key_id
    resolver, key_source_or_error = _build_key_resolver(
        store=store,
        public_key_hex=public_key_hex,
        public_key_file=public_key_file,
        key_id=key_id,
        demo_key=demo_key,
        receipt_key_id=receipt_key_id,
    )
    if resolver is None:
        _print_json(
            {
                "valid": False,
                "errors": [key_source_or_error],
                "key_id": receipt_key_id,
                "receipt_path": _display_path(receipt_path),
            }
        )
        return 1

    try:
        if receipt.host_attestation.algorithm != ALGORITHM_ED25519:
            _print_json(
                {
                    "valid": False,
                    "schema_valid": False,
                    "digest_valid": False,
                    "signature_valid": False,
                    "key_id": receipt_key_id,
                    "key_source": key_source_or_error,
                    "errors": ["unsupported_algorithm"],
                    "receipt_path": _display_path(receipt_path),
                }
            )
            return 1
        result = verify_execution_evidence_proof_receipt(
            receipt,
            key_resolver=resolver,  # type: ignore[arg-type]
            require_policy_bundle_artifact=receipt.policy_bundle_artifact is not None,
        )
    except ValueError as exc:
        _print_json(
            {
                "valid": False,
                "errors": [str(exc)],
                "key_id": receipt_key_id,
                "key_source": key_source_or_error,
                "receipt_path": _display_path(receipt_path),
            }
        )
        return 1

    _print_json(
        {
            "valid": result.valid,
            "schema_valid": result.schema_valid,
            "digest_valid": result.digest_valid,
            "signature_valid": result.signature_valid,
            "key_id": result.key_id,
            "key_source": key_source_or_error,
            "errors": list(result.errors),
            "receipt_path": _display_path(receipt_path),
        }
    )
    return 0 if result.valid else 1
