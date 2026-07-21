# © Artur Czarnecki. All rights reserved.

"""``intergrax external-work`` / ``intergrax demo governed-contractor`` (PC-8 / FH)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.contracts.execution_evidence.receipt import ProofReceipt
from intergrax.runtime.execution_evidence.attestor import (
    ALGORITHM_ED25519,
    build_deterministic_test_attestor,
)
from intergrax.runtime.execution_evidence.key_store import (
    DEMO_OFFLINE_KEY_ID,
    FilesystemHostKeyResolver,
    read_demo_mode_marker,
)
from intergrax.runtime.execution_evidence.verify import (
    StaticKeyResolver,
    verify_proof_receipt,
)


def _ensure_tier_paths() -> None:
    """Make Tier-2/Tier-3 packages importable when running from the repo root."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for base in candidates:
        apps = base / "applications"
        agents = base / "agents"
        if apps.is_dir() and agents.is_dir():
            for rel in (".", "applications", "agents"):
                path = str((base / rel).resolve())
                if path not in sys.path:
                    sys.path.insert(0, path)
            return


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def register_parser(sub: argparse._SubParsersAction) -> None:
    ext = sub.add_parser(
        "external-work",
        help="Governed external-work host demo and receipt tools.",
    )
    ext_sub = ext.add_subparsers(dest="external_work_command", required=True)

    demo_create = ext_sub.add_parser(
        "demo-create",
        help="Run offline CREATE segment of governed-contractor demo.",
    )
    demo_create.add_argument(
        "--store",
        type=Path,
        default=Path("build/external_work_demo"),
        help="Filesystem store root.",
    )

    demo_accept = ext_sub.add_parser(
        "demo-accept",
        help="Alias for full offline demo (create+accept).",
    )
    demo_accept.add_argument(
        "--store",
        type=Path,
        default=Path("build/external_work_demo"),
    )

    show = ext_sub.add_parser("show", help="Show persisted GovernedExecutionResult.")
    show.add_argument("execution_id")
    show.add_argument("--store", type=Path, default=Path("build/external_work_demo"))

    receipt = ext_sub.add_parser("receipt", help="Show persisted ProofReceipt path/JSON.")
    receipt.add_argument("execution_id")
    receipt.add_argument("--store", type=Path, default=Path("build/external_work_demo"))

    retry = ext_sub.add_parser(
        "retry-attestation",
        help="Attestation-only recovery for a persisted execution.",
    )
    retry.add_argument("execution_id")
    retry.add_argument("--store", type=Path, default=Path("build/external_work_demo"))
    retry.add_argument(
        "--demo-key",
        action="store_true",
        help="Use local/test deterministic demo signer (explicit; not production).",
    )

    demo = sub.add_parser("demo", help="Reproducible offline demos.")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    gov = demo_sub.add_parser(
        "governed-contractor",
        help="Full offline CREATE -> ACCEPT -> receipt -> verify lifecycle.",
    )
    gov.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Offline deterministic fake (default).",
    )
    gov.add_argument(
        "--store",
        type=Path,
        default=Path("build/external_work_demo"),
    )
    gov.add_argument(
        "--simulate-signing-failure",
        action="store_true",
        help=(
            "Persist provider-success GER then fail attestation so a fresh "
            "process can run retry-attestation."
        ),
    )

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


def run_external_work(args: argparse.Namespace) -> int:
    cmd = args.external_work_command
    if cmd in {"demo-create", "demo-accept"}:
        return _run_full_demo(Path(args.store), simulate_signing_failure=False)
    if cmd == "show":
        return _show_execution(Path(args.store), args.execution_id)
    if cmd == "receipt":
        return _show_receipt(Path(args.store), args.execution_id)
    if cmd == "retry-attestation":
        return _retry_attestation(
            Path(args.store),
            args.execution_id,
            demo_key=bool(args.demo_key),
        )
    print(f"Unknown external-work command: {cmd}", file=sys.stderr)
    return 2


def run_demo(args: argparse.Namespace) -> int:
    if args.demo_command == "governed-contractor":
        return _run_full_demo(
            Path(args.store),
            simulate_signing_failure=bool(args.simulate_signing_failure),
        )
    print(f"Unknown demo command: {args.demo_command}", file=sys.stderr)
    return 2


def run_receipt(args: argparse.Namespace) -> int:
    if args.receipt_command == "verify":
        return _verify_receipt_file(
            Path(args.receipt_json),
            store=Path(args.store) if args.store is not None else None,
            public_key_hex=args.public_key_hex,
            public_key_file=Path(args.public_key_file)
            if args.public_key_file is not None
            else None,
            key_id=args.key_id,
            demo_key=bool(args.demo_key),
        )
    print(f"Unknown receipt command: {args.receipt_command}", file=sys.stderr)
    return 2


def _run_full_demo(store: Path, *, simulate_signing_failure: bool) -> int:
    _ensure_tier_paths()
    from governed_contractor_application.host.offline_demo import (
        run_offline_governed_contractor_demo,
    )

    report = run_offline_governed_contractor_demo(
        store_root=store,
        simulate_signing_failure=simulate_signing_failure,
    )
    payload = report.as_dict()
    if simulate_signing_failure:
        # FH-8 / FH-11: highlight recovery fields; exit 0 for expected demo outcome.
        payload = {
            **payload,
            "execution_id": report.accept_execution_id,
            "state": report.state,
            "provider_execution_succeeded": report.provider_execution_succeeded,
            "attestation_succeeded": report.attestation_succeeded,
            "receipt_path": None,
            "recovery_command": report.recovery_command,
            "verification_key_path": report.verification_key_path,
            "store_root": report.store_root,
            "create_calls": report.create_calls,
            "accept_calls": report.accept_calls,
            "cancel_calls": report.cancel_calls,
        }
        _print_json(payload)
        return 0
    _print_json(payload)
    return 0 if report.verification_valid else 1


def _show_execution(store: Path, execution_id: str) -> int:
    _ensure_tier_paths()
    from governed_contractor_application.host.stores import FilesystemHostStore

    fs = FilesystemHostStore(store)
    result = fs.get_result(execution_id)
    if result is None:
        print(f"execution not found: {execution_id}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    state = fs.get_state(execution_id)
    if state is not None:
        print(f"\n# host_state={state.value}", file=sys.stderr)
    return 0


def _show_receipt(store: Path, execution_id: str) -> int:
    _ensure_tier_paths()
    from governed_contractor_application.host.stores import FilesystemHostStore

    fs = FilesystemHostStore(store)
    receipt = fs.get_receipt(execution_id)
    if receipt is None:
        print(f"receipt not found: {execution_id}", file=sys.stderr)
        return 1
    path = store / "receipts" / f"{execution_id}.json"
    print(_display_path(path))
    print(receipt.model_dump_json(indent=2))
    return 0


class _RefuseProvider:
    """Integration stub that fails closed if any provider method is called."""

    def __getattr__(self, name: str):
        def _boom(*_a, **_k):
            raise RuntimeError(f"provider_must_not_be_invoked:{name}")

        return _boom


def _resolve_retry_attestor(*, store: Path, demo_key: bool):
    marker = read_demo_mode_marker(store)
    use_demo = demo_key or (
        marker is not None
        and marker.get("recovery_signer") == "deterministic_demo"
        and marker.get("mode") == "offline_deterministic_demo"
    )
    if not use_demo:
        _print_json(
            {
                "valid": False,
                "errors": ["signing_key_source_required"],
                "hint": "Pass --demo-key for offline demo recovery, or inject HostAttestor.",
            }
        )
        return None
    key_id = DEMO_OFFLINE_KEY_ID
    if marker and isinstance(marker.get("key_id"), str) and marker["key_id"].strip():
        key_id = marker["key_id"].strip()
    return build_deterministic_test_attestor(key_id=key_id)


def _retry_attestation(store: Path, execution_id: str, *, demo_key: bool) -> int:
    """Rebuild orchestrator against persisted store and retry attestation only."""
    _ensure_tier_paths()
    from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
    from governed_contractor_application.host.offline_demo import (
        build_demo_policy_bundle,
        display_relative_path,
    )
    from governed_contractor_application.host.orchestrator import (
        GovernedExternalWorkOrchestrator,
    )
    from governed_contractor_application.host.stores import FilesystemHostStore
    from intergrax.contracts.external_work_provider_capabilities import (
        quote_first_partner_capability_fixture,
    )
    from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
        RuntimePolicyBundleEvaluator,
    )

    attestor = _resolve_retry_attestor(store=store, demo_key=demo_key)
    if attestor is None:
        return 1

    fs = FilesystemHostStore(store)
    bundle = build_demo_policy_bundle()
    persisted = fs.get_bundle(bundle.bundle_id, bundle.version)
    if persisted is not None:
        bundle = persisted
    policy = RuntimePolicyBundleEvaluator(bundle)
    # RefuseProvider ensures retry cannot execute side effects even if miswired.
    orch = GovernedExternalWorkOrchestrator(
        adapter=ExternalWorkAdapter(_RefuseProvider(), side_effect_policy=policy),  # type: ignore[arg-type]
        policy=policy,
        bundle=bundle,
        attestor=attestor,
        capabilities=quote_first_partner_capability_fixture(
            provider_id="gec3_deterministic_fake",
        ),
        execution_store=fs,
        receipt_store=fs,
        bundle_store=fs,
        continuation_store=fs,
    )
    try:
        step = orch.retry_attestation(execution_id)
    except ValueError as exc:
        _print_json(
            {
                "execution_id": execution_id,
                "valid": False,
                "errors": [str(exc)],
                "provider_invoked": False,
            }
        )
        return 1

    receipt_path: str | None = None
    verification_valid = False
    if step.receipt is not None:
        export_path = store / "export" / "accept_receipt.json"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(step.receipt.model_dump_json(indent=2), encoding="utf-8")
        receipt_path = display_relative_path(export_path)
        try:
            resolver = FilesystemHostKeyResolver(store)
            vr = verify_proof_receipt(
                step.receipt,
                key_resolver=resolver,
                require_policy_bundle_artifact=True,
            )
            verification_valid = vr.valid
        except ValueError:
            verification_valid = False

    provider_invoked = False  # retry-attestation never repeats provider side effects
    _print_json(
        {
            "execution_id": execution_id,
            "state": step.state.value,
            "reason": step.reason,
            "provider_invoked": provider_invoked,
            "attestation_succeeded": bool(
                step.attestation and step.attestation.attestation_succeeded
            ),
            "receipt_id": step.receipt.receipt_id if step.receipt else None,
            "receipt_path": receipt_path,
            "verification_valid": verification_valid,
        }
    )
    return 0 if step.receipt is not None else 1


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
    receipt = ProofReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
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
        result = verify_proof_receipt(
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
