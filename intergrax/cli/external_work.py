# © Artur Czarnecki. All rights reserved.

"""``intergrax external-work`` / ``intergrax demo governed-contractor`` (PC-8)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.contracts.execution_evidence.receipt import ProofReceipt
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

    demo = sub.add_parser("demo", help="Reproducible offline demos.")
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    gov = demo_sub.add_parser(
        "governed-contractor",
        help="Full offline CREATE→ACCEPT→receipt→verify lifecycle.",
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

    rcpt = sub.add_parser("receipt", help="ProofReceipt tools.")
    rcpt_sub = rcpt.add_subparsers(dest="receipt_command", required=True)
    verify = rcpt_sub.add_parser("verify", help="Offline-verify a receipt JSON file.")
    verify.add_argument("receipt_json", type=Path)
    verify.add_argument(
        "--public-key-hex",
        default=None,
        help="Optional Ed25519 public key hex (32 bytes). "
        "When omitted, uses key embedded in demo store report if present.",
    )
    verify.add_argument(
        "--key-id",
        default=None,
        help="key_id expected in receipt (required with --public-key-hex).",
    )
    verify.add_argument(
        "--store",
        type=Path,
        default=Path("build/external_work_demo"),
        help="Demo store used to load the offline demo public key.",
    )


def run_external_work(args: argparse.Namespace) -> int:
    cmd = args.external_work_command
    if cmd in {"demo-create", "demo-accept"}:
        return _run_full_demo(Path(args.store))
    if cmd == "show":
        return _show_execution(Path(args.store), args.execution_id)
    if cmd == "receipt":
        return _show_receipt(Path(args.store), args.execution_id)
    if cmd == "retry-attestation":
        return _retry_attestation(Path(args.store), args.execution_id)
    print(f"Unknown external-work command: {cmd}", file=sys.stderr)
    return 2


def run_demo(args: argparse.Namespace) -> int:
    if args.demo_command == "governed-contractor":
        return _run_full_demo(Path(args.store))
    print(f"Unknown demo command: {args.demo_command}", file=sys.stderr)
    return 2


def run_receipt(args: argparse.Namespace) -> int:
    if args.receipt_command == "verify":
        return _verify_receipt_file(
            Path(args.receipt_json),
            public_key_hex=args.public_key_hex,
            key_id=args.key_id,
            store=Path(args.store),
        )
    print(f"Unknown receipt command: {args.receipt_command}", file=sys.stderr)
    return 2


def _run_full_demo(store: Path) -> int:
    _ensure_tier_paths()
    from governed_contractor_application.host.offline_demo import (
        run_offline_governed_contractor_demo,
    )

    report = run_offline_governed_contractor_demo(store_root=store)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
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
    print(str(path.resolve()))
    print(receipt.model_dump_json(indent=2))
    return 0


def _retry_attestation(store: Path, execution_id: str) -> int:
    """Rebuild orchestrator against persisted store and retry attestation only."""
    _ensure_tier_paths()
    from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
    from external_contractor_adapter.tests.fakes.deterministic_external_work import (
        DeterministicExternalWorkFake,
    )
    from governed_contractor_application.host.offline_demo import build_demo_policy_bundle
    from governed_contractor_application.host.orchestrator import (
        GovernedExternalWorkOrchestrator,
    )
    from governed_contractor_application.host.stores import FilesystemHostStore
    from intergrax.contracts.external_work_provider_capabilities import (
        quote_first_partner_capability_fixture,
    )
    from intergrax.runtime.execution_evidence.attestor import (
        build_deterministic_test_attestor,
    )
    from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
        RuntimePolicyBundleEvaluator,
    )

    fs = FilesystemHostStore(store)
    bundle = build_demo_policy_bundle()
    # Prefer persisted bundle artifact when present.
    persisted = fs.get_bundle(bundle.bundle_id, bundle.version)
    if persisted is not None:
        bundle = persisted
    policy = RuntimePolicyBundleEvaluator(bundle)
    attestor = build_deterministic_test_attestor(
        key_id="governed-contractor-offline-demo-1",
    )
    orch = GovernedExternalWorkOrchestrator(
        adapter=ExternalWorkAdapter(
            DeterministicExternalWorkFake(), side_effect_policy=policy
        ),
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
    step = orch.retry_attestation(execution_id)
    print(
        json.dumps(
            {
                "execution_id": execution_id,
                "state": step.state.value,
                "reason": step.reason,
                "attestation_succeeded": bool(
                    step.attestation and step.attestation.attestation_succeeded
                ),
                "receipt_id": step.receipt.receipt_id if step.receipt else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if step.receipt is not None else 1


def _verify_receipt_file(
    receipt_path: Path,
    *,
    public_key_hex: str | None,
    key_id: str | None,
    store: Path,
) -> int:
    if not receipt_path.is_file():
        print(f"receipt file not found: {receipt_path}", file=sys.stderr)
        return 1
    receipt = ProofReceipt.model_validate_json(
        receipt_path.read_text(encoding="utf-8")
    )
    if public_key_hex and key_id:
        key_bytes = bytes.fromhex(public_key_hex)
        resolver = StaticKeyResolver({key_id: key_bytes}, current_key_id=key_id)
    else:
        # Load key from a fresh offline attestor with the demo key id.
        from intergrax.runtime.execution_evidence.attestor import (
            build_deterministic_test_attestor,
        )

        attestor = build_deterministic_test_attestor(
            key_id=receipt.host_attestation.key_id
            or "governed-contractor-offline-demo-1",
        )
        # Deterministic attestor uses fixed seed per key_id — must match signer.
        if attestor.key_id != receipt.host_attestation.key_id:
            attestor = build_deterministic_test_attestor(
                key_id=receipt.host_attestation.key_id,
            )
        resolver = StaticKeyResolver(
            {attestor.key_id: attestor.public_key_bytes},
            current_key_id=attestor.key_id,
        )
        _ = store
    result = verify_proof_receipt(
        receipt,
        key_resolver=resolver,
        require_policy_bundle_artifact=receipt.policy_bundle_artifact is not None,
    )
    print(
        json.dumps(
            {
                "valid": result.valid,
                "schema_valid": result.schema_valid,
                "digest_valid": result.digest_valid,
                "signature_valid": result.signature_valid,
                "key_id": result.key_id,
                "errors": list(result.errors),
                "receipt_path": str(receipt_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.valid else 1
