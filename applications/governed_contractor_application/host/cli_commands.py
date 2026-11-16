# © Artur Czarnecki. All rights reserved.

"""Governed-contractor CLI commands (Tier-3 launcher injection)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.runtime.execution_evidence.attestor import build_deterministic_test_attestor
from intergrax.runtime.execution_evidence.key_store import (
    DEMO_OFFLINE_KEY_ID,
    FilesystemHostKeyResolver,
    read_demo_mode_marker,
)
from intergrax.runtime.execution_evidence.verify import verify_proof_receipt
from intergrax.runtime.governance.decision_requirement_policy import (
    PermissiveDecisionRequirementPolicy,
)
from intergrax.runtime.policy.runtime_policy_bundle_evaluator import (
    RuntimePolicyBundleEvaluator,
)

from applications.governed_contractor_application.host.collaborative_work_local_fixture import (
    build_seeded_in_memory_external_work_authorization_boundary,
)
from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
from governed_contractor_application.host.offline_demo import (
    build_demo_policy_bundle,
    display_relative_path,
    run_offline_governed_contractor_demo,
)
from governed_contractor_application.host.orchestrator import GovernedExternalWorkOrchestrator
from governed_contractor_application.host.stores import (
    FilesystemHostStore,
    InMemoryProviderInvocationStore,
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


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


def _run_full_demo(store: Path, *, simulate_signing_failure: bool) -> int:
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
    attestor = _resolve_retry_attestor(store=store, demo_key=demo_key)
    if attestor is None:
        return 1

    fs = FilesystemHostStore(store)
    bundle = build_demo_policy_bundle()
    persisted = fs.get_bundle(bundle.bundle_id, bundle.version)
    if persisted is not None:
        bundle = persisted
    policy = RuntimePolicyBundleEvaluator(bundle)
    authorization_boundary = build_seeded_in_memory_external_work_authorization_boundary(
        policy,
        tenant_id="offline-demo-tenant",
        workspace_id="workspace-a",
        principal_id="offline-demo-user",
        decision_requirement_policy=PermissiveDecisionRequirementPolicy(),
    )
    # RefuseProvider ensures retry cannot execute side effects even if miswired.
    orch = GovernedExternalWorkOrchestrator(
        adapter=ExternalWorkAdapter(_RefuseProvider(), authorization_boundary=authorization_boundary),  # type: ignore[arg-type]
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
        provider_invocation_store=InMemoryProviderInvocationStore(),
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
            "attestation_succeeded": bool(step.attestation and step.attestation.attestation_succeeded),
            "receipt_id": step.receipt.receipt_id if step.receipt else None,
            "receipt_path": receipt_path,
            "verification_valid": verification_valid,
        }
    )
    return 0 if step.receipt is not None else 1


class GovernedContractorCliCommandSet:
    """Tier-3 command set for governed external-work demos."""

    def register_parsers(self, sub: argparse._SubParsersAction) -> None:
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
        demo_create.set_defaults(cli_run=run_external_work)

        demo_accept = ext_sub.add_parser(
            "demo-accept",
            help="Alias for full offline demo (create+accept).",
        )
        demo_accept.add_argument(
            "--store",
            type=Path,
            default=Path("build/external_work_demo"),
        )
        demo_accept.set_defaults(cli_run=run_external_work)

        show = ext_sub.add_parser("show", help="Show persisted GovernedExecutionResult.")
        show.add_argument("execution_id")
        show.add_argument("--store", type=Path, default=Path("build/external_work_demo"))
        show.set_defaults(cli_run=run_external_work)

        receipt = ext_sub.add_parser("receipt", help="Show persisted ProofReceipt path/JSON.")
        receipt.add_argument("execution_id")
        receipt.add_argument("--store", type=Path, default=Path("build/external_work_demo"))
        receipt.set_defaults(cli_run=run_external_work)

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
        retry.set_defaults(cli_run=run_external_work)

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
            help=("Persist provider-success GER then fail attestation so a fresh process can run retry-attestation."),
        )
        gov.set_defaults(cli_run=run_demo)


GOVERNED_CONTRACTOR_COMMANDS = GovernedContractorCliCommandSet()

__all__ = [
    "GOVERNED_CONTRACTOR_COMMANDS",
    "GovernedContractorCliCommandSet",
    "run_demo",
    "run_external_work",
]
