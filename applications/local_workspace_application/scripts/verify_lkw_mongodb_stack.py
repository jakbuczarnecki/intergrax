#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Platform-backed MongoDB document-store infrastructure smoke validation (PROOF-RECEIPTS-1D)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_integration
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
)

SMOKE_PARTITION_KEY = "platform_smoke"
SMOKE_ROW_KEY = "mongodb_document_store"
SMOKE_DATA: dict[str, str] = {
    "proof_kind": "infrastructure_connectivity",
    "task": "PROOF-RECEIPTS-1D",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate MongoDBDocumentStoreIntegration → as_document_store() → "
            "DocumentStore put/get against a live MongoDB backend."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "read-only"),
        default="smoke",
        help="smoke: platform put + get; read-only: get existing smoke record only.",
    )
    parser.add_argument(
        "--verify-volume-configured",
        action="store_true",
        help="Assert named persistent volume is present in resolved compose config.",
    )
    parser.add_argument(
        "--compose-config",
        default="",
        help="Path to resolved docker compose config output for volume inspection.",
    )
    parser.add_argument(
        "--volume-only",
        action="store_true",
        help="Only verify persistent volume configuration; skip platform smoke.",
    )
    return parser.parse_args()


def _resolve_host_mongodb_uri() -> str | None:
    explicit = os.environ.get("INTERGRAX_MONGODB_URI", "").strip()
    if explicit:
        return explicit

    username = os.environ.get("LKW_MONGODB_ROOT_USERNAME", "intergrax").strip() or "intergrax"
    password = (
        os.environ.get("LKW_MONGODB_ROOT_PASSWORD", "intergrax-local-dev-only").strip()
        or "intergrax-local-dev-only"
    )
    database = os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip() or "intergrax_proofs"
    host_port = os.environ.get("LKW_MONGODB_HOST_PORT", "27018").strip() or "27018"
    return (
        f"mongodb://{username}:{password}@127.0.0.1:{host_port}/{database}?authSource=admin"
    )


def _ensure_mongodb_env() -> None:
    if not os.environ.get("INTERGRAX_MONGODB_URI", "").strip():
        resolved = _resolve_host_mongodb_uri()
        if resolved:
            os.environ["INTERGRAX_MONGODB_URI"] = resolved
    if not os.environ.get("INTERGRAX_MONGODB_DATABASE", "").strip():
        os.environ["INTERGRAX_MONGODB_DATABASE"] = (
            os.environ.get("LKW_MONGODB_DATABASE", "intergrax_proofs").strip() or "intergrax_proofs"
        )
    if not os.environ.get("INTERGRAX_MONGODB_COLLECTION", "").strip():
        os.environ["INTERGRAX_MONGODB_COLLECTION"] = (
            os.environ.get("LKW_MONGODB_COLLECTION", "proof_receipts").strip() or "proof_receipts"
        )


def _records_match(left: DocumentRecord, right: DocumentRecord) -> bool:
    return (
        left.partition_key == right.partition_key
        and left.row_key == right.row_key
        and dict(left.data) == dict(right.data)
    )


def _verify_volume_configured(compose_config_path: str) -> bool:
    if not compose_config_path:
        return False
    path = os.path.abspath(compose_config_path)
    if not os.path.isfile(path):
        return False
    content = open(path, encoding="utf-8").read()
    return "lkw_mongodb_data" in content


def _run_smoke(*, read_only: bool) -> dict[str, Any]:
    _ensure_mongodb_env()
    bundle = create_mongodb_integration()
    integration = bundle.document_store
    if not isinstance(integration, MongoDBDocumentStoreIntegration):
        raise TypeError("integration_not_mongodb_document_store")
    store = integration.as_document_store()
    if store is None:
        raise RuntimeError("document_store_adapter_unresolved")

    expected = DocumentRecord(
        partition_key=SMOKE_PARTITION_KEY,
        row_key=SMOKE_ROW_KEY,
        data=dict(SMOKE_DATA),
    )

    platform_put = False
    platform_get = False
    smoke_record_verified = False

    try:
        if not read_only:
            store.put(expected)
            platform_put = True

        loaded = store.get(SMOKE_PARTITION_KEY, SMOKE_ROW_KEY)
        platform_get = loaded is not None
        smoke_record_verified = loaded is not None and _records_match(expected, loaded)
    except Exception:
        store.close()
        raise
    else:
        store.close()

    proof_pass = smoke_record_verified
    return {
        "proof_result": "PASS" if proof_pass else "FAIL",
        "proof_kind": "platform_document_store_infrastructure",
        "document_store_provider": MONGODB_DOCUMENT_STORE_PROVIDER_ID,
        "integration_kind": "document_store",
        "integration_class": type(integration).__name__,
        "adapter_resolved": True,
        "platform_put": platform_put if not read_only else "skipped",
        "platform_get": platform_get,
        "smoke_record_verified": smoke_record_verified,
        "direct_pymongo_from_lkw": False,
        "direct_mongosh_write": False,
        "proof_receipt_recording": False,
        "smoke_partition_key": SMOKE_PARTITION_KEY,
        "smoke_row_key": SMOKE_ROW_KEY,
    }


def _print_results(results: dict[str, Any]) -> None:
    for key in (
        "proof_result",
        "proof_kind",
        "document_store_provider",
        "integration_kind",
        "integration_class",
        "adapter_resolved",
        "platform_put",
        "platform_get",
        "smoke_record_verified",
        "persistent_volume_configured",
        "direct_mongosh_write",
        "direct_pymongo_from_lkw",
        "proof_receipt_recording",
    ):
        if key in results:
            print(f"{key}={results[key]}")


def main() -> int:
    args = _parse_args()
    results: dict[str, Any] = {}

    if args.verify_volume_configured:
        volume_ok = _verify_volume_configured(args.compose_config)
        results["persistent_volume_configured"] = volume_ok
        if not volume_ok:
            results["proof_result"] = "FAIL"
            results["reason"] = "persistent_volume_not_configured"
            _print_results(results)
            return 1
        if args.volume_only:
            results["proof_result"] = "PASS"
            _print_results(results)
            return 0

    try:
        smoke_results = _run_smoke(read_only=args.mode == "read-only")
        results.update(smoke_results)
    except Exception as exc:
        results["proof_result"] = "FAIL"
        results["reason"] = type(exc).__name__
        results["message"] = str(exc)
        _print_results(results)
        return 1

    _print_results(results)
    return 0 if results.get("proof_result") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
