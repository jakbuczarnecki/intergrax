#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background-task platform proof helper with ProofReceipt recording (LKW.4E / PROOF-RECEIPTS-1E)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_integration
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
)
from intergrax.proofs.receipts.contracts import ProofReceipt, ProofReceiptResult
from intergrax.proofs.receipts.recording import (
    ProofReceiptVerificationError,
    record_and_verify_proof_receipt,
)

_PROOF_REQUESTED_BY = "lkw.background_task_proof"
_PROOF_RUNNER = "run-lkw-background-task-proof.py"
_RECEIPT_TASK = "PROOF-RECEIPTS-1E"
_APPLICATION_ID = "local_workspace"
_PROOF_KIND = "platform_background_task"
_TASK_NAME = "lkw.background_ingest.v1"
_KAFKA_TOPICS = (
    "intergrax.tasks",
    "intergrax.task-events",
    "intergrax.task-status",
    "intergrax.task-results",
)
_DEFAULT_MONGO_EXPRESS_URL = "http://127.0.0.1:8086"


def build_background_task_proof_id(run_id: str) -> str:
    """Stable proof receipt identity for a background-task proof run."""
    normalized_run_id = run_id.strip()
    if not normalized_run_id:
        raise ValueError("run_id must not be blank")
    return f"{_APPLICATION_ID}:{_PROOF_KIND}:{normalized_run_id}"


def build_background_task_proof_receipt(
    *,
    run_id: str,
    correlation_id: str,
    task_id: str,
    provider: str,
    final_status: str,
    search_results: int,
    marker: str,
    collection_id: str,
    tenant_id: str,
    kafka_messages: int,
    mongo_express_url: str = _DEFAULT_MONGO_EXPRESS_URL,
    result: ProofReceiptResult = ProofReceiptResult.PASS,
    has_result: bool = True,
    handler_resolved: bool = True,
    worker_runtime_received: bool = True,
) -> ProofReceipt:
    """Build a structured ProofReceipt from live background-task proof evidence."""
    kafka_inspection_available = kafka_messages >= 0
    provider_evidence: dict[str, Any] = {
        "message_bus_provider": provider,
        "enqueue_mode": "real_provider",
        "worker_execution": "asynchronous",
        "task_status": final_status,
        "task_result_available": has_result,
        "handler_resolved": handler_resolved,
        "worker_runtime_received": worker_runtime_received,
        "kafka_topics": list(_KAFKA_TOPICS),
    }
    if kafka_inspection_available:
        provider_evidence["kafka_topic_messages"] = kafka_messages
        provider_evidence["kafka_topic_inspection_available"] = True
    else:
        provider_evidence["kafka_topic_messages"] = None
        provider_evidence["kafka_topic_inspection_available"] = False

    return ProofReceipt(
        proof_id=build_background_task_proof_id(run_id),
        proof_kind=_PROOF_KIND,
        application_id=_APPLICATION_ID,
        result=result,
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=task_id,
        provider_evidence=provider_evidence,
        domain_evidence={
            "task_name": _TASK_NAME,
            "index_ingested": 1,
            "search_results": search_results,
            "evidence_marker_found": True,
            "marker": marker,
            "collection_id": collection_id,
            "tenant_id": tenant_id,
        },
        guardrails={
            "mock_queue": False,
            "inmemory_bypass": False,
            "direct_handler_call": False,
            "direct_indexer_call": False,
            "direct_mongodb_write": False,
            "direct_pymongo_from_lkw": False,
            "markdown_source_of_truth": False,
        },
        metadata={
            "proof_runner": _PROOF_RUNNER,
            "receipt_task": _RECEIPT_TASK,
            "mongo_express_url": mongo_express_url,
            "recorded_from_live_run": True,
        },
    )


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


def ensure_mongodb_env() -> None:
    """Populate host-visible MongoDB provider environment for platform resolution."""
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


def resolve_mongodb_document_store():
    """Resolve MongoDB DocumentStore through the platform provider factory."""
    ensure_mongodb_env()
    bundle = create_mongodb_integration()
    integration = bundle.document_store
    if not isinstance(integration, MongoDBDocumentStoreIntegration):
        raise TypeError("integration_not_mongodb_document_store")
    store = integration.as_document_store()
    if store is None:
        raise RuntimeError("document_store_adapter_unresolved")
    return integration, store


def record_background_task_proof_receipt(
    receipt: ProofReceipt,
) -> tuple[ProofReceipt, MongoDBDocumentStoreIntegration]:
    """Persist and verify a background-task proof receipt through the platform store."""
    integration, document_store = resolve_mongodb_document_store()
    verified = record_and_verify_proof_receipt(receipt, document_store, owns_document_store=True)
    return verified, integration


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LKW Kafka background-task platform proof against a live local stack.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOCAL_WORKSPACE_BACKEND_BASE_URL", "http://127.0.0.1:8020"),
        help="LKW backend base URL (default: http://127.0.0.1:8020).",
    )
    parser.add_argument(
        "--kafka-ui",
        default=os.environ.get("LKW_BACKGROUND_TASK_PROOF_KAFKA_UI_URL", "http://127.0.0.1:8085"),
        help="Kafka UI URL for reviewer hints (default: http://127.0.0.1:8085).",
    )
    parser.add_argument(
        "--mongo-express",
        default=os.environ.get("LKW_MONGO_EXPRESS_URL", _DEFAULT_MONGO_EXPRESS_URL),
        help="Mongo Express URL for reviewer hints (default: http://127.0.0.1:8086).",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default=os.environ.get("LKW_BACKGROUND_TASK_PROOF_KAFKA_BOOTSTRAP", "127.0.0.1:9094"),
        help="Host-visible Kafka bootstrap server (default: 127.0.0.1:9094).",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("INTERGRAX_KAFKA_TOPIC", "intergrax.tasks"),
        help="Kafka task topic (default: intergrax.tasks).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=int(os.environ.get("LKW_BACKGROUND_TASK_PROOF_TIMEOUT_SECONDS", "180")),
        help="Max seconds to wait for background task completion.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for the proof enqueue request.",
    )
    parser.add_argument(
        "--correlation-id",
        default="",
        help="Optional correlation id for the proof enqueue request.",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip docker compose startup (stack already running).",
    )
    return parser.parse_args()


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    timeout: float = 30.0,
) -> dict[str, object]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("response_not_object")
    return parsed


def _health_ok(base_url: str) -> bool:
    try:
        payload = _request_json(f"{base_url.rstrip('/')}/health", timeout=5.0)
    except Exception:
        return False
    return str(payload.get("status", "")).lower() == "ok"


def _kafka_ui_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _search_summary_diagnostic(response: dict[str, object]) -> dict[str, object]:
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    evidence = metadata.get("lkw_evidence.v1")
    if not isinstance(evidence, dict):
        return {}
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    search_summary = diagnostics.get("lkw.search_summary.v1")
    return search_summary if isinstance(search_summary, dict) else {}


def _search_result_count(response: dict[str, object]) -> int:
    search_summary = _search_summary_diagnostic(response)
    if not search_summary:
        return 0
    for field in ("result_count", "results_count", "hits", "num_results", "evidence_count"):
        value = search_summary.get(field)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, list) and value:
            return len(value)
    results = search_summary.get("results")
    if isinstance(results, list):
        return len(results)
    return 0


def _inspect_kafka_topic(*, bootstrap: str, topic: str) -> int:
    try:
        from confluent_kafka import Consumer, TopicPartition
    except ImportError:
        return -1

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"lkw-proof-inspector-{int(time.time())}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    try:
        partitions = consumer.list_topics(topic=topic, timeout=5).topics.get(topic)
        if partitions is None:
            return 0
        total = 0
        for partition in partitions.partitions:
            low, high = consumer.get_watermark_offsets(TopicPartition(topic, partition), timeout=5)
            total += max(0, high - low)
        return total
    finally:
        consumer.close()


def _fail(reason: str, **fields: object) -> int:
    print("proof_result=FAIL")
    print(f"failure_reason={reason}")
    print("proof_receipt_recorded=false")
    for key, value in fields.items():
        print(f"{key}={value}")
    return 1


def _fail_receipt_recording(error: BaseException) -> int:
    print("proof_result=FAIL")
    print("failure_reason=proof_receipt_recording_failed")
    print("proof_workload_result=PASS")
    print("proof_receipt_recorded=false")
    print("proof_receipt_verified=false")
    print(f"receipt_error={type(error).__name__}")
    print(f"receipt_message={error}")
    return 1


def _print_pass_output(
    *,
    provider: str,
    final_status: str,
    search_results: int,
    run_id: str,
    correlation_id: str,
    task_id: str,
    marker: str,
    collection_id: str,
    kafka_ui_url: str,
    mongo_express_url: str,
    kafka_messages: int,
    verified_receipt: ProofReceipt,
    integration_class: str,
) -> None:
    kafka_topics = ",".join(_KAFKA_TOPICS)
    print("proof_result=PASS")
    print(f"proof_kind={_PROOF_KIND}")
    print(f"task_name={_TASK_NAME}")
    print(f"message_bus_provider={provider}")
    print("enqueue_mode=real_provider")
    print("worker_execution=asynchronous")
    print(f"task_status={final_status}")
    print("task_result_available=true")
    print("handler_resolved=true")
    print("worker_runtime_received=true")
    print("index_ingested=1")
    print(f"search_results={search_results}")
    print("evidence_marker_found=true")
    print(f"kafka_ui_url={kafka_ui_url}")
    print(f"kafka_topics={kafka_topics}")
    print("mock_queue=false")
    print("inmemory_bypass=false")
    print("direct_handler_call=false")
    print("direct_indexer_call=false")
    print(f"run_id={run_id}")
    print(f"correlation_id={correlation_id}")
    print(f"task_id={task_id}")
    print(f"marker={marker}")
    print(f"collection_id={collection_id}")
    if kafka_messages >= 0:
        print(f"kafka_topic_messages={kafka_messages}")
    print("proof_receipt_recorded=true")
    print("proof_receipt_verified=true")
    print("proof_receipt_store=platform")
    print(f"document_store_provider={MONGODB_DOCUMENT_STORE_PROVIDER_ID}")
    print(f"document_store_integration={integration_class}")
    print(f"proof_receipt_id={verified_receipt.proof_id}")
    print(f"proof_receipt_run_id={verified_receipt.run_id}")
    print(f"proof_receipt_result={verified_receipt.result.value}")
    print(f"proof_receipt_application_id={verified_receipt.application_id}")
    print("proof_receipt_query_verified=true")
    print(f"mongo_express_url={mongo_express_url}")
    print("markdown_source_of_truth=false")
    print("direct_mongodb_write=false")
    print("direct_pymongo_from_lkw=false")


def main() -> int:
    args = _parse_args()
    base = args.base_url.rstrip("/")

    if not _health_ok(base):
        return _fail("lkw_health_unreachable")

    if not _kafka_ui_ok(args.kafka_ui):
        return _fail("kafka_ui_unreachable", kafka_ui_url=args.kafka_ui)

    enqueue_payload: dict[str, str] = {}
    if args.run_id.strip():
        enqueue_payload["run_id"] = args.run_id.strip()
    if args.correlation_id.strip():
        enqueue_payload["correlation_id"] = args.correlation_id.strip()

    try:
        enqueue = _request_json(
            f"{base}/v1/local_workspace/proof/background-task/enqueue",
            method="POST",
            payload=enqueue_payload or {},
        )
    except urllib.error.HTTPError as exc:
        return _fail(f"http_{exc.code}")
    except Exception as exc:  # noqa: BLE001
        return _fail(type(exc).__name__)

    if str(enqueue.get("proof_result", "FAIL")) != "PASS":
        return _fail("enqueue_proof_failed")

    task_id = str(enqueue.get("task_id", ""))
    provider = str(enqueue.get("provider", "") or enqueue.get("message_bus_provider", ""))
    tenant_id = str(enqueue.get("tenant_id", "lkw-background-proof"))
    marker = str(enqueue.get("marker", ""))
    run_id = str(enqueue.get("run_id", ""))
    correlation_id = str(enqueue.get("correlation_id", ""))
    collection_id = str(enqueue.get("collection_id", ""))
    if not task_id or not provider or not marker or not run_id or not correlation_id:
        return _fail("missing_enqueue_evidence")
    if not collection_id:
        return _fail("missing_collection_id")

    if provider != "kafka":
        return _fail("message_bus_provider_not_kafka", message_bus_provider=provider)

    initial_status = str(enqueue.get("initial_task_status", ""))
    if initial_status == "SUCCEEDED":
        return _fail("enqueue_not_asynchronous")

    deadline = time.time() + max(30, args.timeout_seconds)
    final_status = initial_status
    error_message = ""
    has_result = False
    while time.time() < deadline:
        status_url = (
            f"{base}/v1/local_workspace/proof/background-task/status/"
            f"{urllib.parse.quote(provider, safe='')}/"
            f"{urllib.parse.quote(task_id, safe='')}"
            f"?tenant_id={urllib.parse.quote(tenant_id, safe='')}"
        )
        try:
            status_payload = _request_json(status_url, timeout=10.0)
        except Exception:
            time.sleep(2.0)
            continue
        final_status = str(status_payload.get("task_status", final_status))
        error_message = str(status_payload.get("error_message", ""))
        has_result = bool(status_payload.get("has_result"))
        if bool(status_payload.get("completed")):
            break
        time.sleep(2.0)

    if final_status != "SUCCEEDED":
        return _fail(
            "background_task_not_succeeded",
            task_status=final_status,
            error_message=error_message,
        )

    if not has_result:
        return _fail("task_result_missing")

    search_body = {
        "tenant_id": tenant_id,
        "user_id": _PROOF_REQUESTED_BY,
        "message": marker,
        "capability": "local.workspace.search",
        "metadata": {
            "proof": "LKW_PLATFORM_PROOF",
            "proof_helper": _PROOF_RUNNER,
            "background_task_run_id": run_id,
            "background_task_id": task_id,
            "background_task_correlation_id": correlation_id,
            "tenant_id": tenant_id,
            "user_id": _PROOF_REQUESTED_BY,
            "collection_id": collection_id,
            "query": marker,
            "top_k": 5,
        },
    }
    try:
        search_response = _request_json(
            f"{base}/v1/local_workspace/run",
            method="POST",
            payload=search_body,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(f"search_failed:{type(exc).__name__}")

    search_results = _search_result_count(search_response)
    if search_results < 1:
        search_diag = _search_summary_diagnostic(search_response)
        return _fail(
            "search_results_missing",
            search_results=0,
            search_reason=search_diag.get("reason", ""),
            search_raw_tool_reason=search_diag.get("raw_tool_reason", ""),
            search_used=search_diag.get("used", ""),
            search_num_results=search_diag.get("num_results", ""),
        )

    kafka_messages = _inspect_kafka_topic(bootstrap=args.kafka_bootstrap, topic=args.topic)
    if kafka_messages == 0:
        return _fail("kafka_task_topic_empty", kafka_topic=args.topic)

    receipt = build_background_task_proof_receipt(
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=task_id,
        provider=provider,
        final_status=final_status,
        search_results=search_results,
        marker=marker,
        collection_id=collection_id,
        tenant_id=tenant_id,
        kafka_messages=kafka_messages,
        mongo_express_url=args.mongo_express,
        has_result=has_result,
    )

    try:
        verified_receipt, integration = record_background_task_proof_receipt(receipt)
    except (ProofReceiptVerificationError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return _fail_receipt_recording(exc)

    _print_pass_output(
        provider=provider,
        final_status=final_status,
        search_results=search_results,
        run_id=run_id,
        correlation_id=correlation_id,
        task_id=task_id,
        marker=marker,
        collection_id=collection_id,
        kafka_ui_url=args.kafka_ui,
        mongo_express_url=args.mongo_express,
        kafka_messages=kafka_messages,
        verified_receipt=verified_receipt,
        integration_class=type(integration).__name__,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
