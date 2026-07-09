#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background-task platform proof helper (LKW.4E)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


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
        default=os.environ.get("LKW_BACKGROUND_TASK_PROOF_KAFKA_UI_URL", "http://127.0.0.1:8088"),
        help="Kafka UI URL for reviewer hints (default: http://127.0.0.1:8088).",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default=os.environ.get("LKW_BACKGROUND_TASK_PROOF_KAFKA_BOOTSTRAP", "127.0.0.1:9094"),
        help="Host-visible Kafka bootstrap server (default: 127.0.0.1:9094).",
    )
    parser.add_argument(
        "--topic",
        default=os.environ.get("INTERGRAX_KAFKA_TOPIC", "intergrax-lkw-tasks"),
        help="Kafka task topic (default: intergrax-lkw-tasks).",
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


def _search_result_count(response: dict[str, object]) -> int:
    metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return 0
    evidence = metadata.get("lkw_evidence.v1")
    if not isinstance(evidence, dict):
        return 0
    search_summary = evidence.get("lkw.search_summary.v1")
    if not isinstance(search_summary, dict):
        return 0
    for field in ("result_count", "results_count", "hits"):
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


def main() -> int:
    args = _parse_args()
    base = args.base_url.rstrip("/")
    if not _health_ok(base):
        print("proof_result=FAIL")
        print("reason=lkw_health_unreachable")
        return 1

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
        print("proof_result=FAIL")
        print(f"reason=http_{exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print("proof_result=FAIL")
        print(f"reason={type(exc).__name__}")
        return 1

    if str(enqueue.get("proof_result", "FAIL")) != "PASS":
        print("proof_result=FAIL")
        print("reason=enqueue_proof_failed")
        return 1

    task_id = str(enqueue.get("task_id", ""))
    provider = str(enqueue.get("provider", ""))
    tenant_id = str(enqueue.get("tenant_id", "lkw-background-proof"))
    marker = str(enqueue.get("marker", ""))
    run_id = str(enqueue.get("run_id", ""))
    if not task_id or not provider or not marker:
        print("proof_result=FAIL")
        print("reason=missing_enqueue_evidence")
        return 1

    initial_status = str(enqueue.get("initial_task_status", ""))
    if initial_status == "SUCCEEDED":
        print("proof_result=FAIL")
        print("reason=enqueue_not_asynchronous")
        return 1

    deadline = time.time() + max(30, args.timeout_seconds)
    final_status = initial_status
    error_message = ""
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
        if bool(status_payload.get("completed")):
            break
        time.sleep(2.0)

    if final_status != "SUCCEEDED":
        print("proof_result=FAIL")
        print("reason=background_task_not_succeeded")
        print(f"task_status={final_status}")
        if error_message:
            print(f"error_message={error_message}")
        return 1

    search_body = {
        "tenant_id": tenant_id,
        "message": marker,
        "capability": "local.workspace.search",
        "metadata": {
            "proof": "LKW_PLATFORM_PROOF",
            "proof_helper": "run-lkw-background-task-proof.py",
            "background_task_run_id": run_id,
            "background_task_id": task_id,
        },
    }
    try:
        search_response = _request_json(
            f"{base}/v1/local_workspace/run",
            method="POST",
            payload=search_body,
        )
    except Exception as exc:  # noqa: BLE001
        print("proof_result=FAIL")
        print(f"reason=search_failed:{type(exc).__name__}")
        return 1

    search_results = _search_result_count(search_response)
    if search_results < 1:
        print("proof_result=FAIL")
        print("reason=search_results_missing")
        print("search_results=0")
        return 1

    kafka_messages = _inspect_kafka_topic(bootstrap=args.kafka_bootstrap, topic=args.topic)

    print("proof_result=PASS")
    print("proof_kind=platform_background_task")
    print(f"task_name=lkw.background_ingest.v1")
    print(f"message_bus_provider={provider}")
    print("enqueue_mode=real_provider")
    print("worker_execution=asynchronous")
    print(f"task_status={final_status}")
    print(f"search_results={search_results}")
    print("mock_queue=false")
    print(f"run_id={run_id}")
    print(f"task_id={task_id}")
    print(f"marker={marker}")
    print(f"kafka_ui={args.kafka_ui}")
    print(f"kafka_topic={args.topic}")
    if kafka_messages >= 0:
        print(f"kafka_topic_messages={kafka_messages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
