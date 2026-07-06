#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Controlled LKW Sentry observability proof helper (LKW-OBS-SENTRY-1)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call the LKW local Sentry proof endpoint over HTTP.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOCAL_WORKSPACE_BACKEND_BASE_URL", "http://127.0.0.1:8020"),
        help="LKW backend base URL (default: http://127.0.0.1:8020).",
    )
    parser.add_argument(
        "--sentry-ui",
        default=os.environ.get("LKW_SENTRY_PROOF_UI_URL", "http://127.0.0.1:9000"),
        help="Local Sentry UI URL for operator hints (default: http://127.0.0.1:9000).",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id for the proof event (generated when omitted).",
    )
    parser.add_argument(
        "--correlation-id",
        default="",
        help="Optional correlation id for the proof event (generated when omitted).",
    )
    return parser.parse_args()


def _post_json(url: str, payload: dict[str, str]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("proof endpoint returned non-object JSON")
    return parsed


def main() -> int:
    args = _parse_args()
    endpoint = f"{args.base_url.rstrip('/')}/v1/local_workspace/proof/sentry-error"
    payload: dict[str, str] = {}
    if args.run_id.strip():
        payload["run_id"] = args.run_id.strip()
    if args.correlation_id.strip():
        payload["correlation_id"] = args.correlation_id.strip()

    try:
        result = _post_json(endpoint, payload)
    except urllib.error.HTTPError as exc:
        print("proof_result=FAIL")
        print(f"reason=http_{exc.code}")
        return 1
    except urllib.error.URLError:
        print("proof_result=FAIL")
        print("reason=backend_unreachable")
        return 1
    except ValueError as exc:
        print("proof_result=FAIL")
        print(f"reason={exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - proof helper surfaces operator-safe failure
        print("proof_result=FAIL")
        print(f"reason={type(exc).__name__}")
        return 1

    proof_result = str(result.get("proof_result", "FAIL"))
    run_id = str(result.get("run_id", ""))
    correlation_id = str(result.get("correlation_id", ""))
    safety_check = str(result.get("safety_check", "failed"))

    print(f"proof_result={proof_result}")
    print("backend=sentry")
    print("sentry_mode=local_docker")
    print(f"sentry_ui={args.sentry_ui}")
    print("problem_kind=lkw.proof_controlled_failure")
    print("problem_error_code=LKW_PROOF_CONTROLLED_FAILURE")
    if run_id:
        print(f"run_id={run_id}")
    if correlation_id:
        print(f"correlation_id={correlation_id}")
    print(f"safety_check={safety_check}")
    if proof_result == "PASS":
        print("sentry_event_sent=true")
        print("sentry_search_hint=tag:intergrax.problem_kind:lkw.proof_controlled_failure")
    return 0 if proof_result == "PASS" and safety_check == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
