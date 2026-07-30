# © Artur Czarnecki. All rights reserved.

"""Safe JSON and Markdown serialization for vLLM prefix-cache live proof."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from intergrax.llm_adapters.providers.vllm_diagnostics import VllmMetricDeltas
from intergrax.runtime.token_optimization.vllm_prefix_cache_proof import (
    VllmPrefixCacheProofCaseResult,
)

SCHEMA_VERSION = "token-optimization.vllm-prefix-cache-proof.v1"
TASK_ID = "TOKEN-10C-LIVE-PROOF-1"
JSON_FILENAME = "vllm-prefix-cache-proof.json"
MARKDOWN_FILENAME = "vllm-prefix-cache-proof.md"

_FORBIDDEN_REPORT_MARKERS: tuple[str, ...] = (
    "token-10c-",
    "Synthetic cache-stable qualification prefix",
    "token_optimization_proof_echo",
    "dynamic tail",
    "tool parameters",
    "model response content",
    "HUGGING_FACE_HUB_TOKEN",
    "Authorization",
)


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofConfiguration:
    requested_runs: int
    minimum_prefix_chars: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    startup_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofEnvironment:
    gpu_available: bool
    gpu_name: str | None
    gpu_memory_total_mb: int | None
    nvidia_driver_version: str | None
    docker_available: bool
    docker_compose_available: bool
    compose_contract_valid: bool
    vllm_image: str | None
    vllm_version: str | None
    model: str
    health_ok: bool
    managed_environment: bool
    force_recreated: bool
    exclusive_environment_expected: bool
    server_lifecycle_mode: str
    server_started_by_runner: bool
    environment_verified: bool
    proof_gates_passed: bool


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofRunResult:
    run_index: int
    passed: bool
    reason_codes: tuple[str, ...]
    server_version: str | None
    health_ok: bool
    cases: tuple[VllmPrefixCacheProofCaseResult, ...]


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofAggregateSummary:
    canonical_pass: bool
    all_runs_passed: bool
    requested_runs: int
    completed_runs: int
    passed_runs: int
    failed_runs: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofAggregateResult:
    schema_version: str
    task_id: str
    started_at_utc: str
    finished_at_utc: str
    repository_commit: str | None
    canonical_environment: bool
    environment: VllmPrefixCacheLiveProofEnvironment
    configuration: VllmPrefixCacheLiveProofConfiguration
    runs: tuple[VllmPrefixCacheLiveProofRunResult, ...]
    aggregate: VllmPrefixCacheLiveProofAggregateSummary
    exit_code: int
    json_report_path: Path | None = None
    markdown_report_path: Path | None = None


def metric_deltas_to_safe_dict(deltas: VllmMetricDeltas | None) -> dict[str, float | None] | None:
    if deltas is None:
        return None
    return {
        "prefix_cache_queries": deltas.prefix_cache_queries,
        "prefix_cache_hits": deltas.prefix_cache_hits,
        "prompt_tokens_cached": deltas.prompt_tokens_cached,
        "kv_cache_usage_perc": deltas.kv_cache_usage_perc,
        "request_prefill_kv_computed_tokens": deltas.request_prefill_kv_computed_tokens,
        "request_prefill_time_seconds": deltas.request_prefill_time_seconds,
        "time_to_first_token_seconds": deltas.time_to_first_token_seconds,
        "e2e_request_latency_seconds": deltas.e2e_request_latency_seconds,
    }


def case_result_to_safe_dict(case: VllmPrefixCacheProofCaseResult) -> dict[str, Any]:
    return {
        "case_id": case.case_id.value,
        "prefix_hash": case.prefix_hash,
        "tool_envelope_hash": case.tool_envelope_hash,
        "input_tokens": case.input_tokens,
        "cached_input_tokens": case.cached_input_tokens,
        "uncached_input_tokens": case.uncached_input_tokens,
        "prompt_tokens_details_reported": case.prompt_tokens_details_reported,
        "latency_ms": case.latency_ms,
        "metric_deltas": metric_deltas_to_safe_dict(case.metric_deltas),
        "passed": case.passed,
        "reason_codes": list(case.reason_codes),
    }


def run_result_to_safe_dict(run: VllmPrefixCacheLiveProofRunResult) -> dict[str, Any]:
    return {
        "run_index": run.run_index,
        "passed": run.passed,
        "reason_codes": list(run.reason_codes),
        "server_version": run.server_version,
        "health_ok": run.health_ok,
        "cases": [case_result_to_safe_dict(case) for case in run.cases],
    }


def environment_to_safe_dict(environment: VllmPrefixCacheLiveProofEnvironment) -> dict[str, Any]:
    return {
        "gpu_available": environment.gpu_available,
        "gpu_name": environment.gpu_name,
        "gpu_memory_total_mb": environment.gpu_memory_total_mb,
        "nvidia_driver_version": environment.nvidia_driver_version,
        "docker_available": environment.docker_available,
        "docker_compose_available": environment.docker_compose_available,
        "compose_contract_valid": environment.compose_contract_valid,
        "vllm_image": environment.vllm_image,
        "vllm_version": environment.vllm_version,
        "model": environment.model,
        "health_ok": environment.health_ok,
        "managed_environment": environment.managed_environment,
        "force_recreated": environment.force_recreated,
        "exclusive_environment_expected": environment.exclusive_environment_expected,
        "server_lifecycle_mode": environment.server_lifecycle_mode,
        "server_started_by_runner": environment.server_started_by_runner,
        "environment_verified": environment.environment_verified,
        "proof_gates_passed": environment.proof_gates_passed,
    }


def configuration_to_safe_dict(configuration: VllmPrefixCacheLiveProofConfiguration) -> dict[str, Any]:
    return {
        "requested_runs": configuration.requested_runs,
        "minimum_prefix_chars": configuration.minimum_prefix_chars,
        "connect_timeout_seconds": configuration.connect_timeout_seconds,
        "read_timeout_seconds": configuration.read_timeout_seconds,
        "startup_timeout_seconds": configuration.startup_timeout_seconds,
    }


def aggregate_summary_to_safe_dict(
    aggregate: VllmPrefixCacheLiveProofAggregateSummary,
) -> dict[str, Any]:
    return {
        "canonical_pass": aggregate.canonical_pass,
        "all_runs_passed": aggregate.all_runs_passed,
        "requested_runs": aggregate.requested_runs,
        "completed_runs": aggregate.completed_runs,
        "passed_runs": aggregate.passed_runs,
        "failed_runs": aggregate.failed_runs,
        "reason_codes": list(aggregate.reason_codes),
    }


def aggregate_result_to_safe_dict(result: VllmPrefixCacheLiveProofAggregateResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "task_id": result.task_id,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "repository_commit": result.repository_commit,
        "canonical_environment": result.canonical_environment,
        "environment": environment_to_safe_dict(result.environment),
        "configuration": configuration_to_safe_dict(result.configuration),
        "runs": [run_result_to_safe_dict(run) for run in result.runs],
        "aggregate": aggregate_summary_to_safe_dict(result.aggregate),
    }


def validate_safe_report_text(text: str) -> None:
    lowered = text.lower()
    scrubbed = lowered.replace(TASK_ID.lower(), "")
    for marker in _FORBIDDEN_REPORT_MARKERS:
        if marker.lower() in scrubbed:
            raise ValueError(f"forbidden report marker detected: {marker}")


def serialize_safe_json(result: VllmPrefixCacheLiveProofAggregateResult) -> str:
    payload = aggregate_result_to_safe_dict(result)
    content = json.dumps(payload, indent=2, sort_keys=True)
    validate_safe_report_text(content)
    return content


def render_markdown_report(result: VllmPrefixCacheLiveProofAggregateResult) -> str:
    env = result.environment
    aggregate = result.aggregate
    lines = [
        "# vLLM Prefix Cache Live Proof",
        "",
        f"**Task:** {result.task_id}",
        f"**Date/time (UTC):** {result.finished_at_utc}",
        f"**Repository commit:** {result.repository_commit or 'unknown'}",
        "",
        "## Environment summary",
        "",
        f"- Canonical environment: **{'yes' if result.canonical_environment else 'no'}**",
        f"- Server lifecycle mode: **{env.server_lifecycle_mode}**",
        f"- Server started by runner: {env.server_started_by_runner}",
        f"- Environment verified: **{'yes' if env.environment_verified else 'no'}**",
        f"- Proof gates passed: **{'yes' if env.proof_gates_passed else 'no'}**",
        f"- Managed environment: {env.managed_environment}",
        f"- Force recreated: {env.force_recreated}",
        f"- Exclusive environment expected: {env.exclusive_environment_expected}",
        f"- vLLM image: {env.vllm_image or 'unknown'}",
        f"- vLLM version: {env.vllm_version or 'unknown'}",
        f"- Model: {env.model}",
        f"- Health OK: {env.health_ok}",
        "",
        "## GPU summary",
        "",
        f"- GPU available: {env.gpu_available}",
        f"- GPU model: {env.gpu_name or 'n/a'}",
        f"- GPU memory total (MB): {env.gpu_memory_total_mb if env.gpu_memory_total_mb is not None else 'n/a'}",
        f"- NVIDIA driver: {env.nvidia_driver_version or 'n/a'}",
        "",
        "## Run-by-run summary",
        "",
        "| Run | Passed | Reason codes |",
        "| --- | --- | --- |",
    ]
    for run in result.runs:
        reason_text = ", ".join(run.reason_codes) if run.reason_codes else "-"
        lines.append(f"| {run.run_index} | {'PASS' if run.passed else 'FAIL'} | {reason_text} |")

    lines.extend(
        [
            "",
            "## Case-by-case cached-token evidence",
            "",
            "| Run | Case | Prefix hash | Cached tokens | Cache-hit delta | Passed |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in result.runs:
        for case in run.cases:
            hit_delta = (
                case.metric_deltas.prefix_cache_hits
                if case.metric_deltas is not None
                else None
            )
            hit_text = f"{hit_delta:.3f}" if hit_delta is not None else "n/a"
            lines.append(
                f"| {run.run_index} | {case.case_id.value} | {case.prefix_hash} | "
                f"{case.cached_input_tokens} | {hit_text} | {'PASS' if case.passed else 'FAIL'} |"
            )

    lines.extend(
        [
            "",
            "## Metric deltas (supporting evidence)",
            "",
            "| Run | Case | Queries | Hits | Prompt cached | KV usage |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in result.runs:
        for case in run.cases:
            deltas = case.metric_deltas
            if deltas is None:
                lines.append(
                    f"| {run.run_index} | {case.case_id.value} | n/a | n/a | n/a | n/a |"
                )
                continue
            lines.append(
                f"| {run.run_index} | {case.case_id.value} | {deltas.prefix_cache_queries:.3f} | "
                f"{deltas.prefix_cache_hits:.3f} | {deltas.prompt_tokens_cached:.3f} | "
                f"{deltas.kv_cache_usage_perc:.5f} |"
            )

    lines.extend(
        [
            "",
            "## Reason codes",
            "",
        ]
    )
    if aggregate.reason_codes:
        for code in aggregate.reason_codes:
            lines.append(f"- {code}")
    else:
        lines.append("- none")

    final_status = "PASS" if aggregate.canonical_pass else "FAIL"
    lines.extend(
        [
            "",
            f"## Final result: **{final_status}**",
            "",
            f"- Canonical pass: {aggregate.canonical_pass}",
            f"- All runs passed: {aggregate.all_runs_passed}",
            f"- Passed runs: {aggregate.passed_runs}/{aggregate.requested_runs}",
            f"- Process exit code: {result.exit_code}",
            "",
            "## Known limitations",
            "",
            "- Latency is supporting evidence only and is not a hard PASS gate.",
            "- Managed/shared lifecycle describes server ownership, not proof correctness.",
            "- A manually started vLLM server can produce a canonical PASS when the runner independently verifies the expected model, pinned vLLM version, required metrics, and all cold/warm/changed-prefix gates.",
            "- Common chat-template blocks may still be reused on changed-prefix cases.",
        ]
    )
    content = "\n".join(lines) + "\n"
    validate_safe_report_text(content)
    return content


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def write_proof_artifacts(
    result: VllmPrefixCacheLiveProofAggregateResult,
    *,
    output_dir: Path,
    timestamp_utc: str,
) -> tuple[Path, Path]:
    report_dir = output_dir / timestamp_utc
    json_path = report_dir / JSON_FILENAME
    markdown_path = report_dir / MARKDOWN_FILENAME
    atomic_write_text(json_path, serialize_safe_json(result))
    atomic_write_text(markdown_path, render_markdown_report(result))
    return json_path, markdown_path


def dedupe_reason_codes(codes: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(code for code in codes if code))
