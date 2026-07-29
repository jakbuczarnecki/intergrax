# © Artur Czarnecki. All rights reserved.

"""Markdown rendering and JSON serialization for qualification results."""

from __future__ import annotations

import json
import re
from functools import cmp_to_key
from typing import Iterable

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    LocalModelQualificationResult,
    ObservedExecutionMode,
    ProtocolResult,
    ProtocolStatus,
    ProvisioningResult,
    WarmupStatus,
)

_GENERATED_WARNING = """<!--
GENERATED FILE.
Do not edit benchmark results manually.
Regenerate with:
uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py
-->
"""

_SECRET_PATTERNS = (
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"bearer\s", re.IGNORECASE),
    re.compile(r"api_key", re.IGNORECASE),
    re.compile(r"password=", re.IGNORECASE),
    re.compile(r"mongodb://", re.IGNORECASE),
    re.compile(r"mongodb\+srv://", re.IGNORECASE),
)


def serialize_result_json(result: LocalModelQualificationResult) -> str:
    payload = result.to_json_dict()
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def contains_secret_patterns(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def qualification_rank(status: ProtocolStatus) -> int:
    order = {
        ProtocolStatus.QUALIFIED: 0,
        ProtocolStatus.CONDITIONALLY_QUALIFIED: 1,
        ProtocolStatus.NOT_QUALIFIED: 2,
        ProtocolStatus.PROTOCOL_UNSUPPORTED: 3,
        ProtocolStatus.SCHEMA_INCOMPATIBLE: 4,
        ProtocolStatus.WARMUP_FAILED: 5,
        ProtocolStatus.PROVIDER_ERROR: 6,
        ProtocolStatus.RESOURCE_LIMIT: 7,
        ProtocolStatus.NOT_RUN: 8,
    }
    return order.get(status, 99)


def _compare_protocol_metrics(left: ProtocolResult, right: ProtocolResult) -> int:
    for value in (
        qualification_rank(left.qualification_status) - qualification_rank(right.qualification_status),
        left.unsafe_state_change_count - right.unsafe_state_change_count,
        left.invalid_draft_count - right.invalid_draft_count,
        -1 if left.semantic_success_rate > right.semantic_success_rate else (
            1 if left.semantic_success_rate < right.semantic_success_rate else 0
        ),
        left.provider_failure_count - right.provider_failure_count,
        -1 if left.latency_ms.median < right.latency_ms.median else (
            1 if left.latency_ms.median > right.latency_ms.median else 0
        ),
    ):
        if value != 0:
            return value
    return 0


def compare_candidates(
    left: tuple[str, ProtocolResult],
    right: tuple[str, ProtocolResult],
) -> int:
    left_model, left_proto = left
    right_model, right_proto = right
    metrics = _compare_protocol_metrics(left_proto, right_proto)
    if metrics != 0:
        return metrics
    if left_model < right_model:
        return -1
    if left_model > right_model:
        return 1
    if left_proto.protocol < right_proto.protocol:
        return -1
    if left_proto.protocol > right_proto.protocol:
        return 1
    return 0


def _digest_claim(result: LocalModelQualificationResult) -> str:
    all_digests_available = all(
        model.metadata.digest
        for model in result.models
    )
    if all_digests_available and result.models:
        return (
            "Results apply to the exact model tags, digests, Ollama version, "
            "configuration and benchmark host shown below."
        )
    return (
        "One or more model digests were unavailable. Reproducibility is limited "
        "to the recorded model tags, Ollama version, configuration and host metadata."
    )


def _probe_diagnostics(protocol: ProtocolResult) -> tuple[str, str, str, str]:
    if protocol.schema_probe_status.value == "PASS":
        return "PASS", "n/a", "n/a", "n/a"
    return (
        protocol.schema_probe_status.value,
        protocol.probe_failure_category or "n/a",
        protocol.probe_failure_phase or "n/a",
        protocol.probe_safe_error_code or "n/a",
    )


def _warmup_diagnostics(protocol: ProtocolResult) -> tuple[str, str, str, str, str]:
    if protocol.warmup_status != WarmupStatus.FAILED:
        return "n/a", "n/a", "n/a", "n/a", "n/a"
    return (
        protocol.warmup_failure_category or "n/a",
        protocol.warmup_failure_phase or "n/a",
        protocol.warmup_safe_error_code or "n/a",
        str(protocol.warmup_failure_repetition) if protocol.warmup_failure_repetition is not None else "n/a",
        f"{protocol.warmup_failure_latency_ms:.1f}"
        if protocol.warmup_failure_latency_ms is not None
        else "n/a",
    )


def _render_provisioning_section(provisioning: ProvisioningResult, result: LocalModelQualificationResult) -> list[str]:
    inventory_by_model = {model.name: model for model in result.models}
    lines = [
        "## Docker Ollama provisioning",
        "",
        f"- Runtime: `{provisioning.runtime}`",
        f"- Compose file: `{provisioning.compose_file}`",
        f"- Compose service: `{provisioning.compose_service}`",
        f"- Container name: `{provisioning.container_name}`",
        f"- Persistent model volume: `{provisioning.persistent_model_volume}`",
        f"- Required model count: {len(provisioning.required_models)}",
        f"- Docker Ollama readiness: {provisioning.readiness_result}",
        "",
        "| Model | Provisioning status | Digest | Artifact size |",
        "| --- | --- | --- | --- |",
    ]
    for provisioned in provisioning.models:
        model = inventory_by_model.get(provisioned.model)
        digest = model.metadata.digest if model and model.metadata.digest else "n/a"
        size = _format_bytes(model.metadata.artifact_size_bytes) if model else "n/a"
        lines.append(
            f"| {provisioned.model} | {provisioned.status.value} | `{digest}` | {size} |"
        )
    lines.append("")
    return lines


def render_markdown(result: LocalModelQualificationResult) -> str:
    lines: list[str] = [
        _GENERATED_WARNING.strip(),
        "",
        "# LKW Local Ollama Model Qualification",
        "",
        "## 1. Scope and interpretation",
        "",
        "This is an LKW-specific benchmark for conversational interaction planning. "
        "It is not a universal LLM ranking. "
        f"{_digest_claim(result)}",
        "",
        "## 2. Executive summary",
        "",
        result.summary.message,
        "",
        f"- Required models: {result.summary.required_model_count}",
        f"- Provisioned models: {result.summary.provisioned_model_count}",
        f"- Expected model/protocol pairs: {result.summary.expected_model_protocol_pairs}",
        f"- Attempted model/protocol pairs: {result.summary.attempted_model_protocol_pairs}",
        f"- Expected scored calls: {result.summary.expected_scored_call_count}",
        f"- Actual scored calls: {result.summary.actual_scored_call_count}",
        "",
        "## 3. Recommended configuration",
        "",
    ]
    if result.summary.recommended_model and result.summary.recommended_protocol:
        lines.extend(
            [
                f"- Model: `{result.summary.recommended_model}`",
                f"- Protocol: `{result.summary.recommended_protocol}`",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "No tested model/protocol pair met the full LKW qualification threshold.",
                "",
            ]
        )
    if result.summary.conditional_candidates:
        lines.append("Experimental conditional candidates:")
        for candidate in result.summary.conditional_candidates:
            lines.append(f"- {candidate}")
        lines.append("")

    lines.extend(
        [
            "## 4. Benchmark methodology",
            "",
            "- Corpus version: "
            f"`{result.corpus_version}`",
            "- Production semantic prompt via `build_planning_messages()`",
            "- Structured output remains the current production transport",
            "- `single_plan_tool` is a benchmark candidate only",
            "- Submission tool `submit_conversation_interaction_draft` does not execute operations",
            "- Repair is disabled in benchmark scoring (`repair_attempts=0`)",
            "- Tool protocol uses `tool_choice=auto` with deterministic exactly-one-call "
            "enforcement by the benchmark harness",
            "- Tool protocol adds only this transport instruction after the production system message:",
            "  Call submit_conversation_interaction_draft exactly once with the complete semantic draft "
            "as its arguments. This submission tool does not execute any operation. "
            "Do not answer in plain text and do not call any other tool.",
            "",
            *_render_provisioning_section(result.provisioning, result),
            "## 5. Benchmark host",
            "",
            f"- OS: {result.host.operating_system} {result.host.os_release}",
            f"- Architecture: {result.host.machine_architecture}",
            f"- Python: {result.host.python_version}",
            f"- CPU: {result.host.cpu_description or 'n/a'}",
            f"- RAM: {_format_bytes(result.host.total_system_ram_bytes)}",
            f"- GPU: {result.host.gpu_name or 'n/a'}",
            f"- GPU VRAM: {_format_bytes(result.host.gpu_total_vram_bytes)}",
            f"- NVIDIA driver: {result.host.nvidia_driver_version or 'n/a'}",
            "",
            "Hardware figures are observed on this benchmark host, not universal minimum requirements.",
            "",
            "## 6. Ollama environment",
            "",
            f"- Host: `{result.ollama.host}`",
            f"- Version: {result.ollama.version or 'n/a'}",
            "",
            "## 7. Tested model inventory",
            "",
            "| Model | Role | Installed | Digest | Artifact size |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for model in result.models:
        digest = model.metadata.digest or "n/a"
        lines.append(
            f"| {model.name} | {model.role} | {model.installed} | `{digest}` | "
            f"{_format_bytes(model.metadata.artifact_size_bytes)} |"
        )
    lines.extend(
        [
            "",
            "## 8. Model × protocol comparison",
            "",
            "| Model | Role | Protocol | Capabilities | Schema probe | Probe failure category | "
            "Probe phase | Safe error code | Warmup status | Samples | Semantic success | "
            "Invalid drafts | Provider failures | Unsafe state changes | Median latency | p95 latency | "
            "Execution mode | Qualification |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for model in result.models:
        for protocol in model.protocols:
            capabilities = ", ".join(model.declared_capabilities) if model.declared_capabilities else "n/a"
            probe_status, failure_category, phase, safe_code = _probe_diagnostics(protocol)
            execution_mode = model.observed_execution_mode.value
            if model.observed_execution_mode == ObservedExecutionMode.FULL_GPU:
                execution_mode = "FULL_GPU (measured)"
            lines.append(
                f"| {model.name} | {model.role} | {protocol.protocol} | {capabilities} | "
                f"{probe_status} | {failure_category} | {phase} | {safe_code} | "
                f"{protocol.warmup_status.value} | "
                f"{protocol.case_count} | {protocol.semantic_success_rate:.1%} | "
                f"{protocol.invalid_draft_count} | {protocol.provider_failure_count} | "
                f"{protocol.unsafe_state_change_count} | {protocol.latency_ms.median:.0f} | "
                f"{protocol.latency_ms.p95:.0f} | {execution_mode} | "
                f"{protocol.qualification_status.value} |"
            )

    lines.extend(["", "## 9. Safety and state-change results", ""])
    for model in result.models:
        for protocol in model.protocols:
            lines.append(
                f"- {model.name} / {protocol.protocol}: unsafe state changes = "
                f"{protocol.unsafe_state_change_count}"
            )

    lines.extend(["", "## 10. Failure categories", ""])
    for model in result.models:
        for protocol in model.protocols:
            if not protocol.failure_category_counts:
                continue
            lines.append(f"### {model.name} / {protocol.protocol}")
            for category, count in sorted(protocol.failure_category_counts.items()):
                lines.append(f"- {category}: {count}")
            lines.append("")

    lines.extend(["", "## 11. Per-model details", ""])
    for model in result.models:
        offload_mode = model.observed_execution_mode.value
        if model.observed_execution_mode == ObservedExecutionMode.FULL_GPU:
            offload_mode = "FULL_GPU (measured from Client.ps())"
        elif model.observed_execution_mode == ObservedExecutionMode.UNKNOWN:
            offload_mode = "UNKNOWN (runtime metadata unavailable)"
        lines.extend(
            [
                f"### {model.name}",
                "",
                f"- Digest: `{model.metadata.digest or 'n/a'}`",
                f"- Artifact size: {_format_bytes(model.metadata.artifact_size_bytes)}",
                f"- Parameter size: {model.metadata.parameter_size or 'n/a'}",
                f"- Quantization: {model.metadata.quantization_level or 'n/a'}",
                f"- Declared capabilities: {', '.join(model.declared_capabilities) or 'n/a'}",
                f"- Observed loaded size: {_format_bytes(model.metadata.loaded_size_bytes)}",
                f"- Observed VRAM allocation: {_format_bytes(model.metadata.size_vram_bytes)}",
                f"- Observed offload mode: {offload_mode}",
                "",
            ]
        )
        for protocol in model.protocols:
            failed_cases = sorted(
                {
                    case.case_id
                    for case in protocol.case_results
                    if case.status.value != "PASS"
                }
            )
            top_failures = sorted(
                protocol.failure_category_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:5]
            lines.append(f"#### Protocol: {protocol.protocol}")
            lines.append(f"- Qualification: {protocol.qualification_status.value}")
            probe_status, probe_category, probe_phase, probe_safe_code = _probe_diagnostics(protocol)
            warmup_category, warmup_phase, warmup_safe_code, warmup_repetition, warmup_latency = (
                _warmup_diagnostics(protocol)
            )
            lines.append(f"- Probe status: {probe_status}")
            lines.append(f"- Probe failure category: {probe_category}")
            lines.append(f"- Probe failure phase: {probe_phase}")
            lines.append(f"- Probe safe error code: {probe_safe_code}")
            lines.append(f"- Warmup status: {protocol.warmup_status.value}")
            lines.append(f"- Warmup failure category: {warmup_category}")
            lines.append(f"- Warmup failure phase: {warmup_phase}")
            lines.append(f"- Warmup safe error code: {warmup_safe_code}")
            lines.append(f"- Warmup failure repetition: {warmup_repetition}")
            lines.append(f"- Warmup failure latency: {warmup_latency}")
            if top_failures:
                lines.append("- Top failure categories:")
                for category, count in top_failures:
                    lines.append(f"  - {category}: {count}")
            if failed_cases:
                lines.append(f"- Failed case IDs: {', '.join(failed_cases)}")
            lines.append("")

    lines.extend(
        [
            "## 12. Reproduction",
            "",
            "```powershell",
            "uv run python applications/local_workspace_application/scripts/run-local-model-qualification.py",
            "```",
            "",
            f"- Generated at (UTC): {result.generated_at_utc}",
            f"- Commit: {result.generated_from_commit or 'n/a'}",
            f"- Configuration SHA-256: `{result.configuration_sha256}`",
            "",
            "## 13. Limitations",
            "",
            "- This benchmark measures LKW conversational planning semantics only.",
            "- Results are tied to observed hardware and installed model digests when available.",
            "- `single_plan_tool` is experimental and not used in production.",
            "- No universal minimum hardware requirements are implied.",
            "",
        ]
    )
    return "\n".join(lines)


def format_conditional_candidates(candidates: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(candidates))


def sort_qualified_candidates(
    candidates: list[tuple[str, ProtocolResult]],
) -> list[tuple[str, ProtocolResult]]:
    return sorted(candidates, key=cmp_to_key(compare_candidates))
