# © Artur Czarnecki. All rights reserved.

"""Markdown rendering and JSON serialization for qualification results."""

from __future__ import annotations

import json
import re
from typing import Iterable

from local_workspace_application.benchmarks.local_model_qualification.contracts import (
    LocalModelQualificationResult,
    ProtocolResult,
    ProtocolStatus,
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


def _protocol_row(protocol: ProtocolResult) -> str:
    return (
        f"| {protocol.protocol} | "
        f"{', '.join(protocol.declared_capabilities) if hasattr(protocol, 'declared_capabilities') else 'n/a'} | "
        f"{protocol.schema_probe_status.value} | "
        f"{protocol.case_count} | "
        f"{protocol.semantic_success_rate:.1%} | "
        f"{protocol.invalid_draft_count} | "
        f"{protocol.provider_failure_count} | "
        f"{protocol.unsafe_state_change_count} | "
        f"{protocol.latency_ms.median:.0f} | "
        f"{protocol.latency_ms.p95:.0f} | "
        f"{protocol.qualification_status.value} |"
    )


def render_markdown(result: LocalModelQualificationResult) -> str:
    lines: list[str] = [
        _GENERATED_WARNING.strip(),
        "",
        "# LKW Local Ollama Model Qualification",
        "",
        "## 1. Scope and interpretation",
        "",
        "This is an LKW-specific benchmark for conversational interaction planning. "
        "It is not a universal LLM ranking. Results apply to the exact model digests "
        "and Ollama version shown below.",
        "",
        "## 2. Executive summary",
        "",
        result.summary.message,
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
            "- Tool protocol adds only this transport instruction after the production system message:",
            "  Call submit_conversation_interaction_draft exactly once with the complete semantic draft "
            "as its arguments. This submission tool does not execute any operation. "
            "Do not answer in plain text and do not call any other tool.",
            "",
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
            "| Model | Role | Installed | Digest |",
            "| --- | --- | --- | --- |",
        ]
    )
    for model in result.models:
        digest = model.metadata.digest or "n/a"
        lines.append(f"| {model.name} | {model.role} | {model.installed} | `{digest}` |")
    lines.extend(
        [
            "",
            "## 8. Model × protocol comparison",
            "",
            "| Model | Role | Protocol | Capabilities | Schema probe | Samples | Semantic success | "
            "Invalid drafts | Provider failures | Unsafe state changes | Median latency | p95 latency | "
            "Execution mode | Qualification |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for model in result.models:
        for protocol in model.protocols:
            capabilities = ", ".join(model.declared_capabilities) if model.declared_capabilities else "n/a"
            lines.append(
                f"| {model.name} | {model.role} | {protocol.protocol} | {capabilities} | "
                f"{protocol.schema_probe_status.value} | {protocol.case_count} | "
                f"{protocol.semantic_success_rate:.1%} | {protocol.invalid_draft_count} | "
                f"{protocol.provider_failure_count} | {protocol.unsafe_state_change_count} | "
                f"{protocol.latency_ms.median:.0f} | {protocol.latency_ms.p95:.0f} | "
                f"{model.observed_execution_mode.value} | {protocol.qualification_status.value} |"
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
        lines.extend(
            [
                f"### {model.name}",
                "",
                f"- Digest: `{model.metadata.digest or 'n/a'}`",
                f"- Artifact size: {_format_bytes(model.metadata.artifact_size_bytes)}",
                f"- Parameter size: {model.metadata.parameter_size or 'n/a'}",
                f"- Quantization: {model.metadata.quantization_level or 'n/a'}",
                f"- Declared capabilities: {', '.join(model.declared_capabilities) or 'n/a'}",
                f"- Observed VRAM allocation: {_format_bytes(model.metadata.size_vram_bytes)}",
                f"- Observed offload mode: {model.observed_execution_mode.value}",
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
            "- Results are tied to observed hardware and installed model digests.",
            "- `single_plan_tool` is experimental and not used in production.",
            "- No universal minimum hardware requirements are implied.",
            "",
        ]
    )
    return "\n".join(lines)


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


def format_conditional_candidates(candidates: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(candidates))
