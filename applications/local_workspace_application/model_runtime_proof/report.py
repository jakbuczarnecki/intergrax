# © Artur Czarnecki. All rights reserved.

"""Markdown and JSON evidence rendering for model runtime portability proof."""

from __future__ import annotations

import json
from pathlib import Path

from local_workspace_application.model_runtime_proof.contracts import (
    ModelRuntimeProofResult,
    ProviderQualificationResult,
    StageStatus,
)


def serialize_result_json(result: ModelRuntimeProofResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)


def _stage_line(label: str, status: StageStatus) -> str:
    return f"  {label}: {status.value}"


def _provider_block(name: str, provider: ProviderQualificationResult) -> list[str]:
    stages = provider.stages
    return [
        f"{name.upper()}:",
        _stage_line("health", stages.health),
        _stage_line("generation", stages.generation),
        _stage_line("structured_plan", stages.structured_plan),
        _stage_line("tool_call", stages.tool_call),
        _stage_line("tool_execution", stages.tool_execution),
        _stage_line("grounded_ask", stages.grounded_ask),
        "",
    ]


def render_terminal_summary(result: ModelRuntimeProofResult) -> str:
    lines: list[str] = []
    for provider_name in ("ollama", "vllm"):
        provider = result.provider_results.get(provider_name)
        if provider is not None:
            lines.extend(_provider_block(provider_name, provider))
    inv = result.index_invariance
    lines.extend(
        [
            "INDEX INVARIANCE:",
            _stage_line("embedding_identity", inv.embedding_identity),
            _stage_line("collection_identity", inv.collection_identity),
            _stage_line("vector_count", inv.vector_count),
            _stage_line("no_reindex", inv.no_reindex),
            "",
            "OVERALL:",
            f"  {result.overall_status.value}",
        ]
    )
    return "\n".join(lines)


def render_markdown(result: ModelRuntimeProofResult) -> str:
    lines = [
        "# LKW Model Runtime Portability Proof",
        "",
        f"- schema: `{result.schema_version}`",
        f"- proof_id: `{result.proof_id}`",
        f"- classification: {result.proof_classification}",
        f"- overall: **{result.overall_status.value}**",
        f"- repository_commit: `{result.repository_commit or 'unknown'}`",
        "",
        "## Qualified provider pairs",
        "",
    ]
    for provider in result.provider_results.values():
        lines.extend(
            [
                f"### {provider.provider}",
                f"- configured_model: `{provider.configured_model}`",
                f"- resolved_model: `{provider.resolved_model}`",
                f"- server_model: `{provider.server_model}`",
                f"- adapter_class: `{provider.adapter_class}`",
                f"- server_version: `{provider.server_version}`",
                f"- failure_code: `{provider.failure_code}`",
                "",
            ]
        )
    if result.index_identity is not None:
        identity = result.index_identity
        lines.extend(
            [
                "## Shared index",
                "",
                f"- workspace_id: `{identity.workspace_id}`",
                f"- source_id: `{identity.source_id}`",
                f"- document_id: `{identity.document_id}`",
                f"- vector_count: `{identity.vector_count}`",
                f"- embedding_provider: `{identity.embedding.provider}`",
                f"- embedding_model: `{identity.embedding.model}`",
                "",
            ]
        )
    if result.limitations:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {item}" for item in result.limitations)
    return "\n".join(lines)


def write_evidence(
    result: ModelRuntimeProofResult,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    if result.overall_status.value != "PASS":
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(serialize_result_json(result), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
