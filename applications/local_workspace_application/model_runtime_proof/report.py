# © Artur Czarnecki. All rights reserved.

"""Markdown and JSON evidence rendering for model runtime portability proof."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from local_workspace_application.model_runtime_proof.contracts import (
    PROOF_SCHEMA_VERSION,
    ModelRuntimeProofResult,
    ProofOverallStatus,
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
        _stage_line("citation", stages.citation),
        f"  canonical_resolver: {provider.resolved_through_canonical_resolver}",
        f"  http_ask_status: {provider.http_ask_status_code}",
        f"  ask_persisted: {provider.ask_run_persisted}",
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
            _stage_line("source_identity", inv.source_identity),
            _stage_line("document_identity", inv.document_identity),
            _stage_line("content_hash", inv.content_hash),
            _stage_line("chunk_count", inv.chunk_count),
            _stage_line("no_reindex", inv.no_reindex),
            "",
            "REPOSITORY STATE:",
            f"  head: {result.repository_state.repository_head_at_proof}",
            f"  classification: {result.repository_state.working_tree_classification}",
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
        f"- repository_head_at_proof: `{result.repository_state.repository_head_at_proof or 'unknown'}`",
        f"- repository_head_role: `{result.repository_state.repository_head_role}`",
        f"- working_tree_classification: `{result.repository_state.working_tree_classification}`",
        f"- vllm_provisioning: `{result.vllm_provisioning_classification or 'unknown'}`",
        "",
        "## Qualified provider pairs",
        "",
    ]
    for provider in result.provider_results.values():
        provider_lines = [
            f"### {provider.provider}",
            f"- configured_model: `{provider.configured_model}`",
            f"- resolved_model: `{provider.resolved_model}`",
            f"- server_model: `{provider.server_model}`",
            f"- adapter_class: `{provider.adapter_class}`",
            f"- server_version: `{provider.server_version}`",
            f"- canonical_resolver: `{provider.resolved_through_canonical_resolver}`",
            f"- http_ask_status: `{provider.http_ask_status_code}`",
            f"- ask_persisted: `{provider.ask_run_persisted}`",
            f"- failure_code: `{provider.failure_code}`",
        ]
        if provider.server_model_digest:
            provider_lines.append(
                f"- server_model_digest: `{provider.server_model_digest}`"
            )
        provider_lines.append("")
        lines.extend(provider_lines)
    if result.index_identity is not None:
        identity = result.index_identity
        lines.extend(
            [
                "## Shared index",
                "",
                f"- workspace_id: `{identity.workspace_id}`",
                f"- collection_identity: `{identity.collection_identity}`",
                f"- source_id: `{identity.source_id}`",
                f"- document_id: `{identity.document_id}`",
                f"- content_hash: `{identity.content_hash}`",
                f"- chunk_count: `{identity.chunk_count}`",
                f"- vector_count: `{identity.vector_count}`",
                f"- embedding_provider: `{identity.embedding.provider}`",
                f"- embedding_model: `{identity.embedding.model}`",
                f"- embedding_dimensions: `{identity.embedding.dimensions}`",
                "",
            ]
        )
    inv = result.index_invariance
    lines.extend(
        [
            "## Index invariance",
            "",
            f"- embedding_identity: `{inv.embedding_identity.value}`",
            f"- collection_identity: `{inv.collection_identity.value}`",
            f"- vector_count: `{inv.vector_count.value}`",
            f"- source_identity: `{inv.source_identity.value}`",
            f"- document_identity: `{inv.document_identity.value}`",
            f"- content_hash: `{inv.content_hash.value}`",
            f"- chunk_count: `{inv.chunk_count.value}`",
            f"- no_reindex: `{inv.no_reindex.value}`",
            "",
        ]
    )
    if result.limitations:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {item}" for item in result.limitations)
    return "\n".join(lines)


def read_existing_evidence_schema(json_path: Path) -> str | None:
    if not json_path.is_file():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid"
    schema = payload.get("schema_version")
    return str(schema) if schema is not None else "invalid"


def evidence_is_stale(json_path: Path) -> bool:
    schema = read_existing_evidence_schema(json_path)
    if schema is None:
        return False
    return schema != PROOF_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class _CanonicalFileState:
    existed: bool
    content: bytes | None


@dataclass(frozen=True, slots=True)
class _CanonicalPairState:
    json_state: _CanonicalFileState
    markdown_state: _CanonicalFileState


def _capture_canonical_file_state(path: Path) -> _CanonicalFileState:
    if not path.is_file():
        return _CanonicalFileState(existed=False, content=None)
    return _CanonicalFileState(existed=True, content=path.read_bytes())


def _capture_canonical_pair_state(
    json_path: Path,
    markdown_path: Path,
) -> _CanonicalPairState:
    return _CanonicalPairState(
        json_state=_capture_canonical_file_state(json_path),
        markdown_state=_capture_canonical_file_state(markdown_path),
    )


def _restore_canonical_file(path: Path, state: _CanonicalFileState) -> None:
    if state.existed:
        if state.content is None:
            raise ValueError("canonical_restore_missing_content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(state.content)
        return
    if path.is_file():
        path.unlink()


def _restore_canonical_pair_state(
    json_path: Path,
    markdown_path: Path,
    state: _CanonicalPairState,
) -> None:
    _restore_canonical_file(json_path, state.json_state)
    _restore_canonical_file(markdown_path, state.markdown_state)


def _backup_canonical_file(path: Path) -> Path | None:
    if not path.is_file():
        return None
    backup_path = path.parent / f".{path.name}.rollback.{secrets.token_hex(8)}"
    shutil.copy2(path, backup_path)
    return backup_path


def _write_validated_temp_file(
    directory: Path,
    *,
    suffix: str,
    content: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fd, raw_name = tempfile.mkstemp(suffix=suffix, dir=directory, text=False)
    path = Path(raw_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _unlink_if_exists(path: Path | None) -> None:
    if path is not None and path.is_file():
        path.unlink()


def _validate_evidence_payload(json_text: str, markdown_text: str) -> None:
    payload = json.loads(json_text)
    if payload.get("schema_version") != PROOF_SCHEMA_VERSION:
        raise ValueError("evidence_schema_mismatch")
    if payload.get("overall_status") != ProofOverallStatus.PASS.value:
        raise ValueError("evidence_not_pass")
    if PROOF_SCHEMA_VERSION not in markdown_text:
        raise ValueError("markdown_schema_missing")
    if "**PASS**" not in markdown_text:
        raise ValueError("markdown_overall_missing")

    proof_id = payload.get("proof_id")
    if not isinstance(proof_id, str) or not proof_id:
        raise ValueError("evidence_proof_id_missing")
    if f"proof_id: `{proof_id}`" not in markdown_text:
        raise ValueError("markdown_proof_id_mismatch")

    provider_results = payload.get("provider_results", {})
    if not isinstance(provider_results, dict):
        raise ValueError("evidence_provider_results_invalid")

    for provider_name in ("ollama", "vllm"):
        provider_data = provider_results.get(provider_name)
        if not isinstance(provider_data, dict):
            raise ValueError(f"evidence_provider_missing:{provider_name}")
        configured_model = provider_data.get("configured_model")
        if not isinstance(configured_model, str) or not configured_model:
            raise ValueError(f"evidence_model_missing:{provider_name}")
        if f"configured_model: `{configured_model}`" not in markdown_text:
            raise ValueError(f"markdown_model_mismatch:{provider_name}")

    ollama = provider_results.get("ollama", {})
    if isinstance(ollama, dict):
        digest = ollama.get("server_model_digest")
        if digest is not None:
            if not isinstance(digest, str) or not digest:
                raise ValueError("evidence_digest_invalid")
            if f"server_model_digest: `{digest}`" not in markdown_text:
                raise ValueError("markdown_digest_mismatch")


def _publish_evidence_pair_transactional(
    *,
    json_path: Path,
    markdown_path: Path,
    json_text: str,
    markdown_text: str,
) -> None:
    _validate_evidence_payload(json_text, markdown_text)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_json: Path | None = None
    tmp_markdown: Path | None = None
    json_backup: Path | None = None
    markdown_backup: Path | None = None
    original_state = _capture_canonical_pair_state(json_path, markdown_path)

    try:
        tmp_json = _write_validated_temp_file(
            json_path.parent,
            suffix=".json.tmp",
            content=json_text,
        )
        tmp_markdown = _write_validated_temp_file(
            markdown_path.parent,
            suffix=".md.tmp",
            content=markdown_text,
        )
        _validate_evidence_payload(
            tmp_json.read_text(encoding="utf-8"),
            tmp_markdown.read_text(encoding="utf-8"),
        )

        json_backup = _backup_canonical_file(json_path)
        markdown_backup = _backup_canonical_file(markdown_path)

        os.replace(tmp_json, json_path)
        tmp_json = None
        os.replace(tmp_markdown, markdown_path)
        tmp_markdown = None

        _validate_evidence_payload(
            json_path.read_text(encoding="utf-8"),
            markdown_path.read_text(encoding="utf-8"),
        )
    except Exception:
        _restore_canonical_pair_state(json_path, markdown_path, original_state)
        raise
    finally:
        _unlink_if_exists(tmp_json)
        _unlink_if_exists(tmp_markdown)
        _unlink_if_exists(json_backup)
        _unlink_if_exists(markdown_backup)


def invalidate_canonical_evidence(
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    original_state = _capture_canonical_pair_state(json_path, markdown_path)
    try:
        if json_path.is_file():
            json_path.unlink()
        if markdown_path.is_file():
            markdown_path.unlink()
    except Exception:
        _restore_canonical_pair_state(json_path, markdown_path, original_state)
        raise


def write_evidence(
    result: ModelRuntimeProofResult,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    if result.overall_status is not ProofOverallStatus.PASS:
        invalidate_canonical_evidence(json_path=json_path, markdown_path=markdown_path)
        return

    json_text = serialize_result_json(result)
    markdown_text = render_markdown(result)
    _publish_evidence_pair_transactional(
        json_path=json_path,
        markdown_path=markdown_path,
        json_text=json_text,
        markdown_text=markdown_text,
    )


def stale_evidence_notice(json_path: Path) -> str | None:
    schema = read_existing_evidence_schema(json_path)
    if schema is None or schema == PROOF_SCHEMA_VERSION:
        return None
    return (
        f"STALE EVIDENCE: canonical schema `{schema}` "
        f"differs from current `{PROOF_SCHEMA_VERSION}`"
    )
