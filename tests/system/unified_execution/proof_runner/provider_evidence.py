# © Artur Czarnecki. All rights reserved.

"""OTLP and Ollama provider evidence readers for UE-11G-C1."""

from __future__ import annotations

import json
from pathlib import Path

from tests.system.unified_execution.proof_runner.contracts import (
    OllamaModelEvidence,
    OtlpIdentityEvidence,
)

_INTERGRAX_FIELDS: tuple[tuple[str, str], ...] = (
    ("run_id", "intergrax.run_id"),
    ("task_id", "intergrax.task_id"),
    ("execution_id", "intergrax.execution_id"),
    ("attempt_id", "intergrax.attempt_id"),
    ("capability", "intergrax.capability"),
    ("agent_id", "intergrax.agent_id"),
    ("tool_id", "intergrax.tool_id"),
)


class EvidenceReadError(RuntimeError):
    pass


def read_otlp_identity_evidence(
    *,
    log_path: Path,
    run_id: str,
) -> OtlpIdentityEvidence:
    if not log_path.is_file():
        raise EvidenceReadError("otlp_log_missing")
    records = _load_otlp_records(log_path)
    matched = [record for record in records if record.get("run_id") == run_id]
    if not matched:
        raise EvidenceReadError("otlp_run_id_not_found")
    search_records = [
        record
        for record in matched
        if record.get("capability") == "local.workspace.search"
    ]
    scoped = search_records if search_records else matched
    execution_ids = _unique_field_values(scoped, "execution_id")
    attempt_ids = _unique_field_values(scoped, "attempt_id")
    task_ids = _unique_field_values(scoped, "task_id")
    capabilities = _unique_field_values(scoped, "capability")
    agent_ids = _unique_field_values(scoped, "agent_id")
    tool_ids = _unique_field_values(scoped, "tool_id")
    if not execution_ids:
        raise EvidenceReadError("otlp_execution_id_missing")
    if len(execution_ids) != 1:
        raise EvidenceReadError("otlp_execution_id_not_unique")
    return OtlpIdentityEvidence(
        run_id=run_id,
        task_id=task_ids[0] if task_ids else None,
        execution_id=execution_ids[0],
        attempt_id=attempt_ids[0] if attempt_ids else None,
        capability=capabilities[0] if capabilities else None,
        agent_id=agent_ids[0] if agent_ids else None,
        tool_id=tool_ids[0] if tool_ids else None,
        event_count=len(scoped),
    )


def probe_ollama_model(
    *,
    tags_payload: dict[str, object],
    model_name: str,
) -> OllamaModelEvidence:
    models = tags_payload.get("models")
    listed = False
    digest_present = False
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str):
                continue
            if name == model_name or name.startswith(f"{model_name}:"):
                listed = True
                digest = entry.get("digest")
                digest_present = isinstance(digest, str) and bool(digest.strip())
                break
    return OllamaModelEvidence(
        model_name=model_name,
        digest_present=digest_present,
        listed_after_run=listed,
    )


def _load_otlp_records(log_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_no, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        records.extend(_iter_otlp_records(payload, line_no=line_no))
    return records


def _iter_otlp_records(payload: dict[str, object], *, line_no: int) -> list[dict[str, str]]:
    del line_no
    flattened_records: list[dict[str, str]] = []
    resource_logs = payload.get("resourceLogs")
    if not isinstance(resource_logs, list):
        return flattened_records
    for resource_log in resource_logs:
        if not isinstance(resource_log, dict):
            continue
        scope_logs = resource_log.get("scopeLogs")
        if not isinstance(scope_logs, list):
            continue
        for scope_log in scope_logs:
            if not isinstance(scope_log, dict):
                continue
            log_records = scope_log.get("logRecords")
            if not isinstance(log_records, list):
                continue
            for log_record in log_records:
                if not isinstance(log_record, dict):
                    continue
                attrs = _flatten_attributes(log_record.get("attributes"))
                record: dict[str, str] = {}
                for target, source in _INTERGRAX_FIELDS:
                    value = attrs.get(source)
                    if isinstance(value, str) and value:
                        record[target] = value
                if record:
                    flattened_records.append(record)
    return flattened_records


def _flatten_attributes(attributes: object) -> dict[str, str]:
    if not isinstance(attributes, list):
        return {}
    flattened: dict[str, str] = {}
    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        key = attribute.get("key")
        value = attribute.get("value")
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        scalar = _otlp_scalar(value)
        if scalar is not None:
            flattened[key] = scalar
    return flattened


def _otlp_scalar(value: dict[str, object]) -> str | None:
    for field in ("stringValue", "intValue", "boolValue", "doubleValue"):
        if field in value:
            raw = value[field]
            if isinstance(raw, str):
                return raw
            if isinstance(raw, bool):
                return "true" if raw else "false"
            if isinstance(raw, (int, float)):
                return str(raw)
    return None


def _unique_field_values(records: list[dict[str, str]], field: str) -> list[str]:
    seen: list[str] = []
    for record in records:
        value = record.get(field)
        if value and value not in seen:
            seen.append(value)
    return seen
