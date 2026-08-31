# © Artur Czarnecki. All rights reserved.

"""SQLite runtime-event persistence evidence for UE-11G-C1."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.system.unified_execution.proof_runner.contracts import OtlpIdentityEvidence


class SqliteEvidenceReadError(RuntimeError):
    pass


def read_sqlite_runtime_identity_evidence(
    *,
    db_path: Path,
    tenant_id: str,
    run_id: str,
) -> OtlpIdentityEvidence:
    if not db_path.is_file():
        raise SqliteEvidenceReadError("runtime_events_db_missing")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT event_json
            FROM runtime_events
            WHERE tenant_id = ? AND run_id = ?
            ORDER BY execution_position ASC
            """,
            (tenant_id, run_id),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise SqliteEvidenceReadError("runtime_events_run_not_found")
    parsed_events: list[dict[str, object]] = []
    for row in rows:
        event_json = row["event_json"]
        if not isinstance(event_json, str):
            continue
        payload = json.loads(event_json)
        if isinstance(payload, dict):
            parsed_events.append(payload)
    if not parsed_events:
        raise SqliteEvidenceReadError("runtime_events_empty")
    root_event = parsed_events[0]
    root_execution_id = root_event.get("execution_id")
    root_attempt_id = root_event.get("attempt_id")
    if not isinstance(root_attempt_id, str) or not root_attempt_id:
        raise SqliteEvidenceReadError("runtime_attempt_id_missing")
    root_execution = root_execution_id if isinstance(root_execution_id, str) else None
    execution_ids: list[str] = []
    attempt_ids: list[str] = []
    task_ids: list[str] = []
    agent_ids: list[str] = []
    tool_ids: list[str] = []
    llm_call_events = 0
    for payload in parsed_events:
        execution_id = payload.get("execution_id")
        attempt_id = payload.get("attempt_id")
        task_id = payload.get("task_id")
        agent_id = payload.get("agent_id")
        event_type = payload.get("event_type")
        if isinstance(execution_id, str) and execution_id not in execution_ids:
            execution_ids.append(execution_id)
        if isinstance(attempt_id, str) and attempt_id not in attempt_ids:
            attempt_ids.append(attempt_id)
        if isinstance(task_id, str) and task_id not in task_ids:
            task_ids.append(task_id)
        if isinstance(agent_id, str) and agent_id not in agent_ids:
            agent_ids.append(agent_id)
        if event_type == "llm_call":
            llm_call_events += 1
        event_payload = payload.get("payload")
        if isinstance(event_payload, dict):
            tool_id = event_payload.get("tool_id")
            if isinstance(tool_id, str) and tool_id not in tool_ids:
                tool_ids.append(tool_id)
    if root_execution is not None and root_execution not in execution_ids:
        raise SqliteEvidenceReadError("runtime_root_execution_missing")
    return OtlpIdentityEvidence(
        run_id=run_id,
        task_id=task_ids[0] if task_ids else None,
        execution_id=root_execution,
        attempt_id=root_attempt_id,
        capability="local.workspace.search",
        agent_id=agent_ids[0] if agent_ids else None,
        tool_id=tool_ids[0] if tool_ids else None,
        event_count=len(rows) + llm_call_events,
    )
