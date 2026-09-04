# © Artur Czarnecki. All rights reserved.

"""AW-4C — collaborative work bridge architecture gate tests."""

from __future__ import annotations

import ast
import importlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.autonomous_work.collaborative_work_intake import RecordingCollaborativeWorkIntake
from intergrax.contracts.autonomous_work import (
    GoalEvaluationDisposition,
    GoalEvaluationReasonCode,
    Revision,
    initial_revision,
    mint_wake_up_id,
)
from intergrax.contracts.autonomous_work.collaborative_work_bridge import (
    CollaborativeWorkRequest,
    CollaborativeWorkSubmissionDisposition,
    are_collaborative_work_requests_equivalent,
    derive_collaborative_work_request_identity,
    resolve_collaborative_work_submission_replay,
)
from intergrax.contracts.autonomous_work.references import ProgressProjectionRef
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit

_UTC = UTC
_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=_UTC)

_FORBIDDEN_TOKENS = (
    "class WorkItem",
    "class Assignment",
    "WorkerWorkItem",
    "AutonomousWorkItem",
    "WorkerAssignment",
    "AutonomousAssignment",
    "WorkItemRepository",
    "AssignmentRepository",
    "openai",
    "anthropic",
    "langchain",
    "runtime.task",
    "runtime.events",
    "agents.",
    "APScheduler",
    "celery",
)

_FORBIDDEN_IMPORT_ROOTS = (
    "sqlalchemy",
    "redis",
    "boto3",
    "psycopg",
    "asyncpg",
    "pymongo",
)


def _bridge_module_paths() -> list[Path]:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    return [
        base / "worker_collaborative_work_bridge.py",
        base / "collaborative_work_intake.py",
    ]


def _contract_path() -> Path:
    module = importlib.import_module(
        "intergrax.contracts.autonomous_work.collaborative_work_bridge"
    )
    assert module.__file__ is not None
    return Path(module.__file__)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_bridge_modules_have_no_workitem_or_execution_imports() -> None:
    for path in _bridge_module_paths():
        joined = "\n".join(_imported_modules(path)).lower()
        for token in (
            "runtime.task",
            "runtime.events",
            "runtime.nexus",
            "agents.",
            "execution",
            "openai",
            "anthropic",
        ):
            assert token not in joined, f"{path.name} imports {token}"


def test_bridge_modules_have_no_forbidden_tokens() -> None:
    for path in _bridge_module_paths():
        source = path.read_text(encoding="utf-8")
        lowered = source.lower()
        assert "while true" not in lowered, path.name
        for token in _FORBIDDEN_TOKENS:
            if token.lower() in {"apscheduler", "celery"}:
                assert token.lower() not in lowered, f"{path.name} contains {token}"


def test_bridge_modules_have_no_provider_dependencies() -> None:
    for path in _bridge_module_paths():
        for module_name in _imported_modules(path):
            root = module_name.split(".", 1)[0]
            assert root not in _FORBIDDEN_IMPORT_ROOTS, (
                f"{path.name} imports provider {module_name}"
            )


def test_contract_does_not_define_workitem_or_assignment() -> None:
    source = _contract_path().read_text(encoding="utf-8")
    for token in (
        "class WorkItem",
        "class Assignment",
        "WorkItemRepository",
        "AssignmentRepository",
    ):
        assert token not in source


def test_autonomous_work_package_does_not_define_canonical_workitem() -> None:
    package = importlib.import_module("intergrax.autonomous_work")
    assert package.__file__ is not None
    base = Path(package.__file__).parent
    for path in base.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "class WorkItem" not in source, path.name
        assert "class Assignment" not in source, path.name


def test_bridge_service_does_not_import_collaborative_work_domain() -> None:
    module = importlib.import_module(
        "intergrax.autonomous_work.worker_collaborative_work_bridge"
    )
    assert module.__file__ is not None
    joined = "\n".join(_imported_modules(Path(module.__file__)))
    assert "intergrax.collaborative_work" not in joined


def _sample_request(
    *,
    wake_up_id: str | None = None,
    goal_revision: Revision | None = None,
    reason: str = "SLA risk requires collaborative work",
    evidence_refs: tuple[str, ...] = ("evidence/sla/at-risk",),
) -> CollaborativeWorkRequest:
    worker_id = contract_suite.worker_instance().worker_instance_id
    goal_id = contract_suite.worker_goal().goal_id
    resolved_wake_up_id = wake_up_id or mint_wake_up_id()
    request_identity = derive_collaborative_work_request_identity(
        worker_instance_id=worker_id,
        goal_id=goal_id,
        wake_up_id=resolved_wake_up_id,
    )
    return CollaborativeWorkRequest(
        request_identity=request_identity,
        worker_instance_id=worker_id,
        responsibility_id=contract_suite.responsibility().responsibility_id,
        goal_id=goal_id,
        goal_revision=goal_revision or initial_revision(),
        wake_up_id=resolved_wake_up_id,
        decision_disposition=GoalEvaluationDisposition.ACTION_REQUIRED,
        reason=reason,
        reason_code=GoalEvaluationReasonCode.SLA_RISK,
        evidence_refs=evidence_refs,
        progress_projection_ref=ProgressProjectionRef("projection/sla-30m"),
        requested_priority=contract_suite.worker_goal().priority,
        evaluated_at=_NOW,
        requested_at=_NOW,
        title=contract_suite.worker_goal().objective,
    )


def test_replay_helper_classifies_accept_already_exists_and_conflict() -> None:
    first = _sample_request()
    second = replace(first, reason="different reason")
    assert (
        resolve_collaborative_work_submission_replay(existing=None, incoming=first)
        is CollaborativeWorkSubmissionDisposition.ACCEPTED
    )
    assert (
        resolve_collaborative_work_submission_replay(existing=first, incoming=first)
        is CollaborativeWorkSubmissionDisposition.ALREADY_EXISTS
    )
    assert (
        resolve_collaborative_work_submission_replay(existing=first, incoming=second)
        is CollaborativeWorkSubmissionDisposition.CONFLICT
    )


def test_requested_at_is_excluded_from_logical_equivalence() -> None:
    wake_up_id = mint_wake_up_id()
    first = _sample_request(wake_up_id=wake_up_id)
    second = replace(first, requested_at=_NOW.replace(minute=5))
    assert are_collaborative_work_requests_equivalent(first, second)


def test_recording_adapter_never_overwrites_on_conflict() -> None:
    intake = RecordingCollaborativeWorkIntake()
    first = _sample_request()
    second = replace(first, reason="conflicting reason")
    intake.submit(first)
    result = intake.submit(second)
    assert result.disposition is CollaborativeWorkSubmissionDisposition.CONFLICT
    assert intake.submissions[0].reason == first.reason
