# © Artur Czarnecki. All rights reserved.

"""APP-HOST-8C — LKW foreground facade and CLI unit tests."""

from __future__ import annotations

import ast
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import pytest

from intergrax.hosting import (
    HostedApplicationExitKind,
    HostedApplicationExitRecord,
    HostedApplicationLifecycleState,
    HostedApplicationSupervisorResult,
)
from intergrax.hosting.supervisor import HostedApplicationSupervisorAttemptRecord
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from local_workspace_application.tests.lkw_ac3_projection import (
    create_lkw_hosted_test_process_composition,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting import foreground as foreground_module
from local_workspace_application.hosting.__main__ import (
    _exit_code,
    _safe_result_payload,
    main,
)
from local_workspace_application.hosting.foreground import (
    run_local_workspace_hosted_application,
)

pytestmark = [pytest.mark.unit]

_OCCURRED_AT = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)


def _exit_record(
    kind: HostedApplicationExitKind,
    *,
    reason_code: str,
    retryable: bool = False,
) -> HostedApplicationExitRecord:
    return HostedApplicationExitRecord(
        exit_kind=kind,
        retryable=retryable,
        reason_code=reason_code,
        application_id="local_workspace",
        instance_id="01TESTHOSTEDFOREGROUNDINSTANCE01",
        profile_digest="profile-digest-fixture",
        terminal_lifecycle_state=HostedApplicationLifecycleState.STOPPED,
        occurred_at=_OCCURRED_AT,
    )


def _supervisor_result(
    kind: HostedApplicationExitKind,
) -> HostedApplicationSupervisorResult:
    exit_record = _exit_record(
        kind,
        reason_code=kind.value,
        retryable=False,
    )
    return HostedApplicationSupervisorResult(
        application_id="local_workspace",
        profile_digest="profile-digest-fixture",
        definition_digest="definition-digest-fixture",
        final_exit=exit_record,
        attempts=(
            HostedApplicationSupervisorAttemptRecord(
                attempt_number=0,
                instance_id=exit_record.instance_id,
                exit_record=exit_record,
                cleanup_verified=True,
            ),
        ),
        restart_exhausted=False,
    )


def test_run_local_workspace_hosted_application_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = LocalWorkspaceBackendSettings(backend_port=18021)
    built_profile = object()
    expected_result = _supervisor_result(HostedApplicationExitKind.CLEAN_STOP)
    profile_calls = 0
    run_calls = 0
    captured: dict[str, object] = {}

    def _fake_build(
        *,
        process_composition: ProductionProcessComposition,
        settings: LocalWorkspaceBackendSettings | None = None,
    ) -> object:
        nonlocal profile_calls
        profile_calls += 1
        captured["settings"] = settings
        captured["process_composition"] = process_composition
        return built_profile

    def _fake_run(profile: object) -> HostedApplicationSupervisorResult:
        nonlocal run_calls
        run_calls += 1
        captured["profile"] = profile
        return expected_result

    monkeypatch.setattr(
        foreground_module, "build_local_workspace_hosted_profile", _fake_build
    )
    monkeypatch.setattr(foreground_module, "run_hosted_application", _fake_run)

    composition = create_lkw_hosted_test_process_composition()
    result = run_local_workspace_hosted_application(
        process_composition=composition,
        settings=settings,
    )

    assert profile_calls == 1
    assert run_calls == 1
    assert captured["settings"] is settings
    assert captured["process_composition"] is composition
    assert captured["profile"] is built_profile
    assert result is expected_result


@pytest.mark.parametrize(
    ("kind", "code"),
    [
        (HostedApplicationExitKind.CLEAN_STOP, 0),
        (HostedApplicationExitKind.INSTANCE_CONFLICT, 2),
        (HostedApplicationExitKind.STARTUP_FAILURE, 1),
        (HostedApplicationExitKind.RUNTIME_FAILURE, 1),
    ],
)
def test_exit_code_mapping(kind: HostedApplicationExitKind, code: int) -> None:
    assert _exit_code(_supervisor_result(kind)) == code


def test_main_emits_safe_json(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _supervisor_result(HostedApplicationExitKind.CLEAN_STOP)

    def _fake_run(
        *,
        process_composition: ProductionProcessComposition | None = None,
        settings: LocalWorkspaceBackendSettings | None = None,
    ) -> HostedApplicationSupervisorResult:
        del process_composition, settings
        return expected

    monkeypatch.setattr(
        "local_workspace_application.hosting.__main__.run_local_workspace_hosted_application",
        _fake_run,
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main()
    assert code == 0
    payload = json.loads(buffer.getvalue())
    assert payload["schema_version"] == "local_workspace.hosted_process_result.v1"
    assert payload["application_id"] == "local_workspace"
    assert payload["profile_digest"] == expected.profile_digest
    assert payload["definition_digest"] == expected.definition_digest
    assert "final_exit" in payload
    assert isinstance(payload["attempts"], list)
    assert payload["restart_exhausted"] is False
    serialized = json.dumps(payload)
    assert "terminal_result" not in serialized
    assert "diagnostics" not in serialized
    assert "traceback" not in serialized
    assert "exception" not in serialized
    assert _safe_result_payload(expected)["schema_version"] == payload["schema_version"]


def test_foreground_module_import_boundary() -> None:
    path = Path(foreground_module.__file__).resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "HostedApplicationEngine",
        "HostedApplicationSupervisor",
        "FileHostedApplicationInstanceGuard",
        "PortableForegroundSignalAdapter",
        "HostedApplicationControlCoordinator",
        "HostedApplicationServiceRegistry",
        "asyncio",
        "signal",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[-1] not in forbidden
                assert alias.name not in forbidden
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module not in forbidden
                for part in node.module.split("."):
                    assert part not in forbidden
            for alias in node.names:
                assert alias.name not in forbidden
