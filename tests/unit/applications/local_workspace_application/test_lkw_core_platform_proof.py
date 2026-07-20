# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/run-lkw-core-platform-proof.py"
)
_WINDOWS_BAT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-core-platform-proof-windows.bat"
)
_LINUX_SH = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-core-platform-proof-linux.sh"
)
_MACOS_SH = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "run-lkw-core-platform-proof-macos.sh"
)


def _load_module() -> ModuleType:
    module_name = "run_lkw_core_platform_proof"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def core() -> ModuleType:
    return _load_module()


def _config(core: ModuleType, **overrides: Any) -> Any:
    values = {
        "os_family": core.OsFamily.WINDOWS,
        "wrapper_id": core.WrapperId.WINDOWS_BAT,
        "phase": "all",
        "run_id_prefix": "lkw-core-",
        "base_url": "http://127.0.0.1:8020",
        "kafka_ui": "http://127.0.0.1:8085",
        "mongo_express": "http://127.0.0.1:8086",
        "elasticsearch_url": "http://127.0.0.1:9200",
        "kibana_url": "http://127.0.0.1:5601",
        "sentry_url": "http://127.0.0.1:9000",
        "phase_timeout_seconds": 30,
    }
    values.update(overrides)
    return core.ProofConfig(**values)


def test_os_family_frozen_values(core: ModuleType) -> None:
    assert {item.value for item in core.OsFamily} == {
        "windows",
        "linux",
        "macos",
    }


def test_wrapper_id_frozen_values(core: ModuleType) -> None:
    assert {item.value for item in core.WrapperId} == {
        "windows_bat",
        "linux_sh",
        "macos_sh",
    }


def test_valid_os_wrapper_pairs(core: ModuleType) -> None:
    assert core.VALID_OS_WRAPPER_PAIRS == frozenset(
        {
            (core.OsFamily.WINDOWS, core.WrapperId.WINDOWS_BAT),
            (core.OsFamily.LINUX, core.WrapperId.LINUX_SH),
            (core.OsFamily.MACOS, core.WrapperId.MACOS_SH),
        }
    )


@pytest.mark.parametrize(
    ("system_name", "expected"),
    [
        ("Windows", "WINDOWS"),
        ("Linux", "LINUX"),
        ("Darwin", "MACOS"),
    ],
)
def test_detect_os_family_mapping(
    core: ModuleType,
    system_name: str,
    expected: str,
) -> None:
    assert core.detect_os_family(system_name) is getattr(core.OsFamily, expected)


def test_detect_os_family_unknown_fails(core: ModuleType) -> None:
    with pytest.raises(core.CoreProofError) as exc:
        core.detect_os_family("FreeBSD")
    assert exc.value.reason == "unsupported_operating_system"


def test_os_wrapper_mismatch_fails_before_execution(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def _fail_run(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("run")
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(core.subprocess, "run", _fail_run)
    monkeypatch.setattr(core, "detect_os_family", lambda: core.OsFamily.LINUX)
    cfg = _config(core)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(cfg, phase_runners={})
    assert code == 1
    text = buffer.getvalue()
    assert "failure_reason=operating_system_mismatch" in text
    assert calls == []


def test_all_phase_order(core: ModuleType) -> None:
    assert core.ALL_PHASE_ORDER == (
        "startup",
        "sentry",
        "elasticsearch",
        "persistence",
        "background-task",
        "application-hosting",
        "file-watcher",
    )
    assert core.resolve_phases("all") == core.ALL_PHASE_ORDER


def test_stop_on_first_failure(
    core: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        core,
        "validate_os_wrapper_pair",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    called: list[str] = []

    def _ok(name: str) -> Callable[[Any], Any]:
        def _runner(_config: Any) -> Any:
            called.append(name)
            return core.PhaseOutcome(name=name, ok=True, receipt_id=f"{name}-id")

        return _runner

    def _fail(_config: Any) -> Any:
        called.append("elasticsearch")
        raise core.CoreProofError("elasticsearch_boom")

    runners = {
        "startup": _ok("startup"),
        "sentry": _ok("sentry"),
        "elasticsearch": _fail,
        "persistence": _ok("persistence"),
        "background-task": _ok("background-task"),
        "application-hosting": _ok("application-hosting"),
        "file-watcher": _ok("file-watcher"),
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(_config(core), phase_runners=runners)
    text = buffer.getvalue()
    assert code == 1
    assert called == ["startup", "sentry", "elasticsearch"]
    assert "core_proof_result=FAIL" in text
    assert "failed_phase=elasticsearch" in text
    assert "core_proof_result=PASS" not in text
    assert "core_proof_all_phases_passed=true" not in text


def test_final_pass_contract(core: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        core,
        "validate_os_wrapper_pair",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    order: list[str] = []

    def _ok(name: str, receipt: str | None = None) -> Callable[[Any], Any]:
        def _runner(_config: Any) -> Any:
            order.append(name)
            return core.PhaseOutcome(name=name, ok=True, receipt_id=receipt)

        return _runner

    runners = {
        "startup": _ok("startup"),
        "sentry": _ok("sentry"),
        "elasticsearch": _ok("elasticsearch"),
        "persistence": _ok("persistence"),
        "background-task": _ok("background-task", "bg-receipt"),
        "application-hosting": _ok("application-hosting", "host-receipt"),
        "file-watcher": _ok("file-watcher", "fw-receipt"),
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(_config(core), phase_runners=runners)
    text = buffer.getvalue()
    assert code == 0
    assert order == list(core.ALL_PHASE_ORDER)
    for field in (
        "core_proof_result=PASS",
        "core_proof_os_family=windows",
        "core_proof_wrapper_id=windows_bat",
        "core_proof_shared_python_runner=true",
        "core_proof_all_phases_passed=true",
        "startup_phase=PASS",
        "sentry_phase=PASS",
        "elasticsearch_phase=PASS",
        "persistence_phase=PASS",
        "background_task_phase=PASS",
        "application_hosting_phase=PASS",
        "file_watcher_phase=PASS",
        "individual_proof_receipts_authoritative=true",
        "aggregate_terminal_summary_authoritative=false",
        "optional_os_interaction_proof_executed=false",
        "background_task_proof_receipt_id=bg-receipt",
        "application_hosting_proof_receipt_id=host-receipt",
        "file_watcher_proof_receipt_id=fw-receipt",
    ):
        assert field in text
    pass_index = text.index("core_proof_result=PASS")
    last_phase_index = text.rindex("core_phase_result=PASS")
    assert last_phase_index < pass_index


def test_no_shell_implementation_boundary(core: ModuleType) -> None:
    source = _SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "shell=True",
        "cmd.exe",
        "powershell.exe",
        "pwsh",
        "Invoke-RestMethod",
        "Invoke-WebRequest",
        "curl",
        "jq",
        "run-lkw-persistence-proof.ps1",
        "run-lkw-persistence-proof.bat",
        "run-lkw-elasticsearch-proof.bat",
        "run-lkw-elasticsearch-proof.sh",
        "run-lkw-background-task-proof.bat",
        "run-lkw-hosting-proof.bat",
        "run-lkw-file-watcher-e2e-proof.bat",
        "hard-reset-local-docker-all.bat",
        "check-lkw-platform-proof-status.bat",
    ):
        assert forbidden not in source, forbidden
    assert "shell=False" in source


@pytest.mark.parametrize(
    ("path", "os_family", "wrapper_id"),
    [
        (_WINDOWS_BAT, "windows", "windows_bat"),
        (_LINUX_SH, "linux", "linux_sh"),
        (_MACOS_SH, "macos", "macos_sh"),
    ],
)
def test_wrapper_thinness(
    path: Path,
    os_family: str,
    wrapper_id: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    assert "run-lkw-core-platform-proof.py" in text
    assert (
        f"--os-family {os_family}" in text
        or f"--os-family {os_family}" in text.replace("\\\n", " ")
    )
    # Normalize line continuations for POSIX wrappers.
    compact = " ".join(line.strip() for line in text.splitlines())
    assert f"--os-family {os_family}" in compact
    assert f"--wrapper-id {wrapper_id}" in compact
    assert "uv run --extra integrations-mongodb python" in compact
    if path.suffix == ".bat":
        assert "%*" in text
        assert "EXIT_CODE" in text
        for forbidden in (
            "docker",
            "powershell",
            "curl",
            "Invoke-",
            "proof_receipt_recorded",
        ):
            assert forbidden not in text
    else:
        assert '"$@"' in text
        assert "exec uv run" in text
        for forbidden in (
            "docker",
            "curl",
            "jq",
            "python -c",
            "proof_receipt_recorded",
        ):
            assert forbidden not in text


def test_positive_ingest_and_missing_evidence(core: ModuleType) -> None:
    ok = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.index_summary.v1": {"ingested_count": 2},
                }
            }
        }
    }
    assert core.require_positive_ingest(ok) == 2
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_ingest({"metadata": {}})
    assert exc.value.reason == "index_not_ingested"


def test_positive_search_and_unknown_reason(core: ModuleType) -> None:
    ok = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "retrieve_complete",
                        "evidence_count": 3,
                    }
                }
            }
        }
    }
    assert core.require_positive_search(ok) == 3
    unknown = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "future_unknown_reason",
                        "evidence_count": 3,
                    }
                }
            }
        }
    }
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_search(unknown)
    assert exc.value.reason == "search_results_missing"
    missing = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "retrieve_complete",
                        "evidence_count": 0,
                        "num_results": 0,
                    }
                }
            }
        }
    }
    with pytest.raises(core.CoreProofError):
        core.require_positive_search(missing)


def test_search_retrieve_ready_accepts_zero_hit_complete(core: ModuleType) -> None:
    ready = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "retrieve_complete",
                        "evidence_count": 0,
                        "num_results": 0,
                    }
                }
            }
        }
    }
    assert core.search_retrieve_ready(ready) is True
    not_ready = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": False,
                        "reason": "retrieve_failed",
                        "raw_tool_reason": "no_hits",
                    }
                }
            }
        }
    }
    assert core.search_retrieve_ready(not_ready) is False


def test_safe_failure_reason(core: ModuleType) -> None:
    assert (
        core.safe_failure_reason(
            {"failure_reason": "embedding_warmup_failed"},
            fallback="file_watcher_child_failed",
        )
        == "embedding_warmup_failed"
    )
    assert (
        core.safe_failure_reason(
            {"failure_reason": "bad reason with spaces"},
            fallback="file_watcher_child_failed",
        )
        == "file_watcher_child_failed"
    )


def test_search_after_restart_helpers(core: ModuleType) -> None:
    before = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "retrieve_complete",
                        "num_results": 1,
                    }
                }
            }
        }
    }
    after = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": {
                        "used": True,
                        "reason": "retrieve_complete",
                        "evidence_count": 2,
                    }
                }
            }
        }
    }
    assert core.extract_search_result_count(before) == 1
    assert core.extract_search_result_count(after) == 2


@pytest.mark.parametrize(
    ("validator_name", "good", "mutations"),
    [
        (
            "validate_background_task_child_output",
            {
                "proof_result": "PASS",
                "proof_receipt_recorded": "true",
                "proof_receipt_verified": "true",
                "proof_receipt_query_verified": "true",
                "document_store_provider": "mongodb",
                "message_bus_provider": "kafka",
                "proof_receipt_id": "bg-1",
            },
            (
                {"proof_result": "FAIL"},
                {"proof_receipt_recorded": "false"},
                {"proof_receipt_verified": "false"},
                {"proof_receipt_query_verified": "false"},
                {"proof_receipt_id": ""},
            ),
        ),
        (
            "validate_hosting_child_output",
            {
                "proof_result": "PASS",
                "proof_kind": "platform_application_hosting",
                "proof_receipt_recorded": "true",
                "proof_receipt_verified": "true",
                "proof_receipt_query_verified": "true",
                "proof_receipt_id": "host-1",
            },
            (
                {"proof_result": "FAIL"},
                {"proof_kind": "wrong_kind"},
                {"proof_receipt_recorded": "false"},
                {"proof_receipt_verified": "false"},
                {"proof_receipt_query_verified": "false"},
                {"proof_receipt_id": ""},
            ),
        ),
        (
            "validate_file_watcher_child_output",
            {
                "proof_result": "PASS",
                "proof_kind": "file_watcher_persistent_search",
                "embedding_warmup_completed": "true",
                "reviewer_rerun_required": "false",
                "source_ref_found_before_restart": "true",
                "watcher_restored_after_restart": "true",
                "source_ref_found_after_restart": "true",
                "proof_receipt_recorded": "true",
                "proof_receipt_verified": "true",
                "proof_receipt_query_verified": "true",
                "proof_receipt_id": "fw-1",
            },
            (
                {"proof_result": "FAIL"},
                {"proof_kind": "wrong_kind"},
                {"proof_receipt_recorded": "false"},
                {"proof_receipt_verified": "false"},
                {"proof_receipt_query_verified": "false"},
                {"proof_receipt_id": ""},
            ),
        ),
    ],
)
def test_child_output_validators_fail_closed(
    core: ModuleType,
    validator_name: str,
    good: dict[str, str],
    mutations: tuple[dict[str, str], ...],
) -> None:
    validator = getattr(core, validator_name)
    assert validator(good)
    for mutation in mutations:
        bad = dict(good)
        bad.update(mutation)
        with pytest.raises(core.CoreProofError):
            validator(bad)
