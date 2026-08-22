# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import io
import json
import shutil
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

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


def _noop_teardown() -> None:
    return None


def _all_ok_runners(core: ModuleType) -> dict[str, Callable[[Any], Any]]:
    def _ok(name: str, receipt: str | None = None) -> Callable[[Any], Any]:
        def _runner(_config: Any) -> Any:
            return core.PhaseOutcome(name=name, ok=True, receipt_id=receipt)

        return _runner

    return {
        "startup": _ok("startup"),
        "sentry": _ok("sentry"),
        "elasticsearch": _ok("elasticsearch"),
        "persistence": _ok("persistence"),
        "background-task": _ok("background-task", "bg-receipt"),
        "application-hosting": _ok("application-hosting", "host-receipt"),
        "file-watcher": _ok("file-watcher", "fw-receipt"),
    }


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
    teardown_calls = {"n": 0}

    def _ok(name: str) -> Callable[[Any], Any]:
        def _runner(_config: Any) -> Any:
            called.append(name)
            return core.PhaseOutcome(name=name, ok=True, receipt_id=f"{name}-id")

        return _runner

    def _fail(_config: Any) -> Any:
        called.append("elasticsearch")
        raise core.CoreProofError("elasticsearch_boom")

    def _teardown() -> None:
        teardown_calls["n"] += 1

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
        code = core.run_core_proof(
            _config(core),
            phase_runners=runners,
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 1
    assert called == ["startup", "sentry", "elasticsearch"]
    assert teardown_calls["n"] == 1
    assert "core_proof_result=FAIL" in text
    assert "failed_phase=elasticsearch" in text
    assert "failure_reason=elasticsearch_boom" in text
    assert "core_teardown_attempted=true" in text
    assert "core_teardown_result=PASS" in text
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
    teardown_calls = {"n": 0}

    def _ok(name: str, receipt: str | None = None) -> Callable[[Any], Any]:
        def _runner(_config: Any) -> Any:
            order.append(name)
            return core.PhaseOutcome(name=name, ok=True, receipt_id=receipt)

        return _runner

    def _teardown() -> None:
        teardown_calls["n"] += 1

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
        code = core.run_core_proof(
            _config(core),
            phase_runners=runners,
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 0
    assert teardown_calls["n"] == 1
    assert order == list(core.ALL_PHASE_ORDER)
    for field in (
        "core_teardown_attempted=true",
        "core_teardown_result=PASS",
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
    teardown_index = text.index("core_teardown_result=PASS")
    last_phase_index = text.rindex("core_phase_result=PASS")
    assert last_phase_index < teardown_index < pass_index


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
    assert "uv run --project applications/local_workspace_application python" in compact
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


def test_canonical_compose_input_set_is_deterministic_and_valid(
    core: ModuleType,
) -> None:
    assert core._SAMPLE_DOCS_DIR == core._PROOF_DOCS_DIR
    assert core.discover_compose_files() == [
        core._BASE_COMPOSE,
        core._ES_COMPOSE,
        core._KAFKA_COMPOSE,
        core._MONGODB_COMPOSE,
        core._DOCKER_DIR / "docker-compose.sentry.yml",
    ]

    if shutil.which("docker") is None:
        pytest.skip("docker is required for canonical compose validation")

    completed = core.run_command(
        [*core.compose_args(core.discover_compose_files()), "config"],
        cwd=core._REPO_ROOT,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    rendered = core.run_command(
        [*core.compose_args(core.discover_compose_files()), "config", "--format", "json"],
        cwd=core._REPO_ROOT,
        timeout=120,
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)["services"]
    local_workspace = services["local_workspace"]
    assert (
        local_workspace["environment"]["INTERGRAX_ALLOWED_READ_ROOTS"]
        == "/data/user_docs"
    )
    assert (
        local_workspace["environment"]["INTERGRAX_RAG_CHUNKING_STRATEGY"]
        == "recursive"
    )
    assert (
        services["lkw-background-worker"]["environment"][
            "INTERGRAX_RAG_CHUNKING_STRATEGY"
        ]
        == "recursive"
    )
    assert any(
        volume["target"] == "/data/user_docs"
        for volume in local_workspace["volumes"]
    )


def test_watcher_uses_materialized_runtime_context(core: ModuleType) -> None:
    watcher = (
        _REPO_ROOT
        / "applications/local_workspace_application/docker/file-watcher-e2e.compose.yml"
    )
    text = watcher.read_text(encoding="utf-8")
    assert "context: ./runtime-context" in text
    assert "context: ../../.." not in text
    assert "dockerfile: applications/local_workspace_application/docker/Dockerfile" not in text


def test_startup_materializes_runtime_context_before_build(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        core,
        "check_startup_host_port_preflight",
        lambda *_a, **_k: events.append("preflight"),
    )
    monkeypatch.setattr(core, "materialize_runtime_context", lambda: events.append("materialize"))
    monkeypatch.setattr(core, "compose_config", lambda *_a, **_k: events.append("config"))
    monkeypatch.setattr(core, "compose_down", lambda *_a, **_k: events.append("down"))
    monkeypatch.setattr(core, "clear_sentry_runtime_state", lambda: events.append("clear"))
    monkeypatch.setattr(core, "compose_up", lambda *_a, **_k: events.append("build"))
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)

    core.phase_startup(_config(core))

    assert events == ["preflight", "materialize", "config", "down", "clear", "build"]


def test_materialization_uses_canonical_application_builder(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run_command(args: Any, **kwargs: Any) -> Any:
        calls.append((list(args), kwargs))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(core, "run_command", fake_run_command)
    core.materialize_runtime_context()

    assert calls == [
        (
            [
                "uv",
                "run",
                "python",
                str(core._APPLICATION_IMAGE_BUILDER),
                "--application",
                "local_workspace_application",
                "--context-dir",
                str(core._RUNTIME_CONTEXT_DIR),
                "--materialize-only",
            ],
            {"cwd": core._REPO_ROOT, "timeout": 300},
        )
    ]


def test_background_task_search_payload_preserves_workspace_scope() -> None:
    child_script = _SCRIPT.parent / "run-lkw-background-task-proof.py"
    spec = importlib.util.spec_from_file_location(
        "run_lkw_background_task_proof",
        child_script,
    )
    assert spec is not None
    assert spec.loader is not None
    child = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(child)

    payload = child.build_background_task_search_payload(
        tenant_id="lkw-background-proof",
        marker="marker",
        run_id="run-1",
        task_id="task-1",
        correlation_id="corr-1",
        collection_id="local_workspace",
    )

    assert payload["workspace_id"] == "lkw-background-proof"
    assert payload["metadata"]["workspace_id"] == "lkw-background-proof"
    assert payload["metadata"]["collection_id"] == "local_workspace"


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
    accepted_only = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.index_summary.v1": {
                        "accepted_count": 1,
                        "ingested_count": 0,
                        "chunk_count": 0,
                    },
                }
            }
        }
    }
    with pytest.raises(core.CoreProofError) as accepted_exc:
        core.require_positive_ingest(accepted_only)
    assert accepted_exc.value.reason == "index_not_ingested"
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_ingest({"metadata": {}})
    assert exc.value.reason == "index_not_ingested"


def _search_response(
    *,
    used: Any = True,
    reason: Any = "retrieve_complete",
    evidence_count: Any = 3,
    num_results: Any = None,
    source_refs: Any = None,
    include_reason: bool = True,
    include_used: bool = True,
    include_evidence_count: bool = True,
    include_num_results: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if include_used:
        summary["used"] = used
    if include_reason:
        summary["reason"] = reason
    if include_evidence_count:
        summary["evidence_count"] = evidence_count
    if include_num_results:
        summary["num_results"] = num_results
    if source_refs is not None:
        summary["source_refs"] = source_refs
    return {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.search_summary.v1": summary,
                }
            }
        }
    }


def test_positive_search_requires_retrieve_complete_reason(core: ModuleType) -> None:
    assert (
        core.require_positive_search(
            _search_response(evidence_count=3, include_num_results=False)
        )
        == 3
    )
    assert (
        core.require_positive_search(
            _search_response(
                include_evidence_count=False,
                include_num_results=True,
                num_results=2,
            )
        )
        == 2
    )


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
        pytest.param(123, id="int"),
        pytest.param(True, id="bool"),
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
        pytest.param("future_unknown_reason", id="unknown"),
        pytest.param("Retrieve_Complete", id="wrong_case"),
        pytest.param("retrieve_complete_extra", id="extra_suffix"),
    ],
)
def test_positive_search_rejects_missing_or_malformed_reason(
    core: ModuleType, reason: Any
) -> None:
    response = _search_response(reason=reason, evidence_count=3)
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_search(response)
    assert exc.value.reason == "search_results_missing"


def test_positive_search_rejects_absent_reason_key(core: ModuleType) -> None:
    response = _search_response(include_reason=False, evidence_count=3)
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_search(response)
    assert exc.value.reason == "search_results_missing"


@pytest.mark.parametrize(
    ("include_used", "used"),
    [
        pytest.param(False, True, id="absent"),
        pytest.param(True, False, id="false"),
        pytest.param(True, None, id="none"),
        pytest.param(True, 1, id="int_one"),
        pytest.param(True, "true", id="string_true"),
    ],
)
def test_positive_search_rejects_nonexact_used(
    core: ModuleType, include_used: bool, used: Any
) -> None:
    response = _search_response(
        include_used=include_used,
        used=used,
        reason="retrieve_complete",
        evidence_count=3,
    )
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_search(response)
    assert exc.value.reason == "search_results_missing"


@pytest.mark.parametrize(
    "summary_overrides",
    [
        pytest.param(
            {"evidence_count": 0, "num_results": 0, "include_num_results": True},
            id="both_zero",
        ),
        pytest.param(
            {
                "include_evidence_count": False,
                "include_num_results": False,
            },
            id="both_absent",
        ),
        pytest.param(
            {"evidence_count": -1, "include_num_results": False},
            id="negative_evidence",
        ),
        pytest.param(
            {
                "include_evidence_count": False,
                "include_num_results": True,
                "num_results": -2,
            },
            id="negative_num_results",
        ),
        pytest.param(
            {"evidence_count": "abc", "include_num_results": False},
            id="non_numeric_evidence",
        ),
        pytest.param(
            {
                "include_evidence_count": False,
                "include_num_results": True,
                "num_results": "xyz",
            },
            id="non_numeric_num_results",
        ),
    ],
)
def test_positive_search_rejects_nonpositive_result_count(
    core: ModuleType, summary_overrides: dict[str, Any]
) -> None:
    response = _search_response(
        used=True,
        reason="retrieve_complete",
        **summary_overrides,
    )
    with pytest.raises(core.CoreProofError) as exc:
        core.require_positive_search(response)
    assert exc.value.reason == "search_results_missing"


@pytest.mark.parametrize("bad_position", ["before", "after"])
def test_persistence_fails_closed_when_search_missing_reason(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_position: str,
) -> None:
    index_response = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.index_summary.v1": {"ingested_count": 1},
                }
            }
        }
    }
    bad_search = _search_response(
        include_reason=False,
        evidence_count=0,
        num_results=0,
        include_num_results=True,
    )
    search_calls = {"n": 0}
    index_bodies: list[Mapping[str, Any]] = []
    restarted = {"value": False}

    def fake_http_post_json(
        _url: str, body: Mapping[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        capability = body.get("capability")
        if capability == "local.workspace.index":
            index_bodies.append(body)
            return index_response
        if capability == "local.workspace.search":
            search_calls["n"] += 1
            if bad_position == "before" and not restarted["value"]:
                return bad_search
            if bad_position == "after" and restarted["value"]:
                return bad_search
            proof_files = sorted(tmp_path.glob("lkw_persistence_proof_*.txt"))
            expected = f"/data/user_docs/{proof_files[0].name}"
            return _search_response(evidence_count=2, source_refs=[expected])
        raise AssertionError(f"unexpected capability: {capability!r}")

    monkeypatch.setattr(core, "discover_compose_files", lambda: [])
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "http_post_json", fake_http_post_json)
    monkeypatch.setattr(core, "_PERSISTENCE_SEARCH_RETRY_SLEEP_SECONDS", 0.01)

    def fake_compose_restart(*_a: Any, **_k: Any) -> None:
        restarted["value"] = True

    monkeypatch.setattr(core, "compose_restart", fake_compose_restart)
    compose_up_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_compose_up(*args: Any, **kwargs: Any) -> None:
        compose_up_calls.append((args, kwargs))

    monkeypatch.setattr(core, "compose_up", fake_compose_up)
    monkeypatch.setattr(core, "_SAMPLE_DOCS_DIR", tmp_path)
    monkeypatch.setattr(
        core,
        "prepare_ollama_embedding_model",
        lambda *_a, **_k: "nomic-embed-text",
    )

    with pytest.raises(core.CoreProofError) as exc:
        core.phase_persistence(_config(core, phase_timeout_seconds=1))
    assert exc.value.reason == "search_results_missing"
    assert search_calls["n"] >= 1
    assert index_bodies[0]["metadata"]["chunking_strategy_id"] == "recursive"
    assert len(compose_up_calls) == 1
    assert compose_up_calls[0][0][1] == ["local_workspace", "qdrant", "ollama"]
    assert compose_up_calls[0][1]["build"] is False


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


def test_latest_persistence_proof_marker_reads_proof_docs(
    core: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_docs = tmp_path / "proof_docs"
    proof_docs.mkdir()
    (proof_docs / "lkw_persistence_proof_old.txt").write_text(
        "Unique marker: OLD_MARKER\n",
        encoding="utf-8",
    )
    latest = proof_docs / "lkw_persistence_proof_new.txt"
    latest.write_text("Unique marker: NEW_MARKER\n", encoding="utf-8")
    monkeypatch.setattr(core, "_PROOF_DOCS_DIR", proof_docs)
    assert core._latest_persistence_proof_marker() == "NEW_MARKER"


def test_file_watcher_retrieve_warmup_probes_persistence_collection(core: ModuleType) -> None:
    assert core._PERSISTENCE_PROOF_SCOPE_ID == "lkw-persistence-proof"
    assert core._PERSISTENCE_PROOF_SCOPE_ID != core._FILE_WATCHER_SCOPE_ID


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


def test_background_task_child_failure_propagates_causal_reason(
    core: ModuleType,
) -> None:
    child_output = (
        "proof_result=FAIL\n"
        "failure_reason=background_task_not_succeeded\n"
        "task_status=FAILED\n"
        "error_message=handler_timeout\n"
    )
    error = core.background_task_child_failure(exit_code=1, child_output=child_output)
    assert error.reason == "background_task_not_succeeded"
    assert error.child_exit_code == 1
    assert error.child_details == {
        "task_status": "FAILED",
        "error_message": "handler_timeout",
    }


def test_background_task_child_failure_propagates_search_results_missing(
    core: ModuleType,
) -> None:
    child_output = (
        "proof_result=FAIL\n"
        "failure_reason=search_results_missing\n"
        "search_results=0\n"
        "search_reason=retrieve_complete\n"
        "search_used=False\n"
        "search_num_results=0\n"
    )
    error = core.background_task_child_failure(exit_code=1, child_output=child_output)
    assert error.reason == "search_results_missing"
    assert error.child_details["search_results"] == "0"
    assert error.child_details["search_reason"] == "retrieve_complete"
    assert error.child_details["search_used"] == "False"


def test_background_task_child_failure_falls_back_without_structured_reason(
    core: ModuleType,
) -> None:
    child_output = "random stderr noise\nno structured kv here\n"
    error = core.background_task_child_failure(exit_code=1, child_output=child_output)
    assert error.reason == "background_task_child_failed"


def test_background_task_child_failure_rejects_unsafe_reason(
    core: ModuleType,
) -> None:
    child_output = "failure_reason=bad reason with spaces\n"
    error = core.background_task_child_failure(exit_code=1, child_output=child_output)
    assert error.reason == "background_task_child_failed"


def test_background_task_child_diagnostics_do_not_leak_arbitrary_output(
    core: ModuleType,
) -> None:
    child_output = (
        "proof_result=FAIL\n"
        "failure_reason=search_results_missing\n"
        "arbitrary_stdout=must_not_forward\n"
        "receipt_message=mongodb://user:secret@host/db\n"
        "error_message=connect failed password=leak\n"
        "search_reason=bad reason\n"
    )
    details = core.extract_background_task_child_diagnostics(
        core.parse_kv_output(child_output)
    )
    assert "arbitrary_stdout" not in details
    assert "receipt_message" not in details
    assert "error_message" not in details
    assert "search_reason" not in details
    assert details == {}


def test_background_task_child_diagnostics_allowlist_only(core: ModuleType) -> None:
    child_output = (
        "proof_result=FAIL\n"
        "failure_reason=kafka_task_topic_empty\n"
        "kafka_topic=intergrax.tasks\n"
        "kafka_topic_messages=0\n"
        "receipt_error=ProofReceiptVerificationError\n"
    )
    details = core.extract_background_task_child_diagnostics(
        core.parse_kv_output(child_output)
    )
    assert details == {
        "kafka_topic": "intergrax.tasks",
        "kafka_topic_messages": "0",
        "receipt_error": "ProofReceiptVerificationError",
    }


def test_background_task_phase_prepares_embedding_model_before_child(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    services_started: list[str] = []

    def fake_compose_up(_files: Any, services: Sequence[str], **_k: Any) -> None:
        services_started.extend(services)

    monkeypatch.setattr(core, "compose_config", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "compose_up", fake_compose_up)
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: events.append("health"))
    monkeypatch.setattr(core, "wait_for_compose_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_http_reachable", lambda *_a, **_k: None)
    monkeypatch.setattr(
        core,
        "prepare_ollama_embedding_model",
        lambda *_a, **_k: events.append("embedding_ready") or "nomic-embed-text",
    )
    monkeypatch.setattr(
        core,
        "run_python_child",
        lambda *_a, **_k: (
            0,
            "proof_result=PASS\n"
            "proof_receipt_recorded=true\n"
            "proof_receipt_verified=true\n"
            "proof_receipt_query_verified=true\n"
            "document_store_provider=mongodb\n"
            "message_bus_provider=kafka\n"
            "proof_receipt_id=bg-1\n",
        ),
    )
    monkeypatch.setattr(core, "mongodb_child_env", lambda **_k: {})

    core.phase_background_task(_config(core, phase_timeout_seconds=5))

    assert "qdrant" in services_started
    assert "ollama" in services_started
    assert events == ["health", "embedding_ready"]


def test_background_task_child_failure_emits_parent_kv(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "validate_os_wrapper_pair",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "discover_compose_files", lambda: [])
    monkeypatch.setattr(core, "compose_config", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "compose_up", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_compose_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_http_reachable", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "mongodb_child_env", lambda **_k: {})
    monkeypatch.setattr(
        core,
        "prepare_ollama_embedding_model",
        lambda *_a, **_k: "nomic-embed-text",
    )

    def fake_run_python_child(*_a: Any, **_k: Any) -> tuple[int, str]:
        return (
            1,
            "proof_result=FAIL\n"
            "failure_reason=search_results_missing\n"
            "search_num_results=0\n",
        )

    monkeypatch.setattr(core, "run_python_child", fake_run_python_child)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core, phase="background-task"),
            phase_runners={"background-task": core.phase_background_task},
            teardown=_noop_teardown,
        )
    text = buffer.getvalue()
    assert code == 1
    assert "failed_phase=background-task" in text
    assert "failure_reason=search_results_missing" in text
    assert "child_exit_code=1" in text
    assert "search_num_results=0" in text


def test_run_python_child_uses_utf8_transport(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_command(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(core, "run_command", fake_run_command)
    core.run_python_child(core._BACKGROUND_PROOF_PY, [], cwd=core._REPO_ROOT)
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "strict"
    assert captured["kwargs"]["env"]["PYTHONIOENCODING"] == "utf-8"


def test_run_python_child_decodes_non_ascii_output(
    core: ModuleType,
    tmp_path: Path,
) -> None:
    script = tmp_path / "utf8_child.py"
    script.write_text(
        'import sys\nprint("proof_marker=zażółć")\n',
        encoding="utf-8",
    )
    exit_code, output = core.run_python_child(script, [], cwd=tmp_path)
    assert exit_code == 0
    assert "proof_marker=zażółć" in output


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


def test_extract_search_diagnostics_reads_typed_summary(core: ModuleType) -> None:
    response = _search_response(
        evidence_count=2,
        num_results=1,
        include_num_results=True,
        source_refs=["/data/user_docs/a.txt", "/data/user_docs/b.txt"],
    )
    summary = response["metadata"]["lkw_evidence.v1"]["diagnostics"][
        "lkw.search_summary.v1"
    ]
    summary["raw_tool_reason"] = "retrieve_complete"
    diagnostics = core.extract_search_diagnostics(response)
    assert diagnostics is not None
    assert diagnostics.used is True
    assert diagnostics.reason == "retrieve_complete"
    assert diagnostics.raw_tool_reason == "retrieve_complete"
    assert diagnostics.num_results == 1
    assert diagnostics.evidence_count == 2
    assert diagnostics.source_refs == (
        "/data/user_docs/a.txt",
        "/data/user_docs/b.txt",
    )


def test_persistence_search_requires_exact_source_ref(core: ModuleType) -> None:
    expected = "/data/user_docs/lkw_persistence_proof_20260101120000.txt"
    diagnostics = core.SearchDiagnostics(
        num_results=1,
        evidence_count=1,
        source_refs=(expected,),
        raw_tool_reason=None,
        used=True,
        reason="retrieve_complete",
    )
    assert core.persistence_search_succeeded(
        diagnostics, expected_source_ref=expected
    )
    assert not core.persistence_search_succeeded(
        diagnostics,
        expected_source_ref="/data/user_docs/other.txt",
    )


def test_poll_persistence_search_stops_when_source_ref_found(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "/data/user_docs/proof.txt"
    calls = {"n": 0}

    def fake_http_post_json(
        _url: str, _body: Mapping[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            return _search_response(evidence_count=1, source_refs=["/data/other.txt"])
        return _search_response(evidence_count=1, source_refs=[expected])

    monkeypatch.setattr(core, "http_post_json", fake_http_post_json)
    monkeypatch.setattr(core, "_PERSISTENCE_SEARCH_RETRY_SLEEP_SECONDS", 0.01)
    diagnostics, count = core.poll_persistence_search(
        _config(core, phase_timeout_seconds=5),
        tenant_id="t",
        workspace_id="w",
        collection_id="c",
        marker="MARKER",
        expected_source_ref=expected,
        deadline=time.monotonic() + 1.0,
    )
    assert calls["n"] == 2
    assert count == 1
    assert diagnostics is not None
    assert expected in diagnostics.source_refs


def test_poll_persistence_search_timeout_reports_last_diagnostics(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "http_post_json",
        lambda *_a, **_k: _search_response(
            reason="retrieve_pending",
            evidence_count=0,
            include_evidence_count=True,
        ),
    )
    monkeypatch.setattr(core, "_PERSISTENCE_SEARCH_RETRY_SLEEP_SECONDS", 0.01)
    diagnostics, count = core.poll_persistence_search(
        _config(core, phase_timeout_seconds=1),
        tenant_id="t",
        workspace_id="w",
        collection_id="c",
        marker="MARKER",
        expected_source_ref="/data/user_docs/missing.txt",
        deadline=time.monotonic() + 0.05,
    )
    assert count == 0
    assert diagnostics is not None
    assert diagnostics.reason == "retrieve_pending"


def test_persistence_phase_polls_until_exact_source_ref(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    search_calls = {"n": 0}
    restarted = {"value": False}
    index_response = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.index_summary.v1": {"ingested_count": 1},
                }
            }
        }
    }

    def fake_http_post_json(
        _url: str, body: Mapping[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        capability = body.get("capability")
        if capability == "local.workspace.index":
            return index_response
        if capability == "local.workspace.search":
            search_calls["n"] += 1
            proof_files = sorted(tmp_path.glob("lkw_persistence_proof_*.txt"))
            expected = f"/data/user_docs/{proof_files[0].name}"
            if not restarted["value"] and search_calls["n"] == 1:
                return _search_response(
                    evidence_count=1, source_refs=["/data/other.txt"]
                )
            return _search_response(evidence_count=1, source_refs=[expected])
        raise AssertionError(f"unexpected capability: {capability!r}")

    monkeypatch.setattr(core, "discover_compose_files", lambda: [])
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "http_post_json", fake_http_post_json)
    monkeypatch.setattr(core, "_PERSISTENCE_SEARCH_RETRY_SLEEP_SECONDS", 0.01)

    def fake_compose_restart(*_a: Any, **_k: Any) -> None:
        restarted["value"] = True

    monkeypatch.setattr(core, "compose_restart", fake_compose_restart)
    monkeypatch.setattr(core, "compose_up", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "_SAMPLE_DOCS_DIR", tmp_path)
    monkeypatch.setattr(
        core,
        "prepare_ollama_embedding_model",
        lambda *_a, **_k: "nomic-embed-text",
    )

    outcome = core.phase_persistence(_config(core, phase_timeout_seconds=5))
    assert outcome.ok is True
    assert search_calls["n"] == 3
    assert outcome.details["source_ref_found_before_restart"] == "true"
    assert outcome.details["source_ref_found_after_restart"] == "true"


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


_BOOTSTRAP_SCRIPT = (
    _REPO_ROOT
    / "applications/local_workspace_application/scripts/"
    / "lkw_ollama_embedding_bootstrap.py"
)


def _load_bootstrap_module() -> ModuleType:
    module_name = "lkw_ollama_embedding_bootstrap"
    spec = importlib.util.spec_from_file_location(module_name, _BOOTSTRAP_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bootstrap() -> ModuleType:
    return _load_bootstrap_module()


def test_prepare_ollama_embedding_model_resolves_and_pulls(
    core: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run_command(args: list[str], **_kwargs: Any) -> Any:
        calls.append(list(args))
        if "local_workspace" in args:
            return type(
                "CP", (), {"returncode": 0, "stdout": "nomic-embed-text\n", "stderr": ""}
            )()
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(core, "run_command", fake_run_command)
    model_name = core.prepare_ollama_embedding_model(
        [],
        cwd=core._REPO_ROOT,
        timeout_seconds=10,
    )
    assert model_name == "nomic-embed-text"
    pull_command = " ".join(calls[1])
    assert "ollama" in pull_command
    assert calls[1][-1] == "nomic-embed-text"


@pytest.mark.parametrize("output", ["one\ntwo\n", "\n", "x" * 257, "bad\x01model\n"])
def test_prepare_ollama_embedding_model_rejects_invalid_resolution(
    core: ModuleType, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    monkeypatch.setattr(
        core,
        "run_command",
        lambda *_a, **_k: type(
            "CP", (), {"returncode": 0, "stdout": output, "stderr": ""}
        )(),
    )
    with pytest.raises(core.CoreProofError) as exc:
        core.prepare_ollama_embedding_model(
            [],
            cwd=core._REPO_ROOT,
            timeout_seconds=10,
        )
    assert exc.value.reason == "embedding_model_resolution_failed"


def test_prepare_ollama_embedding_model_pull_failure_fails_closed(
    core: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_command(args: list[str], **_kwargs: Any) -> Any:
        if "local_workspace" in args:
            return type(
                "CP", (), {"returncode": 0, "stdout": "nomic-embed-text\n", "stderr": ""}
            )()
        return type("CP", (), {"returncode": 1, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(core, "run_command", fake_run_command)
    with pytest.raises(core.CoreProofError) as exc:
        core.prepare_ollama_embedding_model(
            [],
            cwd=core._REPO_ROOT,
            timeout_seconds=10,
        )
    assert exc.value.reason == "embedding_model_pull_failed"


def test_persistence_phase_waits_for_embedding_model_before_index(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []
    index_response = {
        "metadata": {
            "lkw_evidence.v1": {
                "diagnostics": {
                    "lkw.index_summary.v1": {"ingested_count": 1},
                }
            }
        }
    }

    def fake_prepare(*_a: Any, **_k: Any) -> str:
        order.append("embedding_ready")
        return "nomic-embed-text"

    def fake_http_post_json(
        _url: str, body: Mapping[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        capability = body.get("capability")
        if capability == "local.workspace.index":
            order.append("index")
            return index_response
        if capability == "local.workspace.search":
            proof_files = sorted(tmp_path.glob("lkw_persistence_proof_*.txt"))
            expected = f"/data/user_docs/{proof_files[0].name}"
            return _search_response(evidence_count=1, source_refs=[expected])
        raise AssertionError(f"unexpected capability: {capability!r}")

    monkeypatch.setattr(core, "discover_compose_files", lambda: [])
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "http_post_json", fake_http_post_json)
    monkeypatch.setattr(core, "compose_up", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "compose_restart", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "_PERSISTENCE_SEARCH_RETRY_SLEEP_SECONDS", 0.01)
    monkeypatch.setattr(core, "_SAMPLE_DOCS_DIR", tmp_path)
    monkeypatch.setattr(core, "prepare_ollama_embedding_model", fake_prepare)

    core.phase_persistence(_config(core, phase_timeout_seconds=5))
    assert order[:2] == ["embedding_ready", "index"]


def test_compose_args_use_explicit_proof_project(core: ModuleType) -> None:
    compose_files = core.discover_compose_files()
    args = core.compose_args(compose_files)
    assert args[:5] == [
        "docker",
        "compose",
        "-p",
        core._COMPOSE_PROJECT,
        "-f",
    ]
    assert str(compose_files[0]) in args
    assert core._COMPOSE_PROJECT == "lkw-core-platform-proof"
    assert core._PRODUCT_COMPOSE_PROJECT == "intergrax_lkw"


def test_occupied_required_port_fails_before_compose_down(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_preflight(_config: Any) -> None:
        events.append("preflight")
        raise core.CoreProofError(
            "required_port_unavailable",
            phase="startup",
            child_details={"occupied_port": "8020"},
        )

    monkeypatch.setattr(core, "check_startup_host_port_preflight", fake_preflight)
    monkeypatch.setattr(
        core,
        "materialize_runtime_context",
        lambda: events.append("materialize"),
    )
    monkeypatch.setattr(core, "compose_down", lambda *_a, **_k: events.append("down"))

    with pytest.raises(core.CoreProofError) as exc:
        core.phase_startup(_config(core))
    assert exc.value.reason == "required_port_unavailable"
    assert events == ["preflight"]
    assert "down" not in events


def test_product_quickstart_port_collision_has_safe_diagnostic(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "resolve_compose_published_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(
        core,
        "canonical_compose_owned_host_ports",
        lambda *, compose_exec_args, **_k: (
            frozenset({8020})
            if "intergrax_lkw" in compose_exec_args("ps")
            else frozenset()
        ),
    )
    monkeypatch.setattr(core, "is_loopback_tcp_port_reachable", lambda _port: True)
    monkeypatch.setattr(core, "probe_host_port_available", lambda _port: False)

    with pytest.raises(core.CoreProofError) as exc:
        core.check_startup_host_port_preflight(_config(core))
    assert exc.value.reason == "required_port_unavailable"
    assert exc.value.child_details is not None
    assert exc.value.child_details["occupied_port"] == "8020"
    assert exc.value.child_details["occupied_by"] == "lkw_product_quickstart"
    assert "Stop the LKW Product Quick Start stack" in exc.value.child_details[
        "recommended_action"
    ]


def test_proof_owned_ports_are_not_rejected(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "resolve_compose_published_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(
        core,
        "canonical_compose_owned_host_ports",
        lambda **_k: frozenset({8020}),
    )
    monkeypatch.setattr(core, "is_loopback_tcp_port_reachable", lambda _port: True)
    monkeypatch.setattr(core, "probe_host_port_available", lambda _port: False)

    core.check_startup_host_port_preflight(_config(core))


def test_startup_port_collision_emits_safe_failure_marker(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        core,
        "validate_os_wrapper_pair",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)

    def fail_startup(_config: Any) -> Any:
        raise core.CoreProofError(
            "required_port_unavailable",
            phase="startup",
            child_details={
                "occupied_port": "8020",
                "occupied_by": "lkw_product_quickstart",
            },
        )

    runners = {
        "startup": fail_startup,
        "sentry": lambda _c: core.PhaseOutcome(name="sentry", ok=True),
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core, phase="startup"),
            phase_runners=runners,
            teardown=_noop_teardown,
        )
    text = buffer.getvalue()
    assert code == 1
    assert "core_proof_result=FAIL" in text
    assert "failed_phase=startup" in text
    assert "failure_reason=required_port_unavailable" in text
    assert "occupied_port=8020" in text
    assert "occupied_by=lkw_product_quickstart" in text


def test_file_watcher_child_receives_compose_project_env(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_python_child(
        _script: Path,
        _args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: int | None = None,
    ) -> tuple[int, str]:
        captured["env"] = dict(env or {})
        return (
            0,
            "proof_result=PASS\n"
            "proof_kind=file_watcher_persistent_search\n"
            "embedding_warmup_completed=true\n"
            "reviewer_rerun_required=false\n"
            "source_ref_found_before_restart=true\n"
            "watcher_restored_after_restart=true\n"
            "source_ref_found_after_restart=true\n"
            "proof_receipt_recorded=true\n"
            "proof_receipt_verified=true\n"
            "proof_receipt_query_verified=true\n"
            "proof_receipt_id=watcher-1\n",
        )

    monkeypatch.setattr(core, "discover_compose_files", lambda: [])
    monkeypatch.setattr(core, "compose_config", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "compose_up", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_lkw_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_compose_health", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "wait_for_http_reachable", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "ensure_file_watcher_retrieve_ready", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "run_python_child", fake_run_python_child)

    core.phase_file_watcher(_config(core, phase_timeout_seconds=5))

    assert captured["env"]["COMPOSE_PROJECT_NAME"] == core._COMPOSE_PROJECT


def test_success_path_invokes_teardown_once_before_pass(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    teardown_calls = {"n": 0}

    def _teardown() -> None:
        teardown_calls["n"] += 1

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core),
            phase_runners=_all_ok_runners(core),
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 0
    assert teardown_calls["n"] == 1
    assert text.index("core_teardown_result=PASS") < text.index("core_proof_result=PASS")


def test_coreprooferror_path_invokes_teardown_preserves_failure(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    teardown_calls = {"n": 0}

    def _fail(_config: Any) -> Any:
        raise core.CoreProofError("HTTPError", phase="elasticsearch")

    def _teardown() -> None:
        teardown_calls["n"] += 1

    runners = _all_ok_runners(core)
    runners["elasticsearch"] = _fail
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core),
            phase_runners=runners,
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 1
    assert teardown_calls["n"] == 1
    assert "failed_phase=elasticsearch" in text
    assert "failure_reason=HTTPError" in text
    assert "core_teardown_attempted=true" in text
    assert "failure_reason=proof_teardown_failed" not in text


def test_unexpected_exception_path_invokes_teardown_preserves_exception_type(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    teardown_calls = {"n": 0}

    def _boom(_config: Any) -> Any:
        raise RuntimeError("boom")

    def _teardown() -> None:
        teardown_calls["n"] += 1

    runners = _all_ok_runners(core)
    runners["persistence"] = _boom
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core),
            phase_runners=runners,
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 1
    assert teardown_calls["n"] == 1
    assert "failed_phase=persistence" in text
    assert "failure_reason=RuntimeError" in text


def test_cleanup_failure_after_functional_failure_preserves_primary_failure(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)

    def _fail(_config: Any) -> Any:
        raise core.CoreProofError("HTTPError")

    def _teardown_fail() -> None:
        raise core.CoreProofError("compose_down_failed")

    runners = _all_ok_runners(core)
    runners["elasticsearch"] = _fail
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core),
            phase_runners=runners,
            teardown=_teardown_fail,
        )
    text = buffer.getvalue()
    assert code == 1
    assert "failed_phase=elasticsearch" in text
    assert "failure_reason=HTTPError" in text
    assert "core_teardown_result=FAIL" in text
    assert "failure_reason=proof_teardown_failed" not in text


def test_cleanup_failure_after_functional_pass_fails_overall(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)

    def _teardown_fail() -> None:
        raise core.CoreProofError("compose_down_failed")

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core),
            phase_runners=_all_ok_runners(core),
            teardown=_teardown_fail,
        )
    text = buffer.getvalue()
    assert code == 1
    assert "core_teardown_result=FAIL" in text
    assert "failure_reason=proof_teardown_failed" in text
    assert "core_proof_result=PASS" not in text
    assert "core_proof_all_phases_passed=true" not in text


def test_teardown_compose_files_include_file_watcher_overlay(
    core: ModuleType,
) -> None:
    files = core.discover_proof_teardown_compose_files()
    assert core._WATCHER_COMPOSE in files
    assert files[: len(core.discover_compose_files())] == core.discover_compose_files()


def test_teardown_uses_non_destructive_compose_down(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_compose_down(
        compose_files: Any,
        *,
        cwd: Path,
        volumes: bool = False,
        remove_orphans: bool = True,
    ) -> None:
        captured["compose_files"] = list(compose_files)
        captured["cwd"] = cwd
        captured["volumes"] = volumes
        captured["remove_orphans"] = remove_orphans

    monkeypatch.setattr(core, "compose_down", fake_compose_down)
    core.teardown_proof_compose_stack()
    assert captured["volumes"] is False
    assert captured["remove_orphans"] is True
    assert core._WATCHER_COMPOSE in captured["compose_files"]


def test_teardown_targets_proof_project_only(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run_command(
        args: Sequence[str],
        **kwargs: Any,
    ) -> Any:
        captured.append(list(args))
        return type("CP", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(core, "run_command", fake_run_command)
    core.teardown_proof_compose_stack()
    assert len(captured) == 1
    command = captured[0]
    assert command[3] == core._COMPOSE_PROJECT
    assert core._PRODUCT_COMPOSE_PROJECT not in command
    assert "down" in command
    assert "-v" not in command
    assert "--remove-orphans" in command


def test_selected_startup_phase_invokes_teardown(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "validate_os_wrapper_pair", lambda *_a, **_k: None)
    monkeypatch.setattr(core, "validate_environment", lambda *_a, **_k: None)
    teardown_calls = {"n": 0}

    def _startup(_config: Any) -> Any:
        return core.PhaseOutcome(name="startup", ok=True)

    def _teardown() -> None:
        teardown_calls["n"] += 1

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(
            _config(core, phase="startup"),
            phase_runners={"startup": _startup},
            teardown=_teardown,
        )
    text = buffer.getvalue()
    assert code == 0
    assert teardown_calls["n"] == 1
    assert "core_teardown_attempted=true" in text
    assert "core_proof_selected_phase=startup" in text


def test_pre_validation_failure_skips_teardown(
    core: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teardown_calls = {"n": 0}

    def _teardown() -> None:
        teardown_calls["n"] += 1

    monkeypatch.setattr(
        core,
        "validate_os_wrapper_pair",
        lambda *_a, **_k: (_ for _ in ()).throw(
            core.CoreProofError("operating_system_mismatch")
        ),
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = core.run_core_proof(_config(core), teardown=_teardown)
    text = buffer.getvalue()
    assert code == 1
    assert teardown_calls["n"] == 0
    assert "core_teardown_attempted=true" not in text
    assert "failure_reason=operating_system_mismatch" in text
