# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.provider_extensions import (
    LLMProviderExtensions,
    VllmProviderExtensions,
)
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.providers.vllm_diagnostics import (
    VllmDiagnosticsSnapshot,
    VllmHealthStatus,
    VllmMetricsSnapshot,
)
from intergrax.runtime.token_optimization.proofs import vllm_prefix_cache_live as live_module
from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_live import (
    VllmPrefixCacheLiveProofConfig,
    _build_arg_parser,
    build_default_config,
    config_from_namespace,
    main,
    run_vllm_prefix_cache_live_proof,
)
from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_report import (
    atomic_write_text,
    render_markdown_report,
    serialize_safe_json,
    validate_safe_report_text,
    write_proof_artifacts,
)
from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_report import (
    VllmPrefixCacheLiveProofAggregateResult,
    VllmPrefixCacheLiveProofAggregateSummary,
    VllmPrefixCacheLiveProofConfiguration,
    VllmPrefixCacheLiveProofEnvironment,
    VllmPrefixCacheLiveProofRunResult,
)
from intergrax.runtime.token_optimization.vllm_prefix_cache_proof import (
    VllmPrefixCacheProofCaseId,
    VllmPrefixCacheProofCaseResult,
    VllmPrefixCacheProofReasonCode,
    evaluate_vllm_prefix_cache_proof,
)


def _metrics() -> VllmMetricsSnapshot:
    return VllmMetricsSnapshot(
        prefix_cache_queries=10.0,
        prefix_cache_hits=4.0,
        prompt_tokens_cached=100.0,
        kv_cache_usage_perc=0.5,
    )


def _diagnostics() -> VllmDiagnosticsSnapshot:
    return VllmDiagnosticsSnapshot(
        health=VllmHealthStatus(healthy=True, status_code=200),
        server_version="0.23.0",
        metrics=_metrics(),
    )


def _fast_command_runner(args, **kwargs):
    if args[:2] == ["nvidia-smi"]:
        return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
    if args in (["docker", "version"], ["docker", "compose", "version"]):
        return subprocess.CompletedProcess(args, 0, "ok", "")
    if args[:3] == ["docker", "compose", "config"]:
        return subprocess.CompletedProcess(args, 0, "", "")
    if args[:2] == ["docker", "inspect"] and "{{.State.Running}}" in args:
        return subprocess.CompletedProcess(args, 0, "false", "")
    return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture(autouse=True)
def fast_gpu_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_module,
        "_collect_gpu_preflight",
        lambda runner: (True, "GPU", 24576, "550.0"),
    )


@pytest.fixture(autouse=True)
def verify_server_model_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_module,
        "_verify_server_model",
        lambda *args, **kwargs: True,
    )


@pytest.fixture
def fast_command_runner():
    return _fast_command_runner


def _proof_config(**overrides: Any) -> VllmPrefixCacheLiveProofConfig:
    defaults = build_default_config()
    values = {
        "runs": defaults.runs,
        "output_dir": defaults.output_dir,
        "base_url": defaults.base_url,
        "model": defaults.model,
        "minimum_prefix_chars": defaults.minimum_prefix_chars,
        "connect_timeout_seconds": defaults.connect_timeout_seconds,
        "read_timeout_seconds": defaults.read_timeout_seconds,
        "startup_timeout_seconds": defaults.startup_timeout_seconds,
        "manage_vllm": defaults.manage_vllm,
        "force_recreate_vllm": defaults.force_recreate_vllm,
        "keep_vllm_running": defaults.keep_vllm_running,
    }
    values.update(overrides)
    return VllmPrefixCacheLiveProofConfig(**values)


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._case_index = 0

    def generate_with_tools(
        self,
        messages,
        tools_schema,
        *,
        max_tokens=None,
        run_id=None,
        **kwargs,
    ) -> LLMAdapterResponse:
        self.calls.append(("generate_with_tools", run_id))
        cached_values = (0, 300, 50, 0, 300, 50, 0, 300, 50)
        hit_values = (0.0, 2.0, 0.5, 0.0, 2.0, 0.5, 0.0, 2.0, 0.5)
        index = self._case_index
        self._case_index += 1
        cached = cached_values[index % len(cached_values)]
        return LLMAdapterResponse(
            content="proof-output",
            usage=LLMTokenUsage.from_counts(
                input_tokens=500,
                output_tokens=10,
                cached_input_tokens=cached,
            ),
            provider_extensions=LLMProviderExtensions(
                vllm=VllmProviderExtensions(prompt_tokens_details_reported=True)
            ),
        )

    def generate_messages(self, messages, *, max_tokens=None, run_id=None, **kwargs):
        raise AssertionError("proof path must use generate_with_tools")


def _passing_case_result(case_id: VllmPrefixCacheProofCaseId, prefix_hash: str) -> VllmPrefixCacheProofCaseResult:
    cached = {
        VllmPrefixCacheProofCaseId.COLD: 0,
        VllmPrefixCacheProofCaseId.WARM: 300,
        VllmPrefixCacheProofCaseId.CHANGED_PREFIX: 50,
    }[case_id]
    hits = {
        VllmPrefixCacheProofCaseId.COLD: 0.0,
        VllmPrefixCacheProofCaseId.WARM: 2.0,
        VllmPrefixCacheProofCaseId.CHANGED_PREFIX: 0.5,
    }[case_id]
    from intergrax.llm_adapters.providers.vllm_diagnostics import VllmMetricDeltas

    return VllmPrefixCacheProofCaseResult(
        case_id=case_id,
        prefix_hash=prefix_hash,
        tool_envelope_hash="tool",
        input_tokens=500,
        cached_input_tokens=cached,
        uncached_input_tokens=500 - cached,
        latency_ms=10.0,
        prompt_tokens_details_reported=True,
        metric_deltas=VllmMetricDeltas(
            prefix_cache_queries=1.0,
            prefix_cache_hits=hits,
            prompt_tokens_cached=float(cached),
            kv_cache_usage_perc=0.01,
        ),
        passed=True,
        reason_codes=(),
    )


def _passing_run(run_index: int) -> VllmPrefixCacheLiveProofRunResult:
    return VllmPrefixCacheLiveProofRunResult(
        run_index=run_index,
        passed=True,
        reason_codes=(),
        server_version="0.23.0",
        health_ok=True,
        cases=(
            _passing_case_result(VllmPrefixCacheProofCaseId.COLD, "hash-a"),
            _passing_case_result(VllmPrefixCacheProofCaseId.WARM, "hash-a"),
            _passing_case_result(VllmPrefixCacheProofCaseId.CHANGED_PREFIX, "hash-b"),
        ),
    )


def _aggregate_result(
    *,
    runs: tuple[VllmPrefixCacheLiveProofRunResult, ...] = (),
    canonical_environment: bool = False,
    exit_code: int = 3,
    reason_codes: tuple[str, ...] = (),
    environment_verified: bool = False,
    proof_gates_passed: bool = False,
    server_lifecycle_mode: str = "shared",
) -> VllmPrefixCacheLiveProofAggregateResult:
    requested = 3 if runs else 1
    passed_runs = sum(1 for run in runs if run.passed)
    all_runs_passed = passed_runs == len(runs) and bool(runs)
    return VllmPrefixCacheLiveProofAggregateResult(
        schema_version="token-optimization.vllm-prefix-cache-proof.v1",
        task_id="TOKEN-10C-LIVE-PROOF-1",
        started_at_utc="20260730T060000Z",
        finished_at_utc="20260730T060100Z",
        repository_commit="abc123",
        canonical_environment=canonical_environment,
        environment=VllmPrefixCacheLiveProofEnvironment(
            gpu_available=True,
            gpu_name="Test GPU",
            gpu_memory_total_mb=24576,
            nvidia_driver_version="550.0",
            docker_available=True,
            docker_compose_available=True,
            compose_contract_valid=True,
            vllm_image="vllm/vllm-openai:v0.23.0",
            vllm_version="0.23.0",
            model="Qwen/Qwen2.5-3B-Instruct",
            health_ok=True,
            managed_environment=canonical_environment,
            force_recreated=canonical_environment,
            exclusive_environment_expected=canonical_environment,
            server_lifecycle_mode=server_lifecycle_mode,
            server_started_by_runner=canonical_environment,
            environment_verified=environment_verified,
            proof_gates_passed=proof_gates_passed,
        ),
        configuration=VllmPrefixCacheLiveProofConfiguration(
            requested_runs=requested,
            minimum_prefix_chars=4096,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=120.0,
            startup_timeout_seconds=1800.0,
        ),
        runs=runs,
        aggregate=VllmPrefixCacheLiveProofAggregateSummary(
            canonical_pass=environment_verified and all_runs_passed and requested == 3 and not reason_codes,
            all_runs_passed=all_runs_passed,
            requested_runs=requested,
            completed_runs=len(runs),
            passed_runs=passed_runs,
            failed_runs=len(runs) - passed_runs,
            reason_codes=reason_codes,
        ),
        exit_code=exit_code,
    )


def test_default_config_uses_canonical_3b_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_DEFAULT_VLLM_MODEL", raising=False)
    config = build_default_config()
    assert config.model == "Qwen/Qwen2.5-3B-Instruct"


def test_explicit_model_override_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_DEFAULT_VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    config = build_default_config()
    assert config.model == "Qwen/Qwen2.5-7B-Instruct"


def test_cli_defaults() -> None:
    parser = _build_arg_parser()
    namespace = parser.parse_args([])
    config = config_from_namespace(namespace)
    defaults = build_default_config()
    assert config.runs == defaults.runs == 3
    assert config.minimum_prefix_chars == 4096
    assert config.connect_timeout_seconds == 5.0
    assert config.read_timeout_seconds == 120.0
    assert config.startup_timeout_seconds == 1800.0
    assert config.manage_vllm is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"runs": 0}, "runs"),
        ({"minimum_prefix_chars": 128}, "minimum-prefix-chars"),
        ({"connect_timeout_seconds": 0}, "connect-timeout"),
        ({"read_timeout_seconds": -1}, "read-timeout"),
        ({"startup_timeout_seconds": 0}, "startup-timeout"),
        ({"model": "   "}, "model"),
        ({"base_url": "not-a-url"}, "base_url"),
    ],
)
def test_cli_validation(kwargs: dict[str, Any], message: str) -> None:
    config = _proof_config(**kwargs)
    with pytest.raises(ValueError, match=message):
        live_module.validate_config(config)


def test_invalid_base_url_fails_before_http_or_docker() -> None:
    commands: list[list[str]] = []

    def command_runner(args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    with pytest.raises(ValueError, match="base_url"):
        run_vllm_prefix_cache_live_proof(
            _proof_config(base_url="not-a-url"),
            command_runner=command_runner,
            skip_report_write=True,
        )
    assert commands == []


def test_three_run_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _FakeAdapter()
    namespaces: list[str] = []
    evaluator_calls: list[int] = []

    def fake_execute_single_run(**kwargs):
        namespaces.append(kwargs["run_namespace"])
        evaluator_calls.append(kwargs["run_index"])
        return _passing_run(kwargs["run_index"])

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "_execute_single_run", fake_execute_single_run)

    uuid_values = iter(["u1", "u2", "u3", "t1", "t2", "t3", "t4", "t5", "t6"])
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=adapter,
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        uuid_factory=lambda: next(uuid_values),
        skip_report_write=True,
    )
    assert len(result.runs) == 3
    assert namespaces == ["token-10c-u1", "token-10c-u2", "token-10c-u3"]
    assert evaluator_calls == [1, 2, 3]


def test_namespace_reused_only_within_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[str] = []
    original_execute_case = live_module._execute_case

    def fake_execute_case(**kwargs):
        captured.append(kwargs["prefix_variant"])
        return original_execute_case(**kwargs)

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "fetch_vllm_metrics", lambda *args, **kwargs: _metrics())
    monkeypatch.setattr(live_module, "_execute_case", fake_execute_case)

    adapter = _FakeAdapter()
    run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path),
        adapter=adapter,
        http_client=MagicMock(spec=httpx.Client),
        uuid_factory=lambda: "same-run",
        skip_report_write=True,
    )
    assert captured[0] == captured[1] == "token-10c-same-run-proof-a"
    assert captured[2] == "token-10c-same-run-proof-b"


def test_case_sequence_cold_warm_changed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sequence: list[str] = []
    original_execute_case = live_module._execute_case

    def fake_execute_case(**kwargs):
        sequence.append(kwargs["case_id"].value)
        return original_execute_case(**kwargs)

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "fetch_vllm_metrics", lambda *args, **kwargs: _metrics())
    monkeypatch.setattr(live_module, "_execute_case", fake_execute_case)

    run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        uuid_factory=lambda: "run-a",
        skip_report_write=True,
    )
    assert sequence == ["COLD", "WARM", "CHANGED_PREFIX"]


def test_evaluator_called_once_per_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = 0

    def fake_execute_single_run(**kwargs):
        nonlocal calls
        calls += 1
        return _passing_run(kwargs["run_index"])

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "_execute_single_run", fake_execute_single_run)
    run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert calls == 3


def test_aggregate_passes_only_when_all_runs_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_execute_single_run(**kwargs):
        run = _passing_run(kwargs["run_index"])
        if kwargs["run_index"] == 2:
            return VllmPrefixCacheLiveProofRunResult(
                run_index=2,
                passed=False,
                reason_codes=(VllmPrefixCacheProofReasonCode.WARM_NOT_GREATER_THAN_COLD.value,),
                server_version="0.23.0",
                health_ok=True,
                cases=run.cases,
            )
        return run

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "_execute_single_run", fake_execute_single_run)
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.aggregate.all_runs_passed is False
    assert result.aggregate.canonical_pass is False
    assert result.exit_code == 3


def test_fewer_than_three_runs_cannot_canonical_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.aggregate.canonical_pass is False
    assert "CANONICAL_RUNS_INSUFFICIENT" in result.aggregate.reason_codes


def test_verified_shared_server_can_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.environment.server_lifecycle_mode == "shared"
    assert result.environment.server_started_by_runner is False
    assert result.environment.environment_verified is True
    assert result.aggregate.all_runs_passed is True
    assert result.aggregate.canonical_pass is True
    assert result.exit_code == 0
    assert "NONCANONICAL_SHARED_SERVER" not in result.aggregate.reason_codes


def test_shared_server_wrong_model_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "_verify_server_model", lambda *args, **kwargs: False)
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.environment.environment_verified is False
    assert result.aggregate.canonical_pass is False
    assert result.exit_code != 0
    assert "VLLM_MODEL_MISMATCH" in result.aggregate.reason_codes
    assert not result.runs


def test_shared_server_wrong_version_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = VllmDiagnosticsSnapshot(
        health=VllmHealthStatus(healthy=True, status_code=200),
        server_version="0.22.0",
        metrics=_metrics(),
    )
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: bad)
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.aggregate.canonical_pass is False
    assert result.exit_code != 0
    assert "VLLM_VERSION_MISMATCH" in result.aggregate.reason_codes


def test_missing_required_cache_metric_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from intergrax.llm_adapters.providers.vllm_diagnostics import VllmDiagnosticsError

    def raise_missing(*args, **kwargs):
        raise VllmDiagnosticsError("required vLLM metrics missing: vllm:prefix_cache_hits_total")

    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", raise_missing)
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.environment.environment_verified is False
    assert result.aggregate.canonical_pass is False
    assert "REQUIRED_METRICS_MISSING" in result.aggregate.reason_codes


def test_shared_lifecycle_visible_in_report() -> None:
    from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_report import (
        aggregate_result_to_safe_dict,
    )

    result = _aggregate_result(
        runs=(_passing_run(1), _passing_run(2), _passing_run(3)),
        environment_verified=True,
        proof_gates_passed=True,
        server_lifecycle_mode="shared",
        exit_code=0,
    )
    payload = aggregate_result_to_safe_dict(result)
    assert payload["environment"]["server_lifecycle_mode"] == "shared"
    assert payload["environment"]["environment_verified"] is True
    assert payload["aggregate"]["canonical_pass"] is True
    markdown = render_markdown_report(result)
    assert "Server lifecycle mode: **shared**" in markdown
    assert "Environment verified: **yes**" in markdown


def test_compose_canonical_default() -> None:
    compose_path = live_module._COMPOSE_FILE
    content = compose_path.read_text(encoding="utf-8")
    assert "Qwen/Qwen2.5-3B-Instruct" in content
    assert "${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" in content


def test_managed_recreated_server_canonical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def command_runner(args, **kwargs):
        commands.append(args)
        stdout = "true"
        if args[:2] == ["docker", "inspect"] and "{{.Config.Image}}" in args:
            stdout = "vllm/vllm-openai:v0.23.0"
        if args[:2] == ["docker", "inspect"] and "{{json .Config.Cmd}}" in args:
            stdout = json.dumps(
                [
                    "Qwen/Qwen2.5-3B-Instruct",
                    "--enable-prefix-caching",
                    "--prefix-caching-hash-algo",
                    "sha256",
                    "--enable-prompt-tokens-details",
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                ]
            )
        if args[:3] == ["docker", "compose", "config"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
        if args == ["docker", "version"] or args == ["docker", "compose", "version"]:
            return subprocess.CompletedProcess(args, 0, "ok", "")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_wait_for_health",
        lambda **kwargs: (True, "0.23.0"),
    )
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(
            runs=3,
            output_dir=tmp_path,
            manage_vllm=True,
            force_recreate_vllm=True,
        ),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert result.canonical_environment is True
    assert result.aggregate.canonical_pass is True
    assert result.exit_code == 0
    assert any("up" in command and "--force-recreate" in command for command in commands)
    assert not any("down" in command for command in commands)
    assert not any("volume" in " ".join(command) and "rm" in command for command in commands)


def test_safe_json_and_markdown_serialization() -> None:
    result = _aggregate_result(runs=(_passing_run(1),))
    json_text = serialize_safe_json(result)
    markdown = render_markdown_report(result)
    validate_safe_report_text(json_text)
    validate_safe_report_text(markdown)
    payload = json.loads(json_text)
    assert payload["schema_version"] == "token-optimization.vllm-prefix-cache-proof.v1"
    assert "runs" in payload


def test_forbidden_markers_rejected() -> None:
    result = _aggregate_result(runs=(_passing_run(1),))
    json_text = serialize_safe_json(result)
    with pytest.raises(ValueError):
        validate_safe_report_text(json_text + " token_optimization_proof_echo")


def test_atomic_report_writing(tmp_path: Path) -> None:
    target = tmp_path / "proof.json"
    atomic_write_text(target, '{"ok": true}')
    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    result = _aggregate_result(runs=(_passing_run(1),))
    json_path, markdown_path = write_proof_artifacts(
        result,
        output_dir=tmp_path,
        timestamp_utc="20260730T060000Z",
    )
    assert json_path.exists()
    assert markdown_path.exists()


def test_environment_unavailable_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def command_runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "fail")

    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=3, output_dir=tmp_path, manage_vllm=True),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert result.exit_code == 2


def test_proof_failure_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: VllmPrefixCacheLiveProofRunResult(
            run_index=kwargs["run_index"],
            passed=False,
            reason_codes=(VllmPrefixCacheProofReasonCode.WARM_NOT_GREATER_THAN_COLD.value,),
            server_version="0.23.0",
            health_ok=True,
            cases=_passing_run(kwargs["run_index"]).cases,
        ),
    )
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.exit_code == 3


def test_internal_failure_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())

    def boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(live_module, "_execute_single_run", boom)
    result = run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=_fast_command_runner,
        skip_report_write=True,
    )
    assert result.exit_code == 4


def test_successful_canonical_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(live_module, "_wait_for_health", lambda **kwargs: (True, "0.23.0"))
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )

    def command_runner(args, **kwargs):
        stdout = "true"
        if "{{.Config.Image}}" in args:
            stdout = "vllm/vllm-openai:v0.23.0"
        if "{{json .Config.Cmd}}" in args:
            stdout = json.dumps(
                [
                    "model",
                    "--enable-prefix-caching",
                    "--prefix-caching-hash-algo",
                    "sha256",
                    "--enable-prompt-tokens-details",
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    "hermes",
                ]
            )
        if args[:3] == ["docker", "compose", "config"] or args in (
            ["docker", "version"],
            ["docker", "compose", "version"],
        ):
            return subprocess.CompletedProcess(args, 0, "ok", "")
        if args[:2] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
        return subprocess.CompletedProcess(args, 0, stdout, "")

    result = run_vllm_prefix_cache_live_proof(
        _proof_config(
            runs=3,
            output_dir=tmp_path,
            manage_vllm=True,
            force_recreate_vllm=True,
        ),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert result.exit_code == 0


def test_cleanup_stops_only_managed_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def command_runner(args, **kwargs):
        commands.append(args)
        if args[:3] == ["docker", "compose", "config"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args in (["docker", "version"], ["docker", "compose", "version"]):
            return subprocess.CompletedProcess(args, 0, "ok", "")
        if args[:2] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
        if args[:2] == ["docker", "inspect"] and "{{.State.Running}}" in args:
            return subprocess.CompletedProcess(args, 0, "false", "")
        if "{{.Config.Image}}" in args:
            return subprocess.CompletedProcess(args, 0, "vllm/vllm-openai:v0.23.0", "")
        if "{{json .Config.Cmd}}" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(["m", "--enable-prefix-caching", "--prefix-caching-hash-algo", "sha256",
                            "--enable-prompt-tokens-details", "--enable-auto-tool-choice",
                            "--tool-call-parser", "hermes"]),
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "_wait_for_health", lambda **kwargs: (True, "0.23.0"))
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )
    run_vllm_prefix_cache_live_proof(
        _proof_config(
            runs=1,
            output_dir=tmp_path,
            manage_vllm=True,
            force_recreate_vllm=True,
        ),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert any(command[:3] == ["docker", "compose", "-f"] and command[-2:] == ["stop", "vllm"] for command in commands)


def test_keep_vllm_running_skips_stop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def command_runner(args, **kwargs):
        commands.append(args)
        if args[:3] == ["docker", "compose", "config"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args in (["docker", "version"], ["docker", "compose", "version"]):
            return subprocess.CompletedProcess(args, 0, "ok", "")
        if args[:2] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
        if args[:2] == ["docker", "inspect"] and "{{.State.Running}}" in args:
            return subprocess.CompletedProcess(args, 0, "false", "")
        if "{{.Config.Image}}" in args:
            return subprocess.CompletedProcess(args, 0, "vllm/vllm-openai:v0.23.0", "")
        if "{{json .Config.Cmd}}" in args:
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps(["m", "--enable-prefix-caching", "--prefix-caching-hash-algo", "sha256",
                            "--enable-prompt-tokens-details", "--enable-auto-tool-choice",
                            "--tool-call-parser", "hermes"]),
                "",
            )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(live_module, "_execute_warmup", lambda **kwargs: None)
    monkeypatch.setattr(live_module, "_wait_for_health", lambda **kwargs: (True, "0.23.0"))
    monkeypatch.setattr(live_module, "collect_vllm_diagnostics", lambda *args, **kwargs: _diagnostics())
    monkeypatch.setattr(
        live_module,
        "_execute_single_run",
        lambda **kwargs: _passing_run(kwargs["run_index"]),
    )
    run_vllm_prefix_cache_live_proof(
        _proof_config(
            runs=1,
            output_dir=tmp_path,
            manage_vllm=True,
            force_recreate_vllm=True,
            keep_vllm_running=True,
        ),
        adapter=_FakeAdapter(),
        http_client=MagicMock(spec=httpx.Client),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert not any(command[-2:] == ["stop", "vllm"] for command in commands)


def test_subprocess_never_uses_shell_true(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    observed_shell: list[bool] = []

    def command_runner(args, **kwargs):
        observed_shell.append(kwargs.get("shell", False))
        if args[:3] == ["docker", "compose", "config"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args in (["docker", "version"], ["docker", "compose", "version"]):
            return subprocess.CompletedProcess(args, 0, "ok", "")
        if args[:2] == ["nvidia-smi"]:
            return subprocess.CompletedProcess(args, 0, "GPU,24576,550.0", "")
        return subprocess.CompletedProcess(args, 1, "", "")

    run_vllm_prefix_cache_live_proof(
        _proof_config(runs=1, output_dir=tmp_path, manage_vllm=True),
        command_runner=command_runner,
        skip_report_write=True,
    )
    assert observed_shell
    assert all(value is False for value in observed_shell)


def test_main_returns_expected_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    passing = _aggregate_result(
        runs=(_passing_run(1), _passing_run(2), _passing_run(3)),
        canonical_environment=True,
        environment_verified=True,
        proof_gates_passed=True,
        server_lifecycle_mode="managed",
        exit_code=0,
        reason_codes=(),
    )
    monkeypatch.setattr(live_module, "run_vllm_prefix_cache_live_proof", lambda config: passing)
    assert main([]) == 0

    failing = _aggregate_result(
        runs=(_passing_run(1),),
        canonical_environment=False,
        environment_verified=True,
        exit_code=3,
        reason_codes=("WARM_NOT_GREATER_THAN_COLD",),
    )
    monkeypatch.setattr(live_module, "run_vllm_prefix_cache_live_proof", lambda config: failing)
    assert main([]) == 3
