# © Artur Czarnecki. All rights reserved.

"""Reproducible live vLLM prefix-cache qualification runner (TOKEN-10C-LIVE-PROOF-1)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from intergrax.llm_adapters.providers.openai_compat_providers import VllmChatAdapter
from intergrax.llm_adapters.providers.vllm_diagnostics import (
    VLLM_PINNED_VERSION,
    VllmDiagnosticsError,
    VllmDiagnosticsSnapshot,
    collect_vllm_diagnostics,
    derive_vllm_server_root,
    fetch_vllm_health,
    fetch_vllm_metrics,
    fetch_vllm_version,
)
from intergrax.runtime.token_optimization.proofs.vllm_prefix_cache_report import (
    SCHEMA_VERSION,
    TASK_ID,
    VllmPrefixCacheLiveProofAggregateResult,
    VllmPrefixCacheLiveProofAggregateSummary,
    VllmPrefixCacheLiveProofConfiguration,
    VllmPrefixCacheLiveProofEnvironment,
    VllmPrefixCacheLiveProofRunResult,
    dedupe_reason_codes,
    write_proof_artifacts,
)

__all__ = [
    "VllmPrefixCacheLiveProofConfig",
    "VllmPrefixCacheLiveProofAggregateResult",
    "build_default_config",
    "config_from_namespace",
    "main",
    "run_vllm_prefix_cache_live_proof",
]
from intergrax.runtime.token_optimization.vllm_prefix_cache_proof import (
    VllmPrefixCacheProofCaseId,
    VllmPrefixCacheProofCaseObservation,
    VllmPrefixCacheProofResult,
    assemble_proof_case,
    build_proof_prefix_variant,
    evaluate_vllm_prefix_cache_proof,
    materialize_proof_send_payload,
)

_MODULE_ROOT = Path(__file__).resolve()
_REPO_ROOT = _MODULE_ROOT.parents[4]
_COMPOSE_FILE = _REPO_ROOT / "infra" / "docker" / "vllm" / "docker-compose.yml"
_CONTAINER_NAME = "intergrax-vllm"
_CANONICAL_IMAGE = "vllm/vllm-openai:v0.23.0"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "proofs" / "token_optimization" / "vllm_prefix_cache"
_DEFAULT_BASE_URL = "http://127.0.0.1:8100/v1"
_DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

_EXIT_CANONICAL_PASS = 0
_EXIT_ENVIRONMENT_UNAVAILABLE = 2
_EXIT_PROOF_FAILED = 3
_EXIT_INTERNAL_FAILURE = 4

_REASON_GPU_UNAVAILABLE = "GPU_UNAVAILABLE"
_REASON_DOCKER_UNAVAILABLE = "DOCKER_UNAVAILABLE"
_REASON_DOCKER_COMPOSE_UNAVAILABLE = "DOCKER_COMPOSE_UNAVAILABLE"
_REASON_COMPOSE_CONFIG_INVALID = "COMPOSE_CONFIG_INVALID"
_REASON_VLLM_IMAGE_MISMATCH = "VLLM_IMAGE_MISMATCH"
_REASON_VLLM_FLAGS_MISMATCH = "VLLM_FLAGS_MISMATCH"
_REASON_VLLM_STARTUP_TIMEOUT = "VLLM_STARTUP_TIMEOUT"
_REASON_VLLM_HEALTH_FAILED = "VLLM_HEALTH_FAILED"
_REASON_VLLM_VERSION_MISMATCH = "VLLM_VERSION_MISMATCH"
_REASON_VLLM_MODEL_MISMATCH = "VLLM_MODEL_MISMATCH"
_REASON_REQUIRED_METRICS_MISSING = "REQUIRED_METRICS_MISSING"
_REASON_INTERNAL_RUNNER_FAILURE = "INTERNAL_RUNNER_FAILURE"
_REASON_CANONICAL_RUNS_INSUFFICIENT = "CANONICAL_RUNS_INSUFFICIENT"
_REASON_INFERENCE_FAILED = "INFERENCE_FAILED"
_REASON_DIAGNOSTICS_FAILED = "DIAGNOSTICS_FAILED"

_REQUIRED_CONTAINER_FLAGS: tuple[str, ...] = (
    "--enable-prefix-caching",
    "--prefix-caching-hash-algo",
    "sha256",
    "--enable-prompt-tokens-details",
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "hermes",
)


class CommandRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class VllmPrefixCacheLiveProofConfig:
    runs: int
    output_dir: Path
    base_url: str
    model: str
    minimum_prefix_chars: int
    connect_timeout_seconds: float
    read_timeout_seconds: float
    startup_timeout_seconds: float
    manage_vllm: bool
    force_recreate_vllm: bool
    keep_vllm_running: bool


@dataclass(slots=True)
class _ManagedEnvironmentState:
    runner_started_or_recreated: bool = False
    force_recreated: bool = False
    was_running_before: bool = False


def _default_command_runner(
    args: list[str],
    *,
    cwd: str | None = None,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        shell=False,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        check=False,
    )


def _resolve_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


def build_default_config() -> VllmPrefixCacheLiveProofConfig:
    return VllmPrefixCacheLiveProofConfig(
        runs=3,
        output_dir=_DEFAULT_OUTPUT_DIR,
        base_url=_resolve_env("INTERGRAX_DEFAULT_VLLM_BASE_URL", _DEFAULT_BASE_URL),
        model=_resolve_env("INTERGRAX_DEFAULT_VLLM_MODEL", _DEFAULT_MODEL),
        minimum_prefix_chars=4096,
        connect_timeout_seconds=5.0,
        read_timeout_seconds=120.0,
        startup_timeout_seconds=1800.0,
        manage_vllm=False,
        force_recreate_vllm=False,
        keep_vllm_running=False,
    )


def validate_config(config: VllmPrefixCacheLiveProofConfig) -> None:
    if config.runs < 1:
        raise ValueError("--runs must be >= 1")
    if config.minimum_prefix_chars < 512:
        raise ValueError("--minimum-prefix-chars must be >= 512")
    if config.connect_timeout_seconds <= 0:
        raise ValueError("--connect-timeout-seconds must be positive")
    if config.read_timeout_seconds <= 0:
        raise ValueError("--read-timeout-seconds must be positive")
    if config.startup_timeout_seconds <= 0:
        raise ValueError("--startup-timeout-seconds must be positive")
    if not config.model.strip():
        raise ValueError("--model must be nonblank")
    derive_vllm_server_root(config.base_url)


def _resolve_repository_commit(
    *,
    command_runner: CommandRunner,
) -> str | None:
    try:
        result = command_runner(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _command_succeeded(
    command_runner: CommandRunner,
    args: list[str],
    *,
    timeout: float = 10.0,
) -> bool:
    try:
        result = command_runner(args, cwd=str(_REPO_ROOT), timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _collect_gpu_preflight(command_runner: CommandRunner) -> tuple[bool, str | None, int | None, str | None]:
    try:
        result = command_runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False, None, None, None
    if result.returncode != 0:
        return False, None, None, None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return False, None, None, None
    parts = [part.strip() for part in lines[0].split(",")]
    gpu_name = parts[0] if parts else None
    memory_mb: int | None = None
    driver_version: str | None = None
    if len(parts) > 1:
        try:
            memory_mb = int(float(parts[1]))
        except ValueError:
            memory_mb = None
    if len(parts) > 2:
        driver_version = parts[2]
    return True, gpu_name, memory_mb, driver_version


def _container_running(command_runner: CommandRunner) -> bool:
    try:
        result = command_runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                _CONTAINER_NAME,
            ],
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def _compose_config_valid(command_runner: CommandRunner) -> bool:
    return _command_succeeded(
        command_runner,
        [
            "docker",
            "compose",
            "-f",
            str(_COMPOSE_FILE),
            "config",
        ],
        timeout=30.0,
    )


def _compose_up(
    command_runner: CommandRunner,
    *,
    force_recreate: bool,
) -> bool:
    args = [
        "docker",
        "compose",
        "-f",
        str(_COMPOSE_FILE),
        "up",
        "-d",
    ]
    if force_recreate:
        args.append("--force-recreate")
    args.append("vllm")
    return _command_succeeded(command_runner, args, timeout=600.0)


def _compose_stop(command_runner: CommandRunner) -> bool:
    return _command_succeeded(
        command_runner,
        [
            "docker",
            "compose",
            "-f",
            str(_COMPOSE_FILE),
            "stop",
            "vllm",
        ],
        timeout=120.0,
    )


def _inspect_container_image(command_runner: CommandRunner) -> str | None:
    try:
        result = command_runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Config.Image}}",
                _CONTAINER_NAME,
            ],
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    image = result.stdout.strip()
    return image or None


def _inspect_container_command(command_runner: CommandRunner) -> list[str]:
    try:
        result = command_runner(
            [
                "docker",
                "inspect",
                "--format",
                "{{json .Config.Cmd}}",
                _CONTAINER_NAME,
            ],
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    try:
        payload = json.loads(result.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _verify_container_contract(command_runner: CommandRunner) -> tuple[bool, list[str]]:
    reason_codes: list[str] = []
    image = _inspect_container_image(command_runner)
    if image != _CANONICAL_IMAGE:
        reason_codes.append(_REASON_VLLM_IMAGE_MISMATCH)
    command = _inspect_container_command(command_runner)
    if not command:
        reason_codes.append(_REASON_VLLM_FLAGS_MISMATCH)
        return False, reason_codes
    joined = " ".join(command)
    for flag in _REQUIRED_CONTAINER_FLAGS:
        if flag not in command and flag not in joined:
            reason_codes.append(_REASON_VLLM_FLAGS_MISMATCH)
            break
    return not reason_codes, dedupe_reason_codes(reason_codes)


def _classify_diagnostics_error(exc: VllmDiagnosticsError) -> str:
    message = str(exc)
    if "required vLLM metrics missing" in message:
        return _REASON_REQUIRED_METRICS_MISSING
    return _REASON_DIAGNOSTICS_FAILED


def _verify_server_model(
    client: httpx.Client,
    *,
    expected_model: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
) -> bool:
    timeout = httpx.Timeout(
        connect=connect_timeout_seconds,
        read=read_timeout_seconds,
        write=read_timeout_seconds,
        pool=connect_timeout_seconds,
    )
    try:
        response = client.get("/v1/models", timeout=timeout, follow_redirects=False)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return False
    data = payload.get("data")
    if not isinstance(data, list):
        return False
    for item in data:
        if isinstance(item, dict) and item.get("id") == expected_model:
            return True
    return False


def _wait_for_health(
    *,
    base_url: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    startup_timeout_seconds: float,
    http_client: httpx.Client,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[bool, str | None]:
    server_root = derive_vllm_server_root(base_url)
    deadline = monotonic() + startup_timeout_seconds
    while monotonic() < deadline:
        try:
            health = fetch_vllm_health(
                http_client,
                server_root=server_root,
                connect_timeout=connect_timeout_seconds,
                read_timeout=read_timeout_seconds,
            )
            if health.healthy:
                version = fetch_vllm_version(
                    http_client,
                    server_root=server_root,
                    connect_timeout=connect_timeout_seconds,
                    read_timeout=read_timeout_seconds,
                )
                fetch_vllm_metrics(
                    http_client,
                    server_root=server_root,
                    connect_timeout=connect_timeout_seconds,
                    read_timeout=read_timeout_seconds,
                )
                return True, version
        except (VllmDiagnosticsError, httpx.HTTPError):
            pass
        sleep(2.0)
    return False, None


def _execute_warmup(
    *,
    config: VllmPrefixCacheLiveProofConfig,
    adapter: VllmChatAdapter,
    uuid_factory: Callable[[], str],
    http_client: httpx.Client,
) -> None:
    warmup_namespace = f"token-10c-warmup-{uuid_factory()}"
    prefix_variant = build_proof_prefix_variant(
        run_namespace=warmup_namespace,
        variant_suffix="warmup-only",
    )
    assembly = assemble_proof_case(
        case_id=VllmPrefixCacheProofCaseId.COLD,
        prefix_variant=prefix_variant,
        dynamic_tail_text="non-measured warm-up tail",
        minimum_prefix_chars=config.minimum_prefix_chars,
        previous_state=None,
    )
    payload = materialize_proof_send_payload(assembly)
    server_root = derive_vllm_server_root(config.base_url)
    _ = fetch_vllm_metrics(
        http_client,
        server_root=server_root,
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
    )
    if payload.tools_schema:
        _ = adapter.generate_with_tools(
            payload.messages,
            list(payload.tools_schema),
            max_tokens=32,
            run_id="token-10c-warmup",
        )
    else:
        _ = adapter.generate_messages(
            payload.messages,
            max_tokens=32,
            run_id="token-10c-warmup",
        )


def _execute_case(
    *,
    config: VllmPrefixCacheLiveProofConfig,
    adapter: VllmChatAdapter,
    http_client: httpx.Client,
    case_id: VllmPrefixCacheProofCaseId,
    prefix_variant: str,
    dynamic_tail_text: str,
    previous_state,
    monotonic: Callable[[], float],
) -> tuple[VllmPrefixCacheProofCaseObservation, object]:
    assembly = assemble_proof_case(
        case_id=case_id,
        prefix_variant=prefix_variant,
        dynamic_tail_text=dynamic_tail_text,
        minimum_prefix_chars=config.minimum_prefix_chars,
        previous_state=previous_state,
    )
    payload = materialize_proof_send_payload(assembly)
    server_root = derive_vllm_server_root(config.base_url)
    metrics_before = fetch_vllm_metrics(
        http_client,
        server_root=server_root,
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
    )
    started = monotonic()
    if payload.tools_schema:
        response = adapter.generate_with_tools(
            payload.messages,
            list(payload.tools_schema),
            max_tokens=64,
            run_id=f"token-10c-{case_id.value.lower()}",
        )
    else:
        response = adapter.generate_messages(
            payload.messages,
            max_tokens=64,
            run_id=f"token-10c-{case_id.value.lower()}",
        )
    latency_ms = (monotonic() - started) * 1000.0
    metrics_after = fetch_vllm_metrics(
        http_client,
        server_root=server_root,
        connect_timeout=config.connect_timeout_seconds,
        read_timeout=config.read_timeout_seconds,
    )
    usage = response.usage
    if usage is None:
        raise RuntimeError("adapter response missing usage")
    extensions = response.provider_extensions
    prompt_details_reported = bool(
        extensions is not None
        and extensions.vllm is not None
        and extensions.vllm.prompt_tokens_details_reported
    )
    observation = VllmPrefixCacheProofCaseObservation(
        case_id=case_id,
        prefix_hash=assembly.state.prefix_hash,
        tool_envelope_hash=assembly.state.tool_envelope_hash,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        uncached_input_tokens=usage.uncached_input_tokens,
        latency_ms=latency_ms,
        prompt_tokens_details_reported=prompt_details_reported,
        metric_deltas=metrics_after.metric_delta(metrics_before),
    )
    return observation, assembly.state


def _execute_single_run(
    *,
    config: VllmPrefixCacheLiveProofConfig,
    adapter: VllmChatAdapter,
    http_client: httpx.Client,
    run_index: int,
    run_namespace: str,
    diagnostics: VllmDiagnosticsSnapshot,
    uuid_factory: Callable[[], str],
    monotonic: Callable[[], float],
) -> VllmPrefixCacheLiveProofRunResult:
    observations: list[VllmPrefixCacheProofCaseObservation] = []
    previous_state = None
    prefix_a = build_proof_prefix_variant(run_namespace=run_namespace, variant_suffix="proof-a")
    prefix_b = build_proof_prefix_variant(run_namespace=run_namespace, variant_suffix="proof-b")
    case_specs = (
        (VllmPrefixCacheProofCaseId.COLD, prefix_a, f"cold tail {run_index} {uuid_factory()}"),
        (VllmPrefixCacheProofCaseId.WARM, prefix_a, f"warm tail {run_index} {uuid_factory()}"),
        (
            VllmPrefixCacheProofCaseId.CHANGED_PREFIX,
            prefix_b,
            f"changed tail {run_index} {uuid_factory()}",
        ),
    )
    for case_id, prefix_variant, tail_text in case_specs:
        observation, previous_state = _execute_case(
            config=config,
            adapter=adapter,
            http_client=http_client,
            case_id=case_id,
            prefix_variant=prefix_variant,
            dynamic_tail_text=tail_text,
            previous_state=previous_state,
            monotonic=monotonic,
        )
        observations.append(observation)
    proof_result = evaluate_vllm_prefix_cache_proof(
        health_ok=diagnostics.health.healthy,
        server_version=diagnostics.server_version,
        expected_server_version=VLLM_PINNED_VERSION,
        metrics_available=True,
        cases=observations,
    )
    return VllmPrefixCacheLiveProofRunResult(
        run_index=run_index,
        passed=proof_result.passed,
        reason_codes=proof_result.reason_codes,
        server_version=proof_result.server_version,
        health_ok=proof_result.health_ok,
        cases=proof_result.cases,
    )


def _build_aggregate_summary(
    *,
    environment_verified: bool,
    requested_runs: int,
    runs: Sequence[VllmPrefixCacheLiveProofRunResult],
    environment_reason_codes: Sequence[str],
) -> VllmPrefixCacheLiveProofAggregateSummary:
    completed_runs = len(runs)
    passed_runs = sum(1 for run in runs if run.passed)
    failed_runs = completed_runs - passed_runs
    all_runs_passed = completed_runs > 0 and failed_runs == 0
    reason_codes: list[str] = list(environment_reason_codes)
    if requested_runs < 3:
        reason_codes.append(_REASON_CANONICAL_RUNS_INSUFFICIENT)
    for run in runs:
        if not run.passed:
            reason_codes.extend(run.reason_codes)
    if completed_runs < requested_runs:
        reason_codes.append(_REASON_INTERNAL_RUNNER_FAILURE)
    canonical_pass = (
        environment_verified
        and requested_runs >= 3
        and completed_runs == requested_runs
        and all_runs_passed
        and not reason_codes
    )
    return VllmPrefixCacheLiveProofAggregateSummary(
        canonical_pass=canonical_pass,
        all_runs_passed=all_runs_passed,
        requested_runs=requested_runs,
        completed_runs=completed_runs,
        passed_runs=passed_runs,
        failed_runs=failed_runs,
        reason_codes=dedupe_reason_codes(reason_codes),
    )


def _resolve_exit_code(
    *,
    aggregate: VllmPrefixCacheLiveProofAggregateSummary,
    environment_blocked: bool,
    internal_failure: bool,
) -> int:
    if internal_failure:
        return _EXIT_INTERNAL_FAILURE
    if environment_blocked:
        return _EXIT_ENVIRONMENT_UNAVAILABLE
    if aggregate.canonical_pass:
        return _EXIT_CANONICAL_PASS
    if aggregate.completed_runs > 0:
        return _EXIT_PROOF_FAILED
    return _EXIT_ENVIRONMENT_UNAVAILABLE


def run_vllm_prefix_cache_live_proof(
    config: VllmPrefixCacheLiveProofConfig,
    *,
    adapter: VllmChatAdapter | None = None,
    command_runner: CommandRunner | None = None,
    uuid_factory: Callable[[], str] | None = None,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    utc_now: Callable[[], datetime] | None = None,
    repository_commit_resolver: Callable[[], str | None] | None = None,
    http_client: httpx.Client | None = None,
    skip_report_write: bool = False,
) -> VllmPrefixCacheLiveProofAggregateResult:
    validate_config(config)
    runner = command_runner or _default_command_runner
    uuid_gen = uuid_factory or (lambda: str(uuid.uuid4()))
    clock = monotonic or time.perf_counter
    sleeper = sleep or time.sleep
    now_factory = utc_now or (lambda: datetime.now(UTC))
    commit_resolver = repository_commit_resolver or (lambda: _resolve_repository_commit(command_runner=runner))

    started_at = now_factory()
    environment_reason_codes: list[str] = []
    environment_blocked = False
    internal_failure = False
    exclusive_environment = False
    environment_verified = False
    exclusive_environment_expected = config.manage_vllm and config.force_recreate_vllm
    managed_state = _ManagedEnvironmentState()
    gpu_available, gpu_name, gpu_memory_mb, driver_version = _collect_gpu_preflight(runner)
    docker_available = _command_succeeded(runner, ["docker", "version"], timeout=10.0)
    docker_compose_available = _command_succeeded(
        runner,
        ["docker", "compose", "version"],
        timeout=10.0,
    )
    compose_contract_valid = _compose_config_valid(runner) if docker_compose_available else False

    vllm_image: str | None = None
    vllm_version: str | None = None
    health_ok = False
    run_results: list[VllmPrefixCacheLiveProofRunResult] = []
    owns_client = http_client is None
    client = http_client
    adapter_instance = adapter
    json_report_path: Path | None = None
    markdown_report_path: Path | None = None

    try:
        if config.manage_vllm:
            if not gpu_available:
                environment_reason_codes.append(_REASON_GPU_UNAVAILABLE)
                environment_blocked = True
            if not docker_available:
                environment_reason_codes.append(_REASON_DOCKER_UNAVAILABLE)
                environment_blocked = True
            if not docker_compose_available:
                environment_reason_codes.append(_REASON_DOCKER_COMPOSE_UNAVAILABLE)
                environment_blocked = True
            if not compose_contract_valid:
                environment_reason_codes.append(_REASON_COMPOSE_CONFIG_INVALID)
                environment_blocked = True

            if not environment_blocked:
                managed_state.was_running_before = _container_running(runner)
                should_recreate = config.force_recreate_vllm
                should_start = should_recreate or not managed_state.was_running_before
                if should_start:
                    if not _compose_up(runner, force_recreate=should_recreate):
                        environment_reason_codes.append(_REASON_INTERNAL_RUNNER_FAILURE)
                        environment_blocked = True
                    else:
                        managed_state.runner_started_or_recreated = True
                        managed_state.force_recreated = should_recreate

        exclusive_environment = (
            config.manage_vllm
            and config.force_recreate_vllm
            and managed_state.force_recreated
            and not environment_blocked
        )

        if environment_blocked:
            pass
        else:
            server_root = derive_vllm_server_root(config.base_url)
            if client is None:
                client = httpx.Client(base_url=server_root)
            if adapter_instance is None:
                adapter_instance = VllmChatAdapter(base_url=config.base_url, model=config.model)

            if config.manage_vllm and managed_state.runner_started_or_recreated:
                ready, version = _wait_for_health(
                    base_url=config.base_url,
                    connect_timeout_seconds=config.connect_timeout_seconds,
                    read_timeout_seconds=config.read_timeout_seconds,
                    startup_timeout_seconds=config.startup_timeout_seconds,
                    http_client=client,
                    monotonic=clock,
                    sleep=sleeper,
                )
                if not ready:
                    environment_reason_codes.append(_REASON_VLLM_STARTUP_TIMEOUT)
                    environment_blocked = True
                else:
                    vllm_version = version
                    health_ok = True
                    if vllm_version != VLLM_PINNED_VERSION:
                        environment_reason_codes.append(_REASON_VLLM_VERSION_MISMATCH)
                        environment_blocked = True
            else:
                try:
                    diagnostics = collect_vllm_diagnostics(
                        config.base_url,
                        connect_timeout=config.connect_timeout_seconds,
                        read_timeout=config.read_timeout_seconds,
                        http_client=client,
                    )
                except VllmDiagnosticsError as exc:
                    environment_reason_codes.append(_classify_diagnostics_error(exc))
                    environment_blocked = True
                else:
                    vllm_version = diagnostics.server_version
                    health_ok = diagnostics.health.healthy
                    if not health_ok:
                        environment_reason_codes.append(_REASON_VLLM_HEALTH_FAILED)
                        environment_blocked = True
                    elif vllm_version != VLLM_PINNED_VERSION:
                        environment_reason_codes.append(_REASON_VLLM_VERSION_MISMATCH)
                        environment_blocked = True

            if not environment_blocked and not _verify_server_model(
                client,
                expected_model=config.model,
                connect_timeout_seconds=config.connect_timeout_seconds,
                read_timeout_seconds=config.read_timeout_seconds,
            ):
                environment_reason_codes.append(_REASON_VLLM_MODEL_MISMATCH)
                environment_blocked = True

            if not environment_blocked and config.manage_vllm and managed_state.runner_started_or_recreated:
                contract_ok, contract_reasons = _verify_container_contract(runner)
                compose_contract_valid = contract_ok
                environment_reason_codes.extend(contract_reasons)
                if not contract_ok:
                    environment_blocked = True
                vllm_image = _inspect_container_image(runner)
            elif not environment_blocked:
                vllm_image = _inspect_container_image(runner)

            if not environment_blocked:
                environment_verified = True
                try:
                    _execute_warmup(
                        config=config,
                        adapter=adapter_instance,
                        uuid_factory=uuid_gen,
                        http_client=client,
                    )
                    diagnostics = collect_vllm_diagnostics(
                        config.base_url,
                        connect_timeout=config.connect_timeout_seconds,
                        read_timeout=config.read_timeout_seconds,
                        http_client=client,
                    )
                    health_ok = diagnostics.health.healthy
                    vllm_version = diagnostics.server_version
                except VllmDiagnosticsError as exc:
                    environment_reason_codes.append(_classify_diagnostics_error(exc))
                    environment_blocked = True
                    environment_verified = False
                except RuntimeError:
                    environment_reason_codes.append(_REASON_INFERENCE_FAILED)
                    environment_blocked = True
                    environment_verified = False

                if not environment_blocked:
                    for run_index in range(1, config.runs + 1):
                        run_namespace = f"token-10c-{uuid_gen()}"
                        try:
                            run_result = _execute_single_run(
                                config=config,
                                adapter=adapter_instance,
                                http_client=client,
                                run_index=run_index,
                                run_namespace=run_namespace,
                                diagnostics=diagnostics,
                                uuid_factory=uuid_gen,
                                monotonic=clock,
                            )
                        except (VllmDiagnosticsError, RuntimeError):
                            environment_reason_codes.append(_REASON_INFERENCE_FAILED)
                            internal_failure = True
                            break
                        run_results.append(run_result)

        finished_at = now_factory()
        configuration = VllmPrefixCacheLiveProofConfiguration(
            requested_runs=config.runs,
            minimum_prefix_chars=config.minimum_prefix_chars,
            connect_timeout_seconds=config.connect_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
            startup_timeout_seconds=config.startup_timeout_seconds,
        )
        proof_gates_passed = (
            len(run_results) == config.runs
            and all(run.passed for run in run_results)
            and bool(run_results)
        )
        environment = VllmPrefixCacheLiveProofEnvironment(
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_total_mb=gpu_memory_mb,
            nvidia_driver_version=driver_version,
            docker_available=docker_available,
            docker_compose_available=docker_compose_available,
            compose_contract_valid=compose_contract_valid,
            vllm_image=vllm_image,
            vllm_version=vllm_version,
            model=config.model,
            health_ok=health_ok,
            managed_environment=config.manage_vllm,
            force_recreated=managed_state.force_recreated,
            exclusive_environment_expected=exclusive_environment_expected,
            server_lifecycle_mode="managed" if config.manage_vllm else "shared",
            server_started_by_runner=managed_state.runner_started_or_recreated,
            environment_verified=environment_verified,
            proof_gates_passed=proof_gates_passed,
        )
        aggregate = _build_aggregate_summary(
            environment_verified=environment_verified and compose_contract_valid,
            requested_runs=config.runs,
            runs=run_results,
            environment_reason_codes=environment_reason_codes,
        )
        exit_code = _resolve_exit_code(
            aggregate=aggregate,
            environment_blocked=environment_blocked and not run_results,
            internal_failure=internal_failure,
        )
        result = VllmPrefixCacheLiveProofAggregateResult(
            schema_version=SCHEMA_VERSION,
            task_id=TASK_ID,
            started_at_utc=started_at.strftime("%Y%m%dT%H%M%SZ"),
            finished_at_utc=finished_at.strftime("%Y%m%dT%H%M%SZ"),
            repository_commit=commit_resolver(),
            canonical_environment=exclusive_environment and compose_contract_valid,
            environment=environment,
            configuration=configuration,
            runs=tuple(run_results),
            aggregate=aggregate,
            exit_code=exit_code,
        )
        if not skip_report_write and (run_results or environment_reason_codes or environment_blocked):
            try:
                json_report_path, markdown_report_path = write_proof_artifacts(
                    result,
                    output_dir=config.output_dir,
                    timestamp_utc=result.finished_at_utc,
                )
                result = replace(
                    result,
                    json_report_path=json_report_path,
                    markdown_report_path=markdown_report_path,
                )
            except Exception:
                internal_failure = True
                result = replace(result, exit_code=_EXIT_INTERNAL_FAILURE)
        return result
    finally:
        if (
            config.manage_vllm
            and managed_state.runner_started_or_recreated
            and not config.keep_vllm_running
        ):
            _compose_stop(runner)
        if owns_client and client is not None:
            client.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = build_default_config()
    parser = argparse.ArgumentParser(
        description="Reproducible vLLM prefix-cache live proof (TOKEN-10C-LIVE-PROOF-1)",
    )
    parser.add_argument("--runs", type=int, default=defaults.runs)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--base-url", default=defaults.base_url)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--minimum-prefix-chars", type=int, default=defaults.minimum_prefix_chars)
    parser.add_argument("--connect-timeout-seconds", type=float, default=defaults.connect_timeout_seconds)
    parser.add_argument("--read-timeout-seconds", type=float, default=defaults.read_timeout_seconds)
    parser.add_argument("--startup-timeout-seconds", type=float, default=defaults.startup_timeout_seconds)
    parser.add_argument("--manage-vllm", action="store_true", default=defaults.manage_vllm)
    parser.add_argument("--force-recreate-vllm", action="store_true", default=defaults.force_recreate_vllm)
    parser.add_argument("--keep-vllm-running", action="store_true", default=defaults.keep_vllm_running)
    return parser


def config_from_namespace(namespace: argparse.Namespace) -> VllmPrefixCacheLiveProofConfig:
    return VllmPrefixCacheLiveProofConfig(
        runs=namespace.runs,
        output_dir=namespace.output_dir,
        base_url=namespace.base_url,
        model=namespace.model,
        minimum_prefix_chars=namespace.minimum_prefix_chars,
        connect_timeout_seconds=namespace.connect_timeout_seconds,
        read_timeout_seconds=namespace.read_timeout_seconds,
        startup_timeout_seconds=namespace.startup_timeout_seconds,
        manage_vllm=namespace.manage_vllm,
        force_recreate_vllm=namespace.force_recreate_vllm,
        keep_vllm_running=namespace.keep_vllm_running,
    )


def _print_terminal_summary(result: VllmPrefixCacheLiveProofAggregateResult) -> None:
    status = "PASS" if result.aggregate.canonical_pass else "FAIL"
    print(f"final status: {status}")
    print(
        "pass count: "
        f"{result.aggregate.passed_runs}/{result.aggregate.requested_runs}"
    )
    if result.aggregate.reason_codes:
        print(f"failure reason codes: {', '.join(result.aggregate.reason_codes)}")
    else:
        print("failure reason codes: none")
    if result.json_report_path is not None:
        print(f"json report: {result.json_report_path}")
    if result.markdown_report_path is not None:
        print(f"markdown report: {result.markdown_report_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    try:
        namespace = parser.parse_args(argv)
        config = config_from_namespace(namespace)
        result = run_vllm_prefix_cache_live_proof(config)
    except ValueError:
        return _EXIT_INTERNAL_FAILURE
    except Exception:
        return _EXIT_INTERNAL_FAILURE
    _print_terminal_summary(result)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "VllmPrefixCacheLiveProofConfig",
    "VllmPrefixCacheLiveProofAggregateResult",
    "build_default_config",
    "config_from_namespace",
    "main",
    "run_vllm_prefix_cache_live_proof",
]
