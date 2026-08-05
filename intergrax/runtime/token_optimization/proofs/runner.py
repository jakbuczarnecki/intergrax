# © Artur Czarnecki. All rights reserved.

"""Backend-neutral composition, execution and artifact persistence for TOKEN-10F."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import (
    LLMAdapter,
    LLMAdapterResponse,
)
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.runtime.token_optimization.builtin_catalog import (
    BuiltInTokenOptimizationLayerCatalog,
    create_builtin_token_optimization_layer_catalog,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationLayerRef,
    TokenOptimizationPipelineConfig,
    TokenOptimizationPipelineMode,
)
from intergrax.runtime.token_optimization.llm_router import TokenOptimizationLLMRouter
from intergrax.runtime.token_optimization.llm_router_catalog import (
    TokenOptimizationRouterConfigurationCatalog,
    create_token_optimization_router_configuration_catalog,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterTransport,
)
from intergrax.runtime.token_optimization.llm_router import (
    token_optimization_router_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner
from intergrax.runtime.token_optimization.proofs.contracts import (
    ProofArtifactError,
    ProofArtifactRef,
    ProofCompositionError,
    ProofExecutionError,
    ProofError,
    ProofMeasurement,
    ProofPipelineEvidence,
    ProofPrefixIdentityEvidence,
    ProofProtectedRegionEvidence,
    ProofProviderUnavailableError,
    ProofRouterEvidence,
    SCHEMA_VERSION,
    UniversalProofArtifactManifest,
    UniversalProofCaseResult,
    UniversalProofEnvironmentSummary,
    UniversalProofRunResult,
    UniversalTokenOptimizationProofConfig,
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ROUTER_REASON_CODES = {
    "no_optimization": TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
    "exact_only": TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
    "extractive_only": TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
    "packing_only": TokenOptimizationRouterReasonCode.PRIORITY_PACKING,
    "exact_then_packing": TokenOptimizationRouterReasonCode.MIXED_DEDUPLICATION_PACKING,
    "exact_then_extractive": TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
    "extractive_then_exact": TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
}
_ROUTER_RISK = TokenOptimizationRouterRisk.LOW


def _safe_id(value: str, reason_code: str) -> str:
    if not value or value != value.strip() or not _SAFE_ID_RE.fullmatch(value):
        raise ProofExecutionError(reason_code)
    return value


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProofArtifactError("NON_CANONICAL_JSON") from exc
    return (text + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise ProofArtifactError("ATOMIC_WRITE_FAILED") from exc


def _measurements_from_pipeline(
    result: object,
) -> tuple[ProofMeasurement, ProofMeasurement]:
    measurement = getattr(result, "aggregate_measurement", None)
    if measurement is None:
        return ProofMeasurement(), ProofMeasurement()
    baseline = getattr(measurement, "baseline_tokens", None)
    optimized = getattr(measurement, "optimized_tokens", None)
    baseline_measurement = (
        ProofMeasurement(available=True, value=baseline)
        if type(baseline) is int and baseline >= 0
        else ProofMeasurement()
    )
    optimized_measurement = (
        ProofMeasurement(available=True, value=optimized)
        if type(optimized) is int and optimized >= 0
        else ProofMeasurement()
    )
    return baseline_measurement, optimized_measurement


def _router_evidence(result) -> ProofRouterEvidence:
    return ProofRouterEvidence(
        status=result.status.value,
        configuration_id=(
            result.configuration_id.value
            if result.configuration_id is not None
            else None
        ),
        reason_code=result.reason_code.value if result.reason_code is not None else None,
        review_required=result.review_required,
        confidence=result.confidence,
        risk=result.risk.value if result.risk is not None else None,
        transport=result.transport.value,
        structured_output_fallback_used=(
            result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT
        ),
    )


def _prefix_identity_evidence(report) -> ProofPrefixIdentityEvidence:
    if report is None:
        return ProofPrefixIdentityEvidence()
    return ProofPrefixIdentityEvidence(
        identity_available=True,
        stable_prefix_identity=report.prefix_hash,
        tool_schema_hash=report.tool_envelope_hash,
        identity_contract_version="TOKEN-10B",
    )


def _protected_identity_digest(regions) -> str | None:
    if not regions:
        return None
    digest = hashlib.sha256()
    for ordinal, region in enumerate(regions):
        value_digest = hashlib.sha256(region.value.encode("utf-8")).hexdigest()
        digest.update(region.kind.value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(ordinal).encode("ascii"))
        digest.update(b"\0")
        digest.update(value_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _latest_validation(pipeline_result):
    if pipeline_result is None:
        return None
    for layer_result in reversed(pipeline_result.layer_results):
        if layer_result.validation is not None:
            return layer_result.validation
    return None


def _safe_receipt_code(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value):
        return value
    return None


def _pipeline_evidence(pipeline_result) -> ProofPipelineEvidence:
    if pipeline_result is None:
        return ProofPipelineEvidence()
    completed = pipeline_result.receipt_metadata.get("completed")
    completed = completed if type(completed) is bool else None
    required_failure = pipeline_result.receipt_metadata.get(
        "required_failure_layer_id"
    )
    required_failure = required_failure if isinstance(required_failure, str) else None
    validation = _latest_validation(pipeline_result)
    receipt_validation_status = _safe_receipt_code(
        pipeline_result.receipt_metadata.get("validation_status")
    )
    validation_status = (
        receipt_validation_status
        if receipt_validation_status is not None
        else validation.status.value if validation is not None else None
    )
    receipt_validation_reason = _safe_receipt_code(
        pipeline_result.receipt_metadata.get("validation_reason_code")
    )
    validation_reason_code = (
        receipt_validation_reason
        if isinstance(receipt_validation_reason, str)
        else "VALIDATION_FAILED"
        if validation_status == "failed"
        else None
    )
    return ProofPipelineEvidence(
        completed=completed,
        fallback_applied=pipeline_result.fallback_used,
        validation_status=validation_status,
        validation_reason_code=validation_reason_code,
        required_layer_failure=required_failure,
        receipt_completion_status=completed,
    )


def _protected_region_evidence(case_request, pipeline_result) -> ProofProtectedRegionEvidence:
    regions = tuple(case_request.protected_regions)
    input_count = len(regions)
    input_digest = _protected_identity_digest(regions)
    validation = _latest_validation(pipeline_result)
    if input_count == 0:
        return ProofProtectedRegionEvidence(
            protected_region_validation_status="not_applicable",
        )
    if validation is None:
        return ProofProtectedRegionEvidence(
            input_protected_region_count=input_count,
            protected_region_validation_status="not_run",
            input_identity_digest=input_digest,
        )
    status = validation.status.value
    preserved_digest = (
        input_digest
        if status == "passed"
        and validation.regions_preserved == input_count
        else None
    )
    return ProofProtectedRegionEvidence(
        input_protected_region_count=input_count,
        validated_protected_region_count=validation.regions_checked,
        preserved_protected_region_count=validation.regions_preserved,
        protected_region_validation_status=status,
        input_identity_digest=input_digest,
        preserved_identity_digest=preserved_digest,
    )


def _case_to_dict(result: UniversalProofCaseResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": result.case_id,
        "status": result.status,
        "router_status": result.router_status,
        "router_reason": result.router_reason,
        "selected_configuration_id": result.selected_configuration_id,
        "pipeline_status": result.pipeline_status,
        "applied_layer_ids": list(result.applied_layer_ids),
        "baseline_measurement": {
            "available": result.baseline_measurement.available,
            "value": result.baseline_measurement.value,
        },
        "optimized_measurement": {
            "available": result.optimized_measurement.available,
            "value": result.optimized_measurement.value,
        },
        "receipt_refs": list(result.receipt_refs),
        "error_reason_code": result.error_reason_code,
        "router_evidence": {
            "status": result.router_evidence.status,
            "configuration_id": result.router_evidence.configuration_id,
            "reason_code": result.router_evidence.reason_code,
            "review_required": result.router_evidence.review_required,
            "confidence": result.router_evidence.confidence,
            "risk": result.router_evidence.risk,
            "transport": result.router_evidence.transport,
            "structured_output_fallback_used": (
                result.router_evidence.structured_output_fallback_used
            ),
        },
        "pipeline_evidence": {
            "completed": result.pipeline_evidence.completed,
            "fallback_applied": result.pipeline_evidence.fallback_applied,
            "validation_status": result.pipeline_evidence.validation_status,
            "validation_reason_code": result.pipeline_evidence.validation_reason_code,
            "required_layer_failure": result.pipeline_evidence.required_layer_failure,
            "receipt_completion_status": (
                result.pipeline_evidence.receipt_completion_status
            ),
        },
        "protected_region_evidence": {
            "input_protected_region_count": (
                result.protected_region_evidence.input_protected_region_count
            ),
            "validated_protected_region_count": (
                result.protected_region_evidence.validated_protected_region_count
            ),
            "preserved_protected_region_count": (
                result.protected_region_evidence.preserved_protected_region_count
            ),
            "protected_region_validation_status": (
                result.protected_region_evidence.protected_region_validation_status
            ),
            "input_identity_digest": (
                result.protected_region_evidence.input_identity_digest
            ),
            "preserved_identity_digest": (
                result.protected_region_evidence.preserved_identity_digest
            ),
        },
        "prefix_identity_evidence": {
            "identity_available": result.prefix_identity_evidence.identity_available,
            "stable_prefix_identity": (
                result.prefix_identity_evidence.stable_prefix_identity
            ),
            "tool_schema_hash": result.prefix_identity_evidence.tool_schema_hash,
            "message_envelope_hash": (
                result.prefix_identity_evidence.message_envelope_hash
            ),
            "identity_contract_version": (
                result.prefix_identity_evidence.identity_contract_version
            ),
        },
        "raw_content_included": False,
    }


def _run_to_dict(result: UniversalProofRunResult) -> dict[str, Any]:
    return {
        "schema_version": result.schema_version,
        "proof_id": result.proof_id,
        "run_id": result.run_id,
        "run_mode": result.run_mode,
        "started_at": result.started_at.isoformat(),
        "completed_at": result.completed_at.isoformat(),
        "adapter_id": result.adapter_id,
        "model": result.model,
        "case_count": result.case_count,
        "completed_count": result.completed_count,
        "failed_count": result.failed_count,
        "cases": [_case_to_dict(case) for case in result.cases],
        "environment": {
            "provider": result.environment.provider,
            "model": result.environment.model,
            "adapter_available": result.environment.adapter_available,
            "network_required": result.environment.network_required,
            "raw_content_included": False,
        },
        "artifact_manifest": {
            "files": [
                {"path": item.path, "sha256": item.sha256}
                for item in result.artifact_manifest.files
            ],
            "raw_content_included": False,
        },
        "success": result.success,
        "raw_content_included": False,
    }


def _manifest_to_dict(
    result: UniversalProofRunResult,
    files: tuple[ProofArtifactRef, ...],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "proof_id": result.proof_id,
        "run_id": result.run_id,
        "files": [{"path": item.path, "sha256": item.sha256} for item in files],
        "raw_content_included": False,
    }


def write_universal_proof_artifacts(
    result: UniversalProofRunResult,
    *,
    output_directory: Path,
    fail_if_exists: bool = True,
) -> UniversalProofArtifactManifest:
    """Write canonical run, case and manifest JSON with atomic replacement."""
    output_directory = _resolve_output_directory(output_directory)
    run_directory = output_directory / result.proof_id / result.run_id
    if run_directory.exists() and fail_if_exists:
        raise ProofArtifactError("RUN_DIRECTORY_EXISTS")
    if run_directory.exists():
        shutil.rmtree(run_directory)
    run_directory.mkdir(parents=True, exist_ok=False)
    try:
        case_refs: list[ProofArtifactRef] = []
        cases_directory = run_directory / "cases"
        cases_directory.mkdir()
        for case in result.cases:
            relative_path = f"cases/{case.case_id}.json"
            case_path = cases_directory / f"{case.case_id}.json"
            _atomic_write(case_path, _json_bytes(_case_to_dict(case)))
            case_refs.append(
                ProofArtifactRef(
                    path=relative_path,
                    sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
                )
            )

        run_ref = ProofArtifactRef(path="run.json")
        manifest = UniversalProofArtifactManifest(files=(run_ref, *case_refs))
        persisted_result = replace(result, artifact_manifest=manifest)
        run_path = run_directory / "run.json"
        _atomic_write(run_path, _json_bytes(_run_to_dict(persisted_result)))
        complete_refs = (
            ProofArtifactRef(
                path="run.json",
                sha256=hashlib.sha256(run_path.read_bytes()).hexdigest(),
            ),
            *case_refs,
        )
        manifest_path = run_directory / "manifest.json"
        _atomic_write(
            manifest_path,
            _json_bytes(_manifest_to_dict(result, complete_refs)),
        )
        return UniversalProofArtifactManifest(files=complete_refs)
    except ProofArtifactError:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise ProofArtifactError("ARTIFACT_PERSISTENCE_FAILED") from exc


def _resolve_output_directory(output_directory: Path) -> Path:
    raw_output_directory = os.fspath(output_directory)
    if "\x00" in raw_output_directory:
        raise ProofArtifactError("UNSAFE_OUTPUT_DIRECTORY")
    try:
        resolved = Path(output_directory).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ProofArtifactError("UNSAFE_OUTPUT_DIRECTORY") from exc
    if resolved.exists() and not resolved.is_dir():
        raise ProofArtifactError("OUTPUT_DIRECTORY_IS_FILE")
    return resolved


def _new_offline_registry() -> type[LLMAdapterRegistry]:
    class _OfflineScopedAdapterRegistry(LLMAdapterRegistry):
        _factories = {}

    return _OfflineScopedAdapterRegistry


def _adapter_create_kwargs(
    config: UniversalTokenOptimizationProofConfig,
    *,
    api_key: str | None,
) -> dict[str, Any]:
    if config.adapter.adapter_type != "openai_compatible":
        raise ProofCompositionError("UNSUPPORTED_ADAPTER_TYPE")
    return {
        "model": config.adapter.model,
        "base_url": config.adapter.base_url,
        "api_key": api_key,
        "timeout_sec": config.adapter.timeout_seconds,
        "max_tokens": config.adapter.max_output_tokens,
        "temperature": config.adapter.temperature,
    }


class _OfflineSmokeAdapter(LLMAdapter):
    """Deterministic structured-output adapter used only by offline_smoke."""

    def __init__(
        self,
        *,
        model: str,
        configuration_id: str,
        provider: LLMProvider,
    ) -> None:
        super().__init__()
        self.provider = provider
        self.model = model
        self._configuration_id = TokenOptimizationRouterConfigurationId(configuration_id)
        self.model_name_for_token_estimation = model

    @property
    def context_window_tokens(self) -> int:
        return 32768

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return LLMAdapterResponse(
            content="offline_smoke",
            model=self.model,
            provider=self.provider.value,
        )

    def generate_structured(
        self,
        messages: list[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        decision = output_model(
            configuration_id=self._configuration_id,
            reason_code=_ROUTER_REASON_CODES[self._configuration_id.value],
            risk=_ROUTER_RISK,
            review_required=False,
            confidence=1.0,
        )
        return LLMStructuredResult(
            parsed=decision,
            response=LLMAdapterResponse(
                content="offline_smoke",
                model=self.model,
                provider=self.provider.value,
            ),
        )


@dataclass(frozen=True, slots=True)
class _Composition:
    adapter: LLMAdapter
    router_catalog: TokenOptimizationRouterConfigurationCatalog
    builtin_catalog: BuiltInTokenOptimizationLayerCatalog


class UniversalTokenOptimizationProofRunner:
    """Run real router/catalog/registry/pipeline components for safe proof cases."""

    def __init__(
        self,
        *,
        adapter_registry: type[LLMAdapterRegistry] = LLMAdapterRegistry,
        router_factory: Callable[..., TokenOptimizationLLMRouter] = TokenOptimizationLLMRouter,
        router_catalog_factory: Callable[
            [], TokenOptimizationRouterConfigurationCatalog
        ] = create_token_optimization_router_configuration_catalog,
        builtin_catalog_factory: Callable[
            [], BuiltInTokenOptimizationLayerCatalog
        ] = create_builtin_token_optimization_layer_catalog,
        pipeline_runner_factory: Callable[..., TokenOptimizationPipelineRunner] = (
            TokenOptimizationPipelineRunner
        ),
        clock: Callable[[], datetime] | None = None,
        run_id_factory: Callable[[], str] | None = None,
        artifact_writer: Callable[..., UniversalProofArtifactManifest] = (
            write_universal_proof_artifacts
        ),
    ) -> None:
        self._adapter_registry = adapter_registry
        self._router_factory = router_factory
        self._router_catalog_factory = router_catalog_factory
        self._builtin_catalog_factory = builtin_catalog_factory
        self._pipeline_runner_factory = pipeline_runner_factory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_id_factory = run_id_factory or (lambda: uuid.uuid4().hex)
        self._artifact_writer = artifact_writer

    def _compose(self, config: UniversalTokenOptimizationProofConfig) -> _Composition:
        registry = (
            _new_offline_registry()
            if config.run_mode == "offline_smoke"
            else self._adapter_registry
        )
        if config.run_mode == "offline_smoke":
            try:
                TokenOptimizationRouterConfigurationId(
                    config.router.configuration_id
                )
            except ValueError as exc:
                raise ProofCompositionError("UNKNOWN_ROUTER_CONFIGURATION") from exc
            provider = LLMProvider(config.adapter.provider)
            registry.register(
                config.adapter.provider,
                lambda **_: _OfflineSmokeAdapter(
                    model=config.adapter.model,
                    configuration_id=config.router.configuration_id,
                    provider=provider,
                ),
            )
            create_kwargs: dict[str, Any] = {"model": config.adapter.model}
        else:
            api_key = (
                os.environ.get(config.adapter.api_key_env)
                if config.adapter.api_key_env
                else None
            )
            if config.adapter.api_key_env and not api_key:
                raise ProofProviderUnavailableError("MISSING_API_KEY_ENV")
            create_kwargs = _adapter_create_kwargs(config, api_key=api_key)
        try:
            adapter = registry.create(config.adapter.provider, **create_kwargs)
        except Exception as exc:
            if config.run_mode == "live_adapter":
                raise ProofProviderUnavailableError("PROVIDER_UNAVAILABLE") from exc
            raise ProofCompositionError("OFFLINE_ADAPTER_UNAVAILABLE") from exc
        return _Composition(
            adapter=adapter,
            router_catalog=self._router_catalog_factory(),
            builtin_catalog=self._builtin_catalog_factory(),
        )

    def _build_pipeline_config(
        self,
        *,
        config: UniversalTokenOptimizationProofConfig,
        selected_id: TokenOptimizationRouterConfigurationId,
        catalog: TokenOptimizationRouterConfigurationCatalog,
    ) -> TokenOptimizationPipelineConfig:
        compiled = catalog.compile(selected_id)
        layer_refs = compiled.layer_refs
        if config.pipeline.layer_ids:
            layer_refs = tuple(
                TokenOptimizationLayerRef(layer_id=layer_id)
                for layer_id in config.pipeline.layer_ids
            )
        return TokenOptimizationPipelineConfig(
            pipeline_id=f"proof.{selected_id.value}",
            mode=TokenOptimizationPipelineMode(config.pipeline.mode),
            layers=layer_refs,
        )

    def _run_case(
        self,
        *,
        config: UniversalTokenOptimizationProofConfig,
        case,
        composition: _Composition,
        run_id: str,
    ) -> UniversalProofCaseResult:
        request_id = f"{run_id}.{case.case_id}"
        router_policy = TokenOptimizationLLMRouterPolicy(
            allow_structured_output_fallback=(
                config.router.allow_structured_output_fallback
            ),
            require_review_for_protected_lossy_content=(
                config.router.require_review_for_protected_lossy_content
            ),
            minimum_confidence=config.router.minimum_confidence,
        )
        if config.router.enabled:
            router = self._router_factory(
                adapter=composition.adapter,
                catalog=composition.router_catalog,
            )
            routed = router.route(
                TokenOptimizationLLMRouterRequest(
                    request=case.request,
                    policy=router_policy,
                    request_id=request_id,
                )
            )
            router_evidence = _router_evidence(routed)
            safe_router = token_optimization_router_result_to_safe_dict(routed)
            router_status = str(safe_router["status"])
            router_reason = safe_router["reason"]
            selected_raw = safe_router["configuration_id"]
            if router_status not in {
                TokenOptimizationRouterStatus.ROUTED.value,
                TokenOptimizationRouterStatus.NO_OPTIMIZATION.value,
            } or not isinstance(selected_raw, str):
                return UniversalProofCaseResult(
                    case_id=case.case_id,
                    status="failed",
                    router_status=router_status,
                    router_reason=router_reason,
                    selected_configuration_id=(
                        selected_raw if isinstance(selected_raw, str) else None
                    ),
                    pipeline_status="not_started",
                    error_reason_code="ROUTER_EXECUTION_FAILED",
                    router_evidence=router_evidence,
                    protected_region_evidence=_protected_region_evidence(
                        case.request,
                        None,
                    ),
                    prefix_identity_evidence=_prefix_identity_evidence(
                        routed.prompt_assembly_report
                    ),
                )
            selected_id = TokenOptimizationRouterConfigurationId(selected_raw)
        else:
            selected_id = TokenOptimizationRouterConfigurationId(
                config.router.configuration_id
            )
            router_status = "disabled"
            router_reason = None
            router_evidence = ProofRouterEvidence(status="disabled")

        if composition.router_catalog.get(selected_id) is None:
            raise ProofCompositionError("UNKNOWN_ROUTER_CONFIGURATION")
        pipeline_config = self._build_pipeline_config(
            config=config,
            selected_id=selected_id,
            catalog=composition.router_catalog,
        )
        compiled = composition.router_catalog.compile(selected_id)
        registry = composition.builtin_catalog.create_registry(compiled.selections)
        pipeline_runner = self._pipeline_runner_factory(registry=registry)
        try:
            pipeline_result = pipeline_runner.run(
                request=case.request,
                config=pipeline_config,
            )
        except Exception as exc:
            raise ProofExecutionError("PIPELINE_EXECUTION_FAILED") from exc
        completed = pipeline_result.receipt_metadata.get("completed") is True
        baseline_measurement, optimized_measurement = _measurements_from_pipeline(
            pipeline_result
        )
        return UniversalProofCaseResult(
            case_id=case.case_id,
            status="completed" if completed else "failed",
            router_status=router_status,
            router_reason=router_reason,
            selected_configuration_id=selected_id.value,
            pipeline_status="completed" if completed else "failed",
            applied_layer_ids=tuple(pipeline_result.applied_layer_ids),
            baseline_measurement=baseline_measurement,
            optimized_measurement=optimized_measurement,
            error_reason_code=None if completed else "PIPELINE_INCOMPLETE",
            router_evidence=router_evidence,
            pipeline_evidence=_pipeline_evidence(pipeline_result),
            protected_region_evidence=_protected_region_evidence(
                case.request,
                pipeline_result,
            ),
            prefix_identity_evidence=(
                _prefix_identity_evidence(routed.prompt_assembly_report)
                if config.router.enabled
                else ProofPrefixIdentityEvidence()
            ),
        )

    def run(
        self,
        config: UniversalTokenOptimizationProofConfig,
        *,
        output_directory: Path | None = None,
        run_id: str | None = None,
        persist_artifacts: bool = True,
    ) -> UniversalProofRunResult:
        run_id = _safe_id(run_id or self._run_id_factory(), "INVALID_RUN_ID")
        started_at = self._clock()
        composition = self._compose(config)
        results: list[UniversalProofCaseResult] = []
        for index, case in enumerate(config.cases):
            if (
                config.pipeline.failure_policy == "fail_fast"
                and results
                and results[-1].status == "failed"
            ):
                results.append(
                    UniversalProofCaseResult(
                        case_id=case.case_id,
                        status="skipped",
                        router_status=None,
                        router_reason=None,
                        selected_configuration_id=None,
                        pipeline_status="not_started",
                        error_reason_code="FAIL_FAST_AFTER_CASE_FAILURE",
                        protected_region_evidence=_protected_region_evidence(
                            case.request,
                            None,
                        ),
                    )
                )
                continue
            try:
                results.append(
                    self._run_case(
                        config=config,
                        case=case,
                        composition=composition,
                        run_id=run_id,
                    )
                )
            except ProofError as exc:
                results.append(
                    UniversalProofCaseResult(
                        case_id=case.case_id,
                        status="failed",
                        router_status=None,
                        router_reason=None,
                        selected_configuration_id=None,
                        pipeline_status="not_started",
                        applied_layer_ids=(),
                        error_reason_code=exc.reason_code,
                        protected_region_evidence=_protected_region_evidence(
                            case.request,
                            None,
                        ),
                    )
                )
            except Exception as exc:
                raise ProofExecutionError("CASE_EXECUTION_FAILED") from exc
        completed_count = sum(case.status == "completed" for case in results)
        failed_count = len(results) - completed_count
        completed_at = self._clock()
        result = UniversalProofRunResult(
            schema_version=SCHEMA_VERSION,
            proof_id=config.proof_id,
            run_id=run_id,
            run_mode=config.run_mode,
            started_at=started_at,
            completed_at=completed_at,
            adapter_id=config.adapter.adapter_id,
            model=config.adapter.model,
            case_count=len(results),
            completed_count=completed_count,
            failed_count=failed_count,
            cases=tuple(results),
            environment=UniversalProofEnvironmentSummary(
                provider=config.adapter.provider,
                model=config.adapter.model,
                adapter_available=True,
                network_required=config.run_mode == "live_adapter",
            ),
            artifact_manifest=UniversalProofArtifactManifest(files=()),
            success=failed_count == 0 and bool(results),
        )
        if persist_artifacts:
            manifest = self._artifact_writer(
                result,
                output_directory=output_directory or config.output.directory,
                fail_if_exists=config.output.fail_if_exists,
            )
            result = replace(result, artifact_manifest=manifest)
        return result


__all__ = [
    "UniversalTokenOptimizationProofRunner",
    "write_universal_proof_artifacts",
]
