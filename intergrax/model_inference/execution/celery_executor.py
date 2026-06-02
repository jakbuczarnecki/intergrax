# © Artur Czarnecki. All rights reserved.

"""Celery-backed modality executor with in-process fallback."""

from __future__ import annotations

from intergrax.model_inference.contracts import (
    InferenceRequest,
    InferenceResult,
    ModelArtifact,
    ModelInferenceAdapter,
    VisionInferenceAdapter,
    VisionInferenceRequest,
    VisionInferenceResult,
    VisionOcrResult,
    VisionSegmentationResult,
)
from intergrax.model_inference.execution.executor import ModalityInferenceExecutor
from intergrax.model_inference.execution.jobs import (
    ModalityDetectJob,
    ModalityJobResult,
    ModalityOcrJob,
    ModalityPredictJob,
    ModalitySegmentJob,
)
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.execution.thread_pool_executor import ThreadPoolModalityInferenceExecutor
from intergrax.model_inference.registry import ModelInferenceRegistry
from intergrax.model_inference.workers.celery_app import MODALITY_CELERY_TASK_NAME, get_modality_celery_app


class CeleryModalityInferenceExecutor(ModalityInferenceExecutor):
    """
    Dispatches heavy adapter slugs to Celery when a broker is configured.

    Falls back to ``ThreadPoolModalityInferenceExecutor`` when broker is missing or dispatch fails.
    """

    def __init__(
        self,
        *,
        profile: ModalityExecutionProfile,
        fallback: ModalityInferenceExecutor | None = None,
    ) -> None:
        self._profile = profile
        self._fallback = fallback or ThreadPoolModalityInferenceExecutor(profile=profile)
        self._celery_app = get_modality_celery_app(
            broker_url=profile.celery_broker_url,
            task_always_eager=profile.celery_task_always_eager,
        )

    def _should_dispatch_celery(self, adapter_slug: str) -> bool:
        return (
            self._profile.mode == ModalityExecutionMode.CELERY
            and adapter_slug in self._profile.heavy_adapter_slugs
            and self._celery_app is not None
        )

    def _dispatch(self, job_json: str) -> ModalityJobResult:
        if self._celery_app is None:
            raise RuntimeError("Celery broker is not configured")
        task = self._celery_app.tasks.get(MODALITY_CELERY_TASK_NAME)
        if task is None:
            raise RuntimeError(f"Celery task not registered: {MODALITY_CELERY_TASK_NAME}")
        raw = task.apply(args=[job_json]).get(timeout=self._profile.celery_result_timeout_s)
        return ModalityJobResult.model_validate_json(raw)

    def run_detect(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionInferenceResult:
        if self._should_dispatch_celery(adapter.slug):
            try:
                job = ModalityDetectJob(adapter_slug=adapter.slug, artifact_id=artifact.artifact_id, request=request)
                result = self._dispatch(job.model_dump_json())
                if result.detect is not None:
                    return result.detect
                if result.error:
                    raise RuntimeError(result.error)
            except Exception:
                pass
        return self._fallback.run_detect(registry=registry, adapter=adapter, artifact=artifact, request=request)

    def run_segment(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionSegmentationResult:
        if self._should_dispatch_celery(adapter.slug):
            try:
                job = ModalitySegmentJob(adapter_slug=adapter.slug, artifact_id=artifact.artifact_id, request=request)
                result = self._dispatch(job.model_dump_json())
                if result.segment is not None:
                    return result.segment
                if result.error:
                    raise RuntimeError(result.error)
            except Exception:
                pass
        return self._fallback.run_segment(registry=registry, adapter=adapter, artifact=artifact, request=request)

    def run_ocr_regions(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionOcrResult:
        if self._should_dispatch_celery(adapter.slug):
            try:
                job = ModalityOcrJob(adapter_slug=adapter.slug, artifact_id=artifact.artifact_id, request=request)
                result = self._dispatch(job.model_dump_json())
                if result.ocr is not None:
                    return result.ocr
                if result.error:
                    raise RuntimeError(result.error)
            except Exception:
                pass
        return self._fallback.run_ocr_regions(
            registry=registry,
            adapter=adapter,
            artifact=artifact,
            request=request,
        )

    def run_predict(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: ModelInferenceAdapter,
        artifact: ModelArtifact,
        request: InferenceRequest,
    ) -> InferenceResult:
        return self._fallback.run_predict(registry=registry, adapter=adapter, artifact=artifact, request=request)
