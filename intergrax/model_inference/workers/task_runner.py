# © Artur Czarnecki. All rights reserved.

"""In-worker execution of serialized modality jobs."""

from __future__ import annotations

import json

from intergrax.model_inference.adapters.stub_vision import StubVisionInferenceAdapter
from intergrax.model_inference.bootstrap import build_harness_model_inference_registry
from intergrax.model_inference.contracts import ExtendedVisionInferenceAdapter
from intergrax.model_inference.execution.in_process_executor import InProcessModalityInferenceExecutor
from intergrax.model_inference.execution.jobs import (
    ModalityDetectJob,
    ModalityJobKind,
    ModalityJobResult,
    ModalityOcrJob,
    ModalityPredictJob,
    ModalitySegmentJob,
)


def run_modality_job(payload_json: str) -> str:
    """Execute a modality job JSON payload and return ``ModalityJobResult`` JSON."""
    raw = json.loads(payload_json)
    kind = ModalityJobKind(raw["kind"])
    if kind == ModalityJobKind.DETECT:
        return _execute_detect(ModalityDetectJob.model_validate(raw)).model_dump_json()
    if kind == ModalityJobKind.SEGMENT:
        return _execute_segment(ModalitySegmentJob.model_validate(raw)).model_dump_json()
    if kind == ModalityJobKind.OCR_REGIONS:
        return _execute_ocr(ModalityOcrJob.model_validate(raw)).model_dump_json()
    return _execute_predict(ModalityPredictJob.model_validate(raw)).model_dump_json()


def _execute_detect(job: ModalityDetectJob) -> ModalityJobResult:
    return _run_vision(lambda executor, registry, adapter, artifact: ModalityJobResult(
        detect=executor.run_detect(registry=registry, adapter=adapter, artifact=artifact, request=job.request)
    ), job.adapter_slug, job.artifact_id)


def _execute_segment(job: ModalitySegmentJob) -> ModalityJobResult:
    def _run(executor, registry, adapter, artifact):
        extended = adapter if isinstance(adapter, ExtendedVisionInferenceAdapter) else StubVisionInferenceAdapter()
        return ModalityJobResult(
            segment=executor.run_segment(
                registry=registry,
                adapter=extended,
                artifact=artifact,
                request=job.request,
            )
        )

    return _run_vision(_run, job.adapter_slug, job.artifact_id)


def _execute_ocr(job: ModalityOcrJob) -> ModalityJobResult:
    def _run(executor, registry, adapter, artifact):
        extended = adapter if isinstance(adapter, ExtendedVisionInferenceAdapter) else StubVisionInferenceAdapter()
        return ModalityJobResult(
            ocr=executor.run_ocr_regions(
                registry=registry,
                adapter=extended,
                artifact=artifact,
                request=job.request,
            )
        )

    return _run_vision(_run, job.adapter_slug, job.artifact_id)


def _execute_predict(job: ModalityPredictJob) -> ModalityJobResult:
    registry = build_harness_model_inference_registry()
    executor = InProcessModalityInferenceExecutor()
    try:
        artifact = registry.get_artifact(job.artifact_id)
        adapter = registry.get_ml_adapter(job.adapter_slug)
        predict = executor.run_predict(
            registry=registry,
            adapter=adapter,
            artifact=artifact,
            request=job.request,
        )
        return ModalityJobResult(predict=predict)
    except Exception as exc:
        return ModalityJobResult(error=str(exc))


def _run_vision(callback, adapter_slug: str, artifact_id: str) -> ModalityJobResult:
    registry = build_harness_model_inference_registry()
    executor = InProcessModalityInferenceExecutor()
    try:
        artifact = registry.get_artifact(artifact_id)
        adapter = registry.get_vision_adapter(adapter_slug)
        return callback(executor, registry, adapter, artifact)
    except Exception as exc:
        return ModalityJobResult(error=str(exc))
