# © Artur Czarnecki. All rights reserved.

"""Background thread-pool executor for heavy modality inference."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

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
from intergrax.model_inference.execution.in_process_executor import InProcessModalityInferenceExecutor
from intergrax.model_inference.execution.profile import ModalityExecutionMode, ModalityExecutionProfile
from intergrax.model_inference.registry import ModelInferenceRegistry

T = TypeVar("T")


class ThreadPoolModalityInferenceExecutor(ModalityInferenceExecutor):
    """
    Offloads jobs whose adapter slug is listed in ``ModalityExecutionProfile.heavy_adapter_slugs``.

    Falls back to in-process execution for lightweight adapters (e.g. OpenCV contours).
    """

    def __init__(
        self,
        *,
        profile: ModalityExecutionProfile,
        delegate: ModalityInferenceExecutor | None = None,
        pool: ThreadPoolExecutor | None = None,
    ) -> None:
        self._profile = profile
        self._delegate = delegate or InProcessModalityInferenceExecutor()
        self._pool = pool or ThreadPoolExecutor(
            max_workers=profile.max_workers,
            thread_name_prefix="modality-inference",
        )
        self._owns_pool = pool is None

    def close(self) -> None:
        if self._owns_pool:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def _should_offload(self, adapter_slug: str) -> bool:
        return (
            self._profile.mode == ModalityExecutionMode.THREAD_POOL
            and adapter_slug in self._profile.heavy_adapter_slugs
        )

    def _submit(self, adapter_slug: str, fn: Callable[[], T]) -> T:
        if not self._should_offload(adapter_slug):
            return fn()
        future: Future[T] = self._pool.submit(fn)
        return future.result()

    def run_detect(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionInferenceResult:
        return self._submit(
            adapter.slug,
            lambda: self._delegate.run_detect(
                registry=registry,
                adapter=adapter,
                artifact=artifact,
                request=request,
            ),
        )

    def run_segment(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionSegmentationResult:
        return self._submit(
            adapter.slug,
            lambda: self._delegate.run_segment(
                registry=registry,
                adapter=adapter,
                artifact=artifact,
                request=request,
            ),
        )

    def run_ocr_regions(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: VisionInferenceAdapter,
        artifact: ModelArtifact,
        request: VisionInferenceRequest,
    ) -> VisionOcrResult:
        return self._submit(
            adapter.slug,
            lambda: self._delegate.run_ocr_regions(
                registry=registry,
                adapter=adapter,
                artifact=artifact,
                request=request,
            ),
        )

    def run_predict(
        self,
        *,
        registry: ModelInferenceRegistry,
        adapter: ModelInferenceAdapter,
        artifact: ModelArtifact,
        request: InferenceRequest,
    ) -> InferenceResult:
        return self._submit(
            adapter.slug,
            lambda: self._delegate.run_predict(
                registry=registry,
                adapter=adapter,
                artifact=artifact,
                request=request,
            ),
        )
