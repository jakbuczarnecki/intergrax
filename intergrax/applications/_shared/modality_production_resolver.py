# © Artur Czarnecki. All rights reserved.

"""Live Triton / HuggingFace inference endpoint resolver (AUDIT-IDEAL-29.1)."""

from __future__ import annotations

import os

from intergrax.model_inference.registry.vision_profile import VisionProfile
from intergrax.model_inference.registry.vision_provider import VisionProvider


def resolve_live_vision_profile(*, prefix: str = "INTERGRAX_VISION") -> VisionProfile:
    """
    Prefer live remote serving when endpoint env vars are configured.

    - ``INTERGRAX_TRITON_URL`` (+ optional ``INTERGRAX_TRITON_MODEL``)
    - ``INTERGRAX_HF_INFERENCE_URL`` or ``HF_INFERENCE_ENDPOINT`` (+ ``HF_TOKEN``)
    """
    triton_url = os.getenv("INTERGRAX_TRITON_URL", "").strip()
    if triton_url:
        return VisionProfile(
            provider=VisionProvider.TRITON,
            options={
                "base_url": triton_url,
                "model_name": os.getenv("INTERGRAX_TRITON_MODEL", "yolo").strip() or "yolo",
            },
        )

    hf_url = (
        os.getenv("INTERGRAX_HF_INFERENCE_URL", "").strip()
        or os.getenv("HF_INFERENCE_ENDPOINT", "").strip()
    )
    if hf_url:
        token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_API_KEY", "").strip()
        return VisionProfile(
            provider=VisionProvider.HUGGINGFACE_INFERENCE,
            options={"endpoint_url": hf_url, "api_key": token},
        )

    provider_raw = os.getenv(f"{prefix}_PROVIDER", VisionProvider.OPENCV.value).strip().lower()
    return VisionProfile(provider=VisionProvider(provider_raw))


def live_modality_endpoints_configured() -> bool:
    """Return True when a remote inference endpoint env var is set."""
    return bool(
        os.getenv("INTERGRAX_TRITON_URL", "").strip()
        or os.getenv("INTERGRAX_HF_INFERENCE_URL", "").strip()
        or os.getenv("HF_INFERENCE_ENDPOINT", "").strip()
    )
