# © Artur Czarnecki. All rights reserved.

"""Vision inference provider slugs (mirrors ``LLMProvider``)."""

from __future__ import annotations

from enum import Enum


class VisionProvider(str, Enum):
    """Harness-registered vision inference backends (Plane C)."""

    STUB = "stub"
    OPENCV = "onnxruntime"
    YOLO_ULTRALYTICS = "yolo_ultralytics"
    TRITON = "vision_serving"
    HUGGINGFACE_INFERENCE = "huggingface_inference"
