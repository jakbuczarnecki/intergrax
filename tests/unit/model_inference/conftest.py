# © Artur Czarnecki. All rights reserved.

"""Shared vision golden fixture helpers for modality unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

GOLDEN_VISION_IMAGE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "vision_golden" / "sample_target.png"
)


def ensure_vision_golden_fixture() -> Path:
    """Create canonical golden image when missing (white rectangle on black)."""
    if GOLDEN_VISION_IMAGE.is_file():
        return GOLDEN_VISION_IMAGE
    from intergrax.model_inference.opencv_availability import opencv_runtime_available

    if not opencv_runtime_available():
        return GOLDEN_VISION_IMAGE
    import cv2
    import numpy as np

    GOLDEN_VISION_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 80), 255, -1)
    cv2.imwrite(str(GOLDEN_VISION_IMAGE), image)
    return GOLDEN_VISION_IMAGE


@pytest.fixture(scope="module")
def vision_golden_image() -> Path:
    path = ensure_vision_golden_fixture()
    if not path.is_file():
        pytest.skip("vision golden fixture unavailable")
    return path
