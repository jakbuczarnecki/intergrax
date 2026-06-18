# © Artur Czarnecki. All rights reserved.

"""OpenCV runtime availability probe for modality tests and adapters."""

from __future__ import annotations

import importlib.util


def opencv_runtime_available() -> bool:
    """Return True when ``cv2`` exposes core image I/O (not a namespace stub)."""
    spec = importlib.util.find_spec("cv2")
    if spec is None or spec.loader is None:
        return False
    import cv2  # noqa: PLC0415 — lazy import after spec check

    return callable(getattr(cv2, "imread", None)) and callable(getattr(cv2, "imwrite", None))
