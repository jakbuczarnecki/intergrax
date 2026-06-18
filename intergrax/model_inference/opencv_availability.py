# © Artur Czarnecki. All rights reserved.

"""OpenCV runtime availability probe for modality tests and adapters."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


def opencv_runtime_available() -> bool:
    """Return True when ``cv2`` can round-trip image I/O (not a namespace stub)."""
    spec = importlib.util.find_spec("cv2")
    if spec is None or spec.loader is None:
        return False
    try:
        import cv2  # noqa: PLC0415 — lazy import after spec check
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return False

    if not callable(getattr(cv2, "imread", None)) or not callable(getattr(cv2, "imwrite", None)):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "opencv_probe.png"
        image = np.zeros((8, 8), dtype=np.uint8)
        if not cv2.imwrite(str(probe), image):
            return False
        loaded = cv2.imread(str(probe))
        return loaded is not None
