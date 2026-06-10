#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-29.1 — live Triton / HF inference endpoint resolver gate."""

from __future__ import annotations

import sys

from intergrax.applications._shared.modality_production_resolver import (
    live_modality_endpoints_configured,
    resolve_live_vision_profile,
)
from intergrax.model_inference.registry.vision_provider import VisionProvider


def main() -> int:
    profile = resolve_live_vision_profile()
    if profile.provider not in (
        VisionProvider.OPENCV,
        VisionProvider.TRITON,
        VisionProvider.HUGGINGFACE_INFERENCE,
        VisionProvider.STUB,
        VisionProvider.YOLO_ULTRALYTICS,
    ):
        print(f"unsupported vision provider: {profile.provider}", file=sys.stderr)
        return 1

    if live_modality_endpoints_configured():
        if profile.provider not in (VisionProvider.TRITON, VisionProvider.HUGGINGFACE_INFERENCE):
            print("live endpoint env requires TRITON or HUGGINGFACE_INFERENCE provider", file=sys.stderr)
            return 1

    adapter = profile.create_adapter()
    if not adapter.slug:
        print("vision adapter slug must be non-empty", file=sys.stderr)
        return 1

    print(f"OK: modality live endpoints (provider={profile.provider.value}, slug={adapter.slug})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
