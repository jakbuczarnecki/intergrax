#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-29.2 — Plane C vision inference on product worker pools."""

from __future__ import annotations

import sys

from intergrax.applications._shared.modality_product_worker_wiring import (
    resolve_modality_product_worker_wiring,
    resolve_product_modality_execution_profile,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.model_inference.execution.profile import ModalityExecutionMode
from intergrax.runtime.modality.modality_profile import ModalityPlane


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_modality_product_worker_wiring(env)
    if not wiring.enabled:
        print("product host must enable modality worker pool", file=sys.stderr)
        return 1
    if wiring.execution_mode is not ModalityExecutionMode.THREAD_POOL:
        print("product worker pool must use THREAD_POOL execution", file=sys.stderr)
        return 1
    if env.modality_profile is None:
        print("product host must declare modality_profile", file=sys.stderr)
        return 1
    if ModalityPlane.DEDICATED_INFERENCE not in env.modality_profile.allowed_planes:
        print("product modality profile must allow dedicated inference plane", file=sys.stderr)
        return 1
    if not env.modality_profile.require_deterministic_cv:
        print("product modality profile must require deterministic CV", file=sys.stderr)
        return 1

    execution = resolve_product_modality_execution_profile(env)
    if execution.mode is not ModalityExecutionMode.THREAD_POOL:
        print("product modality execution profile must target worker pool", file=sys.stderr)
        return 1

    print(f"OK: modality product worker pool (provider={wiring.vision_provider.value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
