# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic fingerprints for idempotent knowledge evolution (SELF-HEALING R5.4)."""

from __future__ import annotations

import hashlib

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


def build_experience_set_fingerprint(
    experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
) -> str:
    if not experiences:
        return "sh_exp_fp_empty"
    ordered_ids = sorted(experience.experience_id for experience in experiences)
    digest = hashlib.sha256(",".join(ordered_ids).encode("utf-8")).hexdigest()
    return f"sh_exp_fp_{digest}"


__all__ = ["build_experience_set_fingerprint"]
