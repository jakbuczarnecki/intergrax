# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Skill version binding disposition for discovery projections (Stage 6)."""

from __future__ import annotations

from enum import Enum


class SkillVersionBindingDisposition(str, Enum):
    """
    Discovery-side disposition for how a skill version label relates to runtime binding.

    Catalog entries advertise catalog manifest versions with ``MATERIALIZED``.
    Runtime agent snapshots retain exact pinned/materialized evidence separately.
    """

    PINNED = "pinned"
    MATERIALIZED = "materialized"
