"""Qualification outcome classification for VPI embedding performance evidence."""

from __future__ import annotations

from enum import Enum


class VpiEmbeddingQualificationStatus(str, Enum):
    PASS = "PASS"
    PARTIAL_PASS_GPU = "PARTIAL_PASS_GPU"
    BLOCKED_GPU = "BLOCKED_GPU"
    BLOCKED_STORAGE_ENVIRONMENT = "BLOCKED_STORAGE_ENVIRONMENT"
    FAILED_CORRECTNESS = "FAILED_CORRECTNESS"
