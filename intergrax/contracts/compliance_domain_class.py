# © Artur Czarnecki. All rights reserved.

"""Regulated compliance domain class — public declarative enum."""

from __future__ import annotations

from enum import Enum


class ComplianceDomainClass(str, Enum):
    REGULATED = "regulated"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
