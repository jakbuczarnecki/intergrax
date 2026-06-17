# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Event taxonomy types shared without catalog/registry cycles (OBS-EVOL-9)."""

from __future__ import annotations

from enum import Enum


class EventCategory(str, Enum):
    """Derived ops grouping for subscribers and metrics cardinality control."""

    TASK = "task"
    PLAN = "plan"
    TOOL = "tool"
    AGENT = "agent"
    CONTEXT = "context"
    HUMAN = "human"
    POLICY = "policy"
    PLATFORM = "platform"


class RetentionClass(str, Enum):
    """Store retention tier aligned with data classification (IDEAL-23.5)."""

    OPERATIONAL = "operational"
    AUDIT = "audit"
    DEBUG = "debug"


def category_for_event_kind(event_kind: str) -> EventCategory:
    """Derive category from namespaced ``event_kind`` (extension path)."""
    if event_kind.startswith("agents."):
        return EventCategory.AGENT
    if event_kind.startswith("applications."):
        return EventCategory.PLATFORM
    if event_kind.startswith("platform.task."):
        return EventCategory.TASK
    if event_kind.startswith("platform.plan."):
        return EventCategory.PLAN
    if event_kind.startswith("platform.context."):
        return EventCategory.CONTEXT
    if event_kind.startswith("platform.policy."):
        return EventCategory.POLICY
    if event_kind.startswith("platform."):
        return EventCategory.PLATFORM
    if event_kind.startswith("intergrax.llm.stream."):
        return EventCategory.AGENT
    return EventCategory.PLATFORM
