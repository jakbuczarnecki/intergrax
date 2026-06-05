# © Artur Czarnecki. All rights reserved.

"""Agent lifecycle state enum shared by contracts and governance evaluators."""

from __future__ import annotations

from enum import Enum


class AgentLifecycleState(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
