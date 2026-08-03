# © Artur Czarnecki. All rights reserved.

"""Shared validators for Workspace Knowledge Configuration mutations."""

from __future__ import annotations

import re

from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationError,
)

_IDEMPOTENCY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_configuration_idempotency_hash(value: object) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_HASH_RE.fullmatch(value) is None:
        raise WorkspaceKnowledgeConfigurationMutationError(
            "knowledge_configuration_idempotency_hash_invalid"
        )
    return value
