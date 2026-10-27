# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""PostgreSQL runtime persistence openers (DIST-SCALE-X P4-A)."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
)
from intergrax.runtime.execution.decision_artifact_payload_codec import (
    DecisionArtifactPayloadCodecRegistry,
)
from intergrax.runtime.execution.postgresql_decision_checkpoint_persistence import (
    PostgreSQLDecisionCheckpointPersistence,
)
from intergrax.runtime.long_running.postgresql_task_checkpoint_store import (
    PostgreSQLTaskCheckpointStore,
)


def open_postgresql_task_checkpoint_store(
    *,
    config: PostgreSQLIntegrationConfig | None = None,
    provider: PostgreSQLConnectionProvider | None = None,
) -> PostgreSQLTaskCheckpointStore:
    return PostgreSQLTaskCheckpointStore(config=config, provider=provider)


def open_postgresql_decision_checkpoint_store(
    *,
    config: PostgreSQLIntegrationConfig | None = None,
    provider: PostgreSQLConnectionProvider | None = None,
    payload_codecs: DecisionArtifactPayloadCodecRegistry,
) -> PostgreSQLDecisionCheckpointPersistence:
    return PostgreSQLDecisionCheckpointPersistence(
        config=config,
        provider=provider,
        payload_codecs=payload_codecs,
    )
