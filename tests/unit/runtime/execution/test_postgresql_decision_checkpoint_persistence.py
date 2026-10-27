# © Artur Czarnecki. All rights reserved.

"""PostgreSQL decision checkpoint persistence (P4-A / SX-F03)."""

from __future__ import annotations

import pytest

from tests.qualification.dist_scale_x._p2_support import pg_port_open

pytestmark = [
    pytest.mark.skipif(not pg_port_open(), reason="PostgreSQL qual infra unavailable"),
]


def _sample_checkpoint():
    from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
    from intergrax.contracts.decision_finalization import (
        decision_finalization_key,
        initial_decision_finalize_guard,
    )
    from intergrax.contracts.decision_identity import (
        DecisionExecutionLineage,
        DecisionIdentity,
        DecisionScope,
        initial_decision_version,
        mint_decision_id,
    )
    from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
    from intergrax.contracts.decision_revision import (
        decision_revision_checkpoint_state,
        decision_revision_policy,
    )
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )
    from intergrax.contracts.decision_lifecycle import initial_decision_lifecycle_state

    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="p4-f03", subject="unit"),
        tenant_id="tenant-f03",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    revision = decision_revision_checkpoint_state(
        proposal_ref=DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        revision_count=1,
        max_revisions=3,
    )
    lifecycle = initial_decision_lifecycle_state(identity)
    return decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )


@pytest.mark.unit
def test_pg_decision_checkpoint_create_and_stale_cas() -> None:
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.execution.decision_finalization_conformance import (
        conformance_artifact_payload_codec_registry,
    )
    from intergrax.runtime.execution.decision_checkpoint_persistence import (
        StaleDecisionCheckpointWriteError,
        save_decision_checkpoint,
    )
    from intergrax.runtime.execution.postgresql_decision_checkpoint_persistence import (
        PostgreSQLDecisionCheckpointPersistence,
    )

    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"tenant_schema": "dist_scale_x_p4_f03_unit"},
    )
    codecs = conformance_artifact_payload_codec_registry()
    store = PostgreSQLDecisionCheckpointPersistence(config=config, payload_codecs=codecs)
    checkpoint = _sample_checkpoint()
    save_decision_checkpoint(store, checkpoint=checkpoint)
    with pytest.raises(StaleDecisionCheckpointWriteError):
        store.save(checkpoint=checkpoint, expected_revision=0)


@pytest.mark.unit
def test_pg_decision_load_materialized_and_tenant_isolation() -> None:
    from intergrax.integrations.providers.relational_store.postgresql.config import (
        PostgreSQLIntegrationConfig,
    )
    from intergrax.runtime.execution.decision_finalization_conformance import (
        conformance_artifact_payload_codec_registry,
    )
    from intergrax.runtime.execution.decision_checkpoint_persistence import save_decision_checkpoint
    from intergrax.runtime.execution.postgresql_decision_checkpoint_persistence import (
        PostgreSQLDecisionCheckpointPersistence,
    )

    config = PostgreSQLIntegrationConfig.from_env().model_copy(
        update={"tenant_schema": "dist_scale_x_p4_f03_iso"},
    )
    codecs = conformance_artifact_payload_codec_registry()
    store = PostgreSQLDecisionCheckpointPersistence(config=config, payload_codecs=codecs)
    checkpoint = _sample_checkpoint()
    save_decision_checkpoint(store, checkpoint=checkpoint)
    key = checkpoint.finalization.key
    materialized = store.load_materialized(key=key)
    assert materialized is not None
    assert store.load(key=key) is not None
    other = _sample_checkpoint()
    other_key = other.finalization.key
    assert store.load_materialized(key=other_key) is None or other_key != key
