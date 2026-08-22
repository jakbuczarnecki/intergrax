# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition helpers for experiment persistence backends (PBA-FIX-E)."""

from __future__ import annotations

from pathlib import Path

from intergrax.experiments.persistence_contract import ExperimentPersistence

__all__ = [
    "resolve_experiment_persistence",
]


def resolve_experiment_persistence(
    *,
    experiment_store: ExperimentPersistence | None = None,
    experiments_db: Path | None = None,
) -> ExperimentPersistence:
    """
    Resolve experiment persistence for lab workflow and debug surfaces.

    Priority: explicit implementation > SQLite at ``experiments_db`` > default SQLite path.
    """
    if experiment_store is not None:
        return experiment_store

    from intergrax.integrations.providers.relational_store.sqlite import (
        create_sqlite_experiment_store,
    )

    if experiments_db is not None:
        return create_sqlite_experiment_store(db_path=experiments_db)  # type: ignore[return-value]
    return create_sqlite_experiment_store()  # type: ignore[return-value]
