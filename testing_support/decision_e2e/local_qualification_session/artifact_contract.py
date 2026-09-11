# © Artur Czarnecki. All rights reserved.

"""Central canonical artifact set for local qualification sessions."""

from __future__ import annotations

from testing_support.decision_e2e.local_qualification_session.artifact_finalization_contract import (
    QualificationArtifactFinalizationContract,
)

DEFAULT_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    QualificationArtifactFinalizationContract.required_artifact_names()
)
