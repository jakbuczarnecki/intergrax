# © Artur Czarnecki. All rights reserved.

"""Flagship governed hybrid knowledge proof package."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLICATIONS = _REPO_ROOT / "applications"
for path in (_REPO_ROOT, _APPLICATIONS):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from proof_infrastructure.governed_hybrid_knowledge_proof.models import (
    FlagshipProofResultV1,
    FlagshipProofScenarioResultV1,
    SemanticDecisionV1,
)
from proof_infrastructure.governed_hybrid_knowledge_proof.runner import (
    run_flagship_proof,
)

__all__ = [
    "FlagshipProofResultV1",
    "FlagshipProofScenarioResultV1",
    "SemanticDecisionV1",
    "run_flagship_proof",
]
