# © Artur Czarnecki. All rights reserved.

"""ROADMAP-REPLAY-X L2 corrective wave — scoped production delta dispositions @ 91649e5 publication lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

REPLAY_X_L2_QUALIFICATION_HEAD: Final[str] = "91649e5d883664a855c02dea1baf14d467052fca"


@dataclass(frozen=True, slots=True)
class ProdDeltaDisposition:
    path: str
    disposition: str
    regression_evidence: str


REPLAY_X_L2_SCOPED_PROD_DELTA: Final[tuple[ProdDeltaDisposition, ...]] = (
    ProdDeltaDisposition(
        path="intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        disposition="CTRL-X pyright callback typing + ruff format; admission/shutdown semantics unchanged",
        regression_evidence="tests/qualification/control_planes/ctrl_x/test_ctrl_x_typing_gates.py",
    ),
)

REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS: Final[frozenset[str]] = frozenset(
    row.path for row in REPLAY_X_L2_SCOPED_PROD_DELTA
)
