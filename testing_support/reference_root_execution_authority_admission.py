# © Artur Czarnecki. All rights reserved.

"""Reference/test root execution authority admission for worker and lab composition tests."""

from __future__ import annotations

from intergrax.runtime.governance.execution_admission_composition import (
    build_reference_allowing_root_execution_authority_admission,
)

REFERENCE_ROOT_EXECUTION_AUTHORITY_ADMISSION = (
    build_reference_allowing_root_execution_authority_admission()
)

__all__ = [
    "REFERENCE_ROOT_EXECUTION_AUTHORITY_ADMISSION",
    "build_reference_allowing_root_execution_authority_admission",
]
