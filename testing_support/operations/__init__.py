# © Artur Czarnecki. All rights reserved.

"""EE-B4-C runbook and incident validation support (not runtime authority)."""

from testing_support.operations.forbidden_patterns import FORBIDDEN_RUNBOOK_PATTERNS
from testing_support.operations.incident_taxonomy import (
    EXECUTION_ENGINE_INCIDENT_CATALOG,
    REQUIRED_RUNBOOK_IDS,
    ExecutionIncidentDescriptor,
    IncidentCategoryId,
    OperationalSeverity,
)
from testing_support.operations.runbook_contract import RUNBOOK_REQUIRED_SECTIONS
from testing_support.operations.runbook_validator import (
    validate_runbook_document,
)

__all__ = [
    "EXECUTION_ENGINE_INCIDENT_CATALOG",
    "FORBIDDEN_RUNBOOK_PATTERNS",
    "REQUIRED_RUNBOOK_IDS",
    "RUNBOOK_REQUIRED_SECTIONS",
    "ExecutionIncidentDescriptor",
    "IncidentCategoryId",
    "OperationalSeverity",
    "validate_runbook_document",
]
