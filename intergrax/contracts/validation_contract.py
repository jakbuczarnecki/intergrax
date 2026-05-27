# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Multi-stage validation contracts (architecture §42.16)."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.validation import ValidationResult


class ValidationScope(str, Enum):
    STEP = "step"
    AGENT = "agent"
    NODE = "node"
    TASK = "task"


class ValidationSeverity(str, Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationRule(BaseModel):
    rule_id: str
    description: str = ""
    severity: ValidationSeverity = ValidationSeverity.ERROR
    evaluator: str = ""


class ValidationContract(BaseModel):
    validation_id: str
    scope: ValidationScope = ValidationScope.AGENT
    rules: List[ValidationRule] = Field(default_factory=list)
    on_failure: AgentDecisionType = AgentDecisionType.FAIL
    schema_version: str = "validation_contract.v1"


class ExtendedValidationResult(ValidationResult):
    stage: str = ""
    validator_id: str = ""
    scope: ValidationScope = ValidationScope.AGENT
