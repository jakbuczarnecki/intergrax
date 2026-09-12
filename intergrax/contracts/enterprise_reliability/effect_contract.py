# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External effect safety declarations (ERL Phase 2 — contracts foundation)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NON_EMPTY = Field(min_length=1)


class ExternalEffectContractValidationError(ValueError):
    """Declared safety capabilities are internally inconsistent."""


SCHEMA_EXTERNAL_EFFECT_CONTRACT_V1: Final = "external_effect_contract.v1"


class ExternalEffectCategory(StrEnum):
    """Coarse domain of external state change — provider-neutral."""

    FINANCIAL = "financial"
    INVENTORY = "inventory"
    COMMUNICATION = "communication"
    INFRASTRUCTURE = "infrastructure"


class ExternalEffectCapabilitySupport(StrEnum):
    """Whether a safety dimension is declared for an operation."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


class ExternalEffectSafetyCapabilities(BaseModel):
    """Declared safety guarantees — mechanisms are implemented in later ERL phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    idempotency: ExternalEffectCapabilitySupport
    reconciliation: ExternalEffectCapabilitySupport
    compensation: ExternalEffectCapabilitySupport


class ExternalEffectContract(BaseModel):
    """Platform contract describing one external mutating or observable operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external_effect_contract.v1"] = SCHEMA_EXTERNAL_EFFECT_CONTRACT_V1
    contract_id: str = _NON_EMPTY
    operation_key: str = _NON_EMPTY
    category: ExternalEffectCategory
    safety: ExternalEffectSafetyCapabilities
    reconciliation_probe_refs: tuple[str, ...] = ()
    compensation_operation_ref: str | None = None

    @field_validator("contract_id", "operation_key")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("reconciliation_probe_refs")
    @classmethod
    def _normalize_probe_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for ref in value:
            stripped = ref.strip()
            if not stripped:
                raise ValueError("reconciliation probe ref must be non-empty")
            normalized.append(stripped)
        return tuple(normalized)

    @field_validator("compensation_operation_ref")
    @classmethod
    def _strip_optional_compensation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_capability_consistency(self) -> ExternalEffectContract:
        if (
            self.safety.reconciliation is ExternalEffectCapabilitySupport.SUPPORTED
            and not self.reconciliation_probe_refs
        ):
            raise ValueError(
                "reconciliation_probe_refs required when reconciliation is supported",
            )
        if self.safety.reconciliation is ExternalEffectCapabilitySupport.NOT_SUPPORTED:
            if self.reconciliation_probe_refs:
                raise ValueError(
                    "reconciliation_probe_refs forbidden when reconciliation is not supported",
                )
        if self.safety.compensation is ExternalEffectCapabilitySupport.SUPPORTED:
            if self.compensation_operation_ref is None:
                raise ValueError(
                    "compensation_operation_ref required when compensation is supported",
                )
        if self.safety.compensation is ExternalEffectCapabilitySupport.NOT_SUPPORTED:
            if self.compensation_operation_ref is not None:
                raise ValueError(
                    "compensation_operation_ref forbidden when compensation is not supported",
                )
        return self


class UnknownUncertaintyPosture(StrEnum):
    """Contract-derived posture while ``ExternalEffectOutcome`` is UNKNOWN."""

    RECONCILE_OR_IDEMPOTENT_REPEAT = "reconcile_or_idempotent_repeat"
    RECONCILE_ONLY = "reconcile_only"
    ESCALATE_REQUIRED = "escalate_required"


def contract_declares_idempotency(contract: ExternalEffectContract) -> bool:
    """Whether UNKNOWN may treat repetition as safe per declaration (not execution)."""
    return contract.safety.idempotency is ExternalEffectCapabilitySupport.SUPPORTED


def contract_declares_reconciliation(contract: ExternalEffectContract) -> bool:
    return contract.safety.reconciliation is ExternalEffectCapabilitySupport.SUPPORTED


def contract_declares_compensation(contract: ExternalEffectContract) -> bool:
    return contract.safety.compensation is ExternalEffectCapabilitySupport.SUPPORTED


def evaluate_unknown_uncertainty_posture(
    contract: ExternalEffectContract,
) -> UnknownUncertaintyPosture:
    """
    Map declared capabilities to UNKNOWN handling posture.

    Does not execute reconcile, retry, or escalation — only classifies allowed paths.
    """
    if contract_declares_idempotency(contract):
        return UnknownUncertaintyPosture.RECONCILE_OR_IDEMPOTENT_REPEAT
    if contract_declares_reconciliation(contract):
        return UnknownUncertaintyPosture.RECONCILE_ONLY
    return UnknownUncertaintyPosture.ESCALATE_REQUIRED


def validate_external_effect_contract(
    contract: ExternalEffectContract,
) -> ExternalEffectContract:
    """Re-validate a contract instance; raises on inconsistency."""
    try:
        return ExternalEffectContract.model_validate(contract.model_dump())
    except ValueError as exc:
        raise ExternalEffectContractValidationError(str(exc)) from exc


__all__ = [
    "ExternalEffectCapabilitySupport",
    "ExternalEffectCategory",
    "ExternalEffectContract",
    "ExternalEffectContractValidationError",
    "ExternalEffectSafetyCapabilities",
    "SCHEMA_EXTERNAL_EFFECT_CONTRACT_V1",
    "UnknownUncertaintyPosture",
    "contract_declares_compensation",
    "contract_declares_idempotency",
    "contract_declares_reconciliation",
    "evaluate_unknown_uncertainty_posture",
    "validate_external_effect_contract",
]
