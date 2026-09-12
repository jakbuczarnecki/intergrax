"""Scenario-owned ERL reconciliation identifiers — generic at the plugin boundary."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
)

SCENARIO_RECONCILIATION_PLUGIN_ID = "erl-qual-004-external-reality-reconciliation"
SCENARIO_RECONCILIATION_PLUGIN_VERSION = "1.0.0"
SCENARIO_RECONCILIATION_PLUGIN_OWNER = "erl-qual-004-scenario"

EXTERNAL_EFFECT_SOR_PROBE_REF = "external_effect.system_of_record_read"

SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID = "erl-qual-004-external-effect-capture"


def scenario_external_effect_contract() -> ExternalEffectContract:
    """Contract for UNKNOWN capture episodes — probe ref is domain-neutral."""
    return ExternalEffectContract(
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        operation_key="external_effect.capture",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=(EXTERNAL_EFFECT_SOR_PROBE_REF,),
    )
