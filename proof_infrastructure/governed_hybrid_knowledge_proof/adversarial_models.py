# © Artur Czarnecki. All rights reserved.

"""Typed adversarial proof result contracts for COMM-5E."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AdversarialDefenseLayerV1(StrEnum):
    PLAN_REJECTED = "PLAN_REJECTED"
    AUTHORITY_DENIED = "AUTHORITY_DENIED"
    PROVIDER_REJECTED = "PROVIDER_REJECTED"
    EVIDENCE_UNSATISFIED = "EVIDENCE_UNSATISFIED"
    SYNTHESIS_BLOCKED = "SYNTHESIS_BLOCKED"
    NOT_REACHABLE_BY_CONTRACT = "NOT_REACHABLE_BY_CONTRACT"


class AdversarialAttackIdV1(StrEnum):
    A_REQUIRED_LIVE_MISSING = "A"
    B_MIDFLIGHT_REVOKE = "B"
    C_WRONG_CONNECTION_PROVIDER = "C"
    D_WRONG_TENANT = "D"
    E_WRONG_WORKSPACE = "E"
    F_MALFORMED_PROVIDER_PAYLOAD = "F"
    G_PROVIDER_404_5XX = "G"
    H_CALLER_DOWNGRADE = "H"
    I_STALE_PLAN = "I"
    J_CONNECTION_DISABLED = "J"
    K_CAPABILITY_MISMATCH = "K"
    L_EPHEMERAL_LEAK = "L"
    M_HISTORICAL_IMMUTABILITY = "M"
    N_WRONG_CALL_EVIDENCE = "N"
    O_DUPLICATE_REPLAY_EVIDENCE = "O"


class AdversarialAttackResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attack_id: AdversarialAttackIdV1
    reachable: bool
    defense_layer: AdversarialDefenseLayerV1
    http_calls: int
    llm_calls: int
    passed: bool
    notes: str | None = Field(default=None, max_length=512)
