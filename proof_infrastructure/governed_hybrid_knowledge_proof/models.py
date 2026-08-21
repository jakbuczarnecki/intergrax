# © Artur Czarnecki. All rights reserved.

"""Typed proof result contracts for the governed hybrid knowledge flagship proof."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SemanticDecisionV1(StrEnum):
    YES = "YES"
    NO = "NO"
    CANNOT_DETERMINE = "CANNOT DETERMINE"


class FlagshipScenarioChecksV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: bool = True
    http_reads: bool = True
    llm_calls: bool = True
    admissibility: bool = True
    ask_status: bool = True
    indexed_evidence_present: bool | None = None
    live_evidence_present: bool | None = None
    ephemeral_body_not_durable: bool | None = None
    indexed_policy_present: bool | None = None
    same_configuration_revision: bool | None = None
    same_plan_policy_contract: bool | None = None
    binding_disabled_before_live: bool | None = None
    indexed_only_insufficient: bool | None = None


class FlagshipScenarioMetricsV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checks: FlagshipScenarioChecksV1 | None = None
    policy_revision: int | None = None
    plan_id: str | None = None
    revoke_boundary: str | None = None
    configuration_revision: int | None = None
    indexed_evidence_id: str | None = None
    live_content_hash: str | None = None
    live_binding_id: str | None = None
    historical_live_body_retained: bool | None = None
    structural_provenance_only: bool | None = None
    configuration_revision_before_disable: int | None = None
    configuration_revision_after_disable: int | None = None


class FlagshipProofScenarioResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    passed: bool
    expected: SemanticDecisionV1 | str
    observed: SemanticDecisionV1 | str
    http_read_count: int | None = None
    llm_call_count: int | None = None
    admissibility: str | None = None
    ask_status: str | None = None
    run_id: str | None = None
    key_metrics: FlagshipScenarioMetricsV1 = Field(default_factory=FlagshipScenarioMetricsV1)


class FlagshipProofResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_1: FlagshipProofScenarioResultV1
    scenario_2: FlagshipProofScenarioResultV1
    scenario_3: FlagshipProofScenarioResultV1
    scenario_4: FlagshipProofScenarioResultV1
    overall_status: str
    scenario_1_run_id: str | None = None

    @property
    def all_passed(self) -> bool:
        return all(
            scenario.passed
            for scenario in (
                self.scenario_1,
                self.scenario_2,
                self.scenario_3,
                self.scenario_4,
            )
        )

    @property
    def passed_count(self) -> int:
        return sum(
            1
            for scenario in (
                self.scenario_1,
                self.scenario_2,
                self.scenario_3,
                self.scenario_4,
            )
            if scenario.passed
        )
