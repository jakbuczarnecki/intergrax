# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from testing_support.change_safety.contracts.enums import SelectionStatus


class SelectedTestNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    reason_rule_id: str


class SelectionReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    rule_id: str
    witness: str = ""


class SelectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "evol_test_selection.v1"
    decision_id: str
    status: SelectionStatus
    selected: tuple[SelectedTestNode, ...] = ()
    mandatory: tuple[SelectedTestNode, ...] = ()
    reasons: tuple[SelectionReason, ...] = ()
    graph_digest: str = ""
    inventory_digest: str = ""
    suite_collection_digest: str = ""
    mandatory_manifest_digest: str = ""
    ruleset_version: str = ""
    tool_version: str = "evol-x-0.1.0"
